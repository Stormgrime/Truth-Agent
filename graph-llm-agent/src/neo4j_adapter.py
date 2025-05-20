import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from neo4j import GraphDatabase, Driver, ManagedTransaction, Record, Result
from neo4j.exceptions import Neo4jError, ServiceUnavailable, ConstraintError
from datetime import datetime # Ensure datetime is imported

from src.config import settings
from src.memory_schema import NodeLabel, RelationshipType, Episode # Import specific models if needed for type hinting

logger = logging.getLogger(__name__)

class Neo4jAdapter:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self._uri = uri if uri else settings.NEO4J_URI
        self._user = user if user else settings.NEO4J_USER
        self._password = password if password else settings.NEO4J_PASSWORD
        self._driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self._uri}")
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            raise ServiceUnavailable(f"Could not connect to Neo4j at {self._uri}") from e

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Record]:
        if not self._driver:
            raise ServiceUnavailable("Neo4j driver not initialized. Cannot execute query.")
        
        parameters = parameters or {}
        try:
            with self._driver.session() as session:
                result: Result = session.run(query, parameters)
                return [record for record in result]
        except Neo4jError as e:
            logger.error(f"Neo4j query failed: {query} | Params: {parameters} | Error: {e}", exc_info=True)
            raise

    def _execute_write_transaction(self, tx_function, **kwargs) -> Any:
        if not self._driver:
            raise ServiceUnavailable("Neo4j driver not initialized. Cannot execute transaction.")
        try:
            with self._driver.session() as session:
                return session.write_transaction(tx_function, **kwargs)
        except Neo4jError as e:
            logger.error(f"Neo4j write transaction failed for {tx_function.__name__} | Error: {e}", exc_info=True)
            raise

    def _execute_read_transaction(self, tx_function, **kwargs) -> Any:
        if not self._driver:
            raise ServiceUnavailable("Neo4j driver not initialized. Cannot execute transaction.")
        try:
            with self._driver.session() as session:
                return session.read_transaction(tx_function, **kwargs)
        except Neo4jError as e:
            logger.error(f"Neo4j read transaction failed for {tx_function.__name__} | Error: {e}", exc_info=True)
            raise

    def create_constraints(self):
        constraints_queries = [
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{NodeLabel.EPISODE}) REQUIRE n.uuid IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{NodeLabel.SEMANTIC_MEMORY}) REQUIRE n.uuid IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{NodeLabel.META_MEMORY}) REQUIRE n.uuid IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{NodeLabel.COMPACTION}) REQUIRE n.uuid IS UNIQUE",
        ]
        for query in constraints_queries:
            try:
                self._execute_query(query)
                logger.info(f"Successfully applied constraint: {query.split('REQUIRE')[0].strip()}...")
            except ConstraintError:
                logger.warning(f"Constraint already exists or error applying: {query}")
            except Exception as e:
                logger.error(f"Error applying constraint {query}: {e}")

    def create_vector_indexes(self):
        if not settings.EMBEDDING_DIMENSIONS:
            logger.error("EMBEDDING_DIMENSIONS not set in config. Cannot create vector index.")
            raise ValueError("EMBEDDING_DIMENSIONS must be set to create vector indexes.")

        vector_indexes_map = {
            NodeLabel.EPISODE: f"{NodeLabel.EPISODE.lower()}_vector_index",
            NodeLabel.SEMANTIC_MEMORY: f"{NodeLabel.SEMANTIC_MEMORY.lower()}_vector_index",
            NodeLabel.META_MEMORY: f"{NodeLabel.META_MEMORY.lower()}_vector_index",
            NodeLabel.COMPACTION: f"{NodeLabel.COMPACTION.lower()}_vector_index",
        }
        
        existing_indexes_records = self._execute_query("SHOW INDEXES WHERE type = 'VECTOR' YIELD name")
        existing_vector_index_names = {record["name"] for record in existing_indexes_records}

        for node_label, index_name in vector_indexes_map.items():
            if index_name not in existing_vector_index_names:
                query = f"CALL db.index.vector.createNodeIndex('{index_name}', '{node_label}', 'embedding', {settings.EMBEDDING_DIMENSIONS}, 'cosine')"
                try:
                    self._execute_query(query)
                    logger.info(f"Successfully created vector index: {index_name} for label {node_label}")
                except Exception as e:
                    logger.error(f"Error creating vector index '{index_name}' for label '{node_label}': {e}. This might happen if the GDS plugin is not installed or dimensions mismatch an existing index with a similar name but different label.")
            else:
                logger.info(f"Vector index {index_name} for label {node_label} already exists.")

    def add_node(self, node_label: str, properties: Dict[str, Any]) -> UUID:
        if 'uuid' in properties and isinstance(properties['uuid'], UUID):
            properties['uuid'] = str(properties['uuid'])
        if 'timestamp' in properties and hasattr(properties['timestamp'], 'isoformat'): # Check if it's a datetime object
            properties['timestamp'] = properties['timestamp'].isoformat()
        
        # Convert other potential UUID or datetime fields in properties
        for key, value in properties.items():
            if isinstance(value, UUID):
                properties[key] = str(value)
            elif isinstance(value, datetime): # Ensure datetime is imported
                properties[key] = value.isoformat()


        def _tx_add_node(tx: ManagedTransaction, label: str, props: Dict[str, Any]):
            query = f"CREATE (n:{label} $props) RETURN n.uuid AS uuid"
            result = tx.run(query, props=props)
            record = result.single()
            if record and record["uuid"]:
                return UUID(record["uuid"])
            raise Neo4jError("Failed to create node or retrieve UUID.")
        try:
            return self._execute_write_transaction(_tx_add_node, label=node_label, props=properties)
        except Exception as e:
            logger.error(f"Failed to add node with label {node_label} and props {list(properties.keys())}: {e}")
            raise

    def get_node_by_uuid(self, uuid_val: UUID, node_label: Optional[str] = None) -> Optional[Dict[str, Any]]:
        label_match = f":{node_label}" if node_label else ""
        query = f"MATCH (n{label_match} {{uuid: $uuid_str}}) RETURN properties(n) AS props"
        
        def _tx_get_node(tx: ManagedTransaction, q: str, uuid_s: str):
            result = tx.run(q, uuid_str=uuid_s)
            record = result.single()
            return record["props"] if record else None
        return self._execute_read_transaction(_tx_get_node, q=query, uuid_s=str(uuid_val))

    def add_relationship(self, start_node_uuid: UUID, end_node_uuid: UUID,
                         start_node_label: str, end_node_label: str,
                         rel_type: str, properties: Optional[Dict[str, Any]] = None):
        properties = properties or {}
        for k, v in properties.items():
            if isinstance(v, UUID): properties[k] = str(v)
            elif isinstance(v, datetime): properties[k] = v.isoformat() # Ensure datetime is imported

        def _tx_add_rel(tx: ManagedTransaction, s_uuid: str, e_uuid: str, s_label: str, e_label: str, r_type: str, props: Dict):
            query = (
                f"MATCH (a:{s_label} {{uuid: $s_uuid}}), (b:{e_label} {{uuid: $e_uuid}}) "
                f"CREATE (a)-[r:{r_type} $props]->(b) RETURN type(r)"
            )
            result = tx.run(query, s_uuid=s_uuid, e_uuid=e_uuid, props=props)
            return result.single() is not None
        try:
            created = self._execute_write_transaction(
                _tx_add_rel, s_uuid=str(start_node_uuid), e_uuid=str(end_node_uuid),
                s_label=start_node_label, e_label=end_node_label, r_type=rel_type, props=properties
            )
            if not created:
                logger.warning(f"Relationship {rel_type} from {start_node_uuid} to {end_node_uuid} might not have been created.")
            return created
        except Exception as e:
            logger.error(f"Failed to add relationship {rel_type} from {start_node_uuid} to {end_node_uuid}: {e}")
            raise

    def query_vector_nodes(self, index_name: str, knn_embedding: List[float], k_value: int) -> List[Tuple[UUID, float]]:
        query = (
            f"CALL db.index.vector.queryNodes($index_name, $k, $embedding) "
            f"YIELD node, score RETURN node.uuid AS uuid, score"
        )
        def _tx_query_vector(tx: ManagedTransaction, q: str, idx_name: str, k_val: int, emb: List[float]):
            result = tx.run(q, index_name=idx_name, k=k_val, embedding=emb)
            return [(UUID(record["uuid"]), float(record["score"])) for record in result]
        return self._execute_read_transaction(_tx_query_vector, q=query, idx_name=index_name, k_val=k_value, emb=knn_embedding)

    def mark_superseded_by(self, old_node_uuid: UUID, new_node_uuid: UUID,
                           old_node_label: str, new_node_label: str):
        return self.add_relationship(
            start_node_uuid=old_node_uuid, end_node_uuid=new_node_uuid,
            start_node_label=old_node_label, end_node_label=new_node_label,
            rel_type=RelationshipType.SUPERSEDED_BY
        )

    def ensure_schema(self):
        logger.info("Ensuring Neo4j schema (constraints and vector indexes)...")
        self.create_constraints()
        try:
            self.create_vector_indexes()
        except ValueError as e:
            logger.error(f"Skipping vector index creation: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during vector index creation: {e}")
        logger.info("Schema setup process complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Ensure datetime is imported for the test data
    from uuid import uuid4 
    
    if not settings.EMBEDDING_DIMENSIONS:
        logger.warning("EMBEDDING_DIMENSIONS not set in config. Defaulting to 384 for testing.")
        settings.EMBEDDING_DIMENSIONS = 384 

    adapter = None # Define adapter outside try block for finally
    try:
        adapter = Neo4jAdapter()
        logger.info("Neo4jAdapter initialized. Ensuring schema...")
        adapter.ensure_schema()
        logger.info("Schema ensured.")

        logger.info("Testing node addition...")
        # Use the Episode Pydantic model for creating test data
        episode_data = Episode(
            speaker="test_user", 
            content="This is a test episode for Neo4j.",
            embedding=[0.1] * settings.EMBEDDING_DIMENSIONS
        )
        # Convert Pydantic model to dict for add_node
        test_episode_props = episode_data.model_dump(exclude_none=True)
        
        created_uuid = adapter.add_node(NodeLabel.EPISODE, test_episode_props)
        logger.info(f"Added Episode node with UUID: {created_uuid}")
        assert created_uuid is not None

        logger.info(f"Testing fetching node {created_uuid}...")
        fetched_node_props = adapter.get_node_by_uuid(created_uuid, NodeLabel.EPISODE)
        logger.info(f"Fetched node properties: {fetched_node_props}")
        assert fetched_node_props is not None
        assert fetched_node_props['content'] == "This is a test episode for Neo4j."

        logger.info("Testing relationship addition...")
        semantic_node_uuid = uuid4()
        test_semantic_props = {
            "uuid": str(semantic_node_uuid),
            "timestamp": datetime.now().isoformat(), # Use imported datetime
            "statement": "Test fact related to episode.",
            "importance": 0.7,
            "embedding": [0.2] * settings.EMBEDDING_DIMENSIONS
        }
        adapter.add_node(NodeLabel.SEMANTIC_MEMORY, test_semantic_props)
        logger.info(f"Added SemanticMemory node with UUID: {semantic_node_uuid}")
        
        rel_created = adapter.add_relationship(
            start_node_uuid=created_uuid, end_node_uuid=semantic_node_uuid,
            start_node_label=NodeLabel.EPISODE, end_node_label=NodeLabel.SEMANTIC_MEMORY,
            rel_type=RelationshipType.MENTIONS, properties={"how": "directly"}
        )
        logger.info(f"Relationship MENTIONS created: {rel_created}")
        assert rel_created

        if settings.EMBEDDING_DIMENSIONS:
            logger.info("Testing vector query (conceptual)...")
            try:
                index_name_to_query = f"{NodeLabel.EPISODE.lower()}_vector_index"
                dummy_search_embedding = [0.15] * settings.EMBEDDING_DIMENSIONS
                
                vector_results = adapter.query_vector_nodes(
                    index_name=index_name_to_query,
                    knn_embedding=dummy_search_embedding,
                    k_value=1
                )
                logger.info(f"Vector query results (first result if any): {vector_results[:1]}")
                if vector_results:
                     assert isinstance(vector_results[0][0], UUID)
                     assert isinstance(vector_results[0][1], float)
            except Exception as e:
                logger.error(f"Vector query test failed: {e}.")
        else:
            logger.warning("EMBEDDING_DIMENSIONS not set, skipping vector query test.")
        logger.info("Neo4jAdapter tests completed (basic functionality).")

    except ServiceUnavailable:
        logger.error("Neo4j service is unavailable. Ensure Neo4j is running and configured correctly.")
    except Exception as e:
        logger.error(f"An error occurred during Neo4jAdapter tests: {e}", exc_info=True)
    finally:
        if adapter and adapter._driver:
            adapter.close()
