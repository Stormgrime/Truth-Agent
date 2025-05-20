import logging
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import UUID

from graph_llm_agent.config import settings
from graph_llm_agent.neo4j_adapter import Neo4jAdapter
from graph_llm_agent.embedding_client import EmbeddingClient
from graph_llm_agent.memory_schema import NodeLabel # To construct index names

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, neo4j_adapter: Neo4jAdapter, embedding_client: EmbeddingClient):
        self.neo4j_adapter = neo4j_adapter
        self.embedding_client = embedding_client
        if not settings.EMBEDDING_DIMENSIONS:
            logger.warning("EMBEDDING_DIMENSIONS not set in config. Retrieval might not work as expected.")
            # Attempt to get it from embedding client if available
            dim = self.embedding_client.get_embedding_dimensionality()
            if dim:
                settings.EMBEDDING_DIMENSIONS = dim
            else:
                # Raise error or log critical if still not found, as it's vital for index names too
                logger.critical("EMBEDDING_DIMENSIONS could not be determined. Vector index names might be incorrect.")


    def retrieve_relevant_memories(self, query_text: str, k_value: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query_text:
            logger.warning("Received empty query_text for retrieval. Returning empty list.")
            return []

        k_value = k_value if k_value is not None else settings.RETRIEVAL_K_VALUE
        
        query_embedding = self.embedding_client.get_embedding(query_text)
        if not query_embedding:
            logger.warning("Failed to generate embedding for query_text. Returning empty list.")
            return []

        # For now, primarily query Episode memories. This can be expanded later to query other types
        # (SemanticMemory, MetaMemory) and merge results.
        # The index name should match how it's created in neo4j_adapter.py
        index_name = f"{NodeLabel.EPISODE.lower()}_vector_index" # Consistent with neo4j_adapter

        try:
            # neo4j_adapter.query_vector_nodes now returns List[Dict[str, Any]]
            # Each dict contains: uuid, content, speaker, timestamp, importance, labels, score (similarity)
            candidate_memories_data = self.neo4j_adapter.query_vector_nodes(
                index_name=index_name,
                knn_embedding=query_embedding,
                k_value=k_value * 2 # Retrieve more candidates initially to allow for re-scoring and filtering
            )
        except Exception as e:
            logger.error(f"Error querying vector nodes from Neo4j for index {index_name}: {e}", exc_info=True)
            return []

        scored_memories: List[Dict[str, Any]] = []
        for mem_data in candidate_memories_data:
            try:
                similarity_score = mem_data.get('score', 0.0)
                importance_score = mem_data.get('importance', 0.5) # Default importance if not present
                
                timestamp_str = mem_data.get('timestamp')
                if not timestamp_str:
                    logger.warning(f"Memory {mem_data.get('uuid')} missing timestamp. Skipping recency calculation.")
                    age_hours = float('inf') # Penalize missing timestamp by making it very old
                else:
                    # Assuming timestamp is stored as ISO string by adapter or directly by Neo4j if it's a datetime property
                    if isinstance(timestamp_str, datetime):
                        memory_timestamp = timestamp_str
                    elif isinstance(timestamp_str, str):
                        try:
                            memory_timestamp = datetime.fromisoformat(timestamp_str)
                        except ValueError: # Handle cases where it might be a different string format or already a naive datetime
                             try:
                                memory_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f%z') # Example with tz
                             except ValueError:
                                try:
                                    memory_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f') # Example without tz
                                    if memory_timestamp.tzinfo is None: # If naive, assume UTC
                                        memory_timestamp = memory_timestamp.replace(tzinfo=timezone.utc)
                                except ValueError:
                                    logger.error(f"Could not parse timestamp string: {timestamp_str} for memory {mem_data.get('uuid')}")
                                    age_hours = float('inf')
                                    continue # Skip if timestamp is unparseable
                    else: # Might be Neo4j native ZonedDateTime etc.
                         logger.warning(f"Timestamp for memory {mem_data.get('uuid')} is of unhandled type: {type(timestamp_str)}. Trying to convert.")
                         try:
                            # Neo4j driver might return a specific datetime object
                            # For ZonedDateTime: memory_timestamp = timestamp_str.to_native()
                            # This requires checking the exact type returned by the driver for Neo4j datetimes
                            if hasattr(timestamp_str, 'to_native') and callable(timestamp_str.to_native):
                                memory_timestamp = timestamp_str.to_native()
                            else: # Fallback, or raise error
                                logger.error(f"Cannot convert Neo4j timestamp type {type(timestamp_str)} to standard datetime.")
                                age_hours = float('inf')
                                continue
                         except Exception as te:
                            logger.error(f"Error converting Neo4j timestamp {timestamp_str}: {te}")
                            age_hours = float('inf')
                            continue

                    # Ensure memory_timestamp is offset-aware for correct comparison with now()
                    if memory_timestamp.tzinfo is None:
                        memory_timestamp = memory_timestamp.replace(tzinfo=timezone.utc) # Assume UTC if naive

                    current_time = datetime.now(timezone.utc)
                    age_delta = current_time - memory_timestamp
                    age_hours = age_delta.total_seconds() / 3600.0
                    if age_hours < 0: age_hours = 0 # Should not happen if data is correct

                recency_score_component = math.exp(-age_hours / settings.RETRIEVAL_RECENCY_DECAY_HOURS)

                final_score = (settings.RETRIEVAL_SCORE_SIMILARITY_WEIGHT * similarity_score) + \
                              (settings.RETRIEVAL_SCORE_RECENCY_WEIGHT * recency_score_component) + \
                              (settings.RETRIEVAL_SCORE_IMPORTANCE_WEIGHT * importance_score)
                
                # Add final_score to the memory data dictionary
                mem_data['final_score'] = final_score
                mem_data['calculated_age_hours'] = age_hours # For debugging/inspection
                mem_data['similarity_score'] = similarity_score # For debugging/inspection

                scored_memories.append(mem_data)

            except Exception as e:
                logger.error(f"Error processing memory data {mem_data.get('uuid')}: {e}", exc_info=True)
                continue # Skip this memory if there's an error processing it

        # Sort by final_score in descending order
        scored_memories.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        return scored_memories[:k_value]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mocking dependencies for testing
    # We need to import these here if they are not available globally for the test block
    # For example, if running `python graph_llm_agent/retrieval.py` directly
    from graph_llm_agent.neo4j_adapter import Neo4jAdapter
    from graph_llm_agent.embedding_client import EmbeddingClient
    from graph_llm_agent.memory_schema import NodeLabel # Assuming this is used by mocks
    from graph_llm_agent.config import settings # For test overrides

    class MockNeo4jAdapter(Neo4jAdapter):
        def __init__(self):
            self._driver = True # Mock connected state
            logger.info("MockNeo4jAdapter initialized")

        def query_vector_nodes(self, index_name: str, knn_embedding: List[float], k_value: int) -> List[Dict[str, Any]]:
            logger.info(f"Mocked query_vector_nodes for index '{index_name}' with k={k_value}")
            # Return dummy data that matches the expected structure from the real adapter
            base_time = datetime.now(timezone.utc)
            return [
                {
                    "uuid": UUID("00000000-0000-0000-0000-000000000001"), "content": "Memory 1 (recent, high sim, avg importance)", 
                    "speaker": "user", "timestamp": base_time.isoformat(), 
                    "importance": 0.7, "labels": [NodeLabel.EPISODE], "score": 0.9
                },
                {
                    "uuid": UUID("00000000-0000-0000-0000-000000000002"), "content": "Memory 2 (older, med sim, high importance)", 
                    "speaker": "assistant", "timestamp": (base_time - timedelta(hours=48)).isoformat(), 
                    "importance": 0.9, "labels": [NodeLabel.EPISODE], "score": 0.75
                },
                {
                    "uuid": UUID("00000000-0000-0000-0000-000000000003"), "content": "Memory 3 (very recent, low sim, low importance)", 
                    "speaker": "user", "timestamp": (base_time - timedelta(hours=1)).isoformat(), 
                    "importance": 0.3, "labels": [NodeLabel.EPISODE], "score": 0.5
                },
            ]
        def close(self): pass # Mock close

    class MockEmbeddingClient(EmbeddingClient):
        def __init__(self):
            self.model_name = "mock_model"
            self.device = "cpu"
            # settings.EMBEDDING_DIMENSIONS = 384 # Mock dimension for test
            if not settings.EMBEDDING_DIMENSIONS: settings.EMBEDDING_DIMENSIONS = 384 # Ensure it's set for test
            logger.info(f"MockEmbeddingClient initialized with dim {settings.EMBEDDING_DIMENSIONS}")

        def get_embedding(self, text: str) -> List[float]:
            return [0.1] * settings.EMBEDDING_DIMENSIONS 
        
        def get_embedding_dimensionality(self) -> Optional[int]:
            return settings.EMBEDDING_DIMENSIONS

    # Need timedelta for mock data
    from datetime import timedelta

    logger.info("Testing RetrievalService...")
    
    # Override settings for predictable scoring in test
    original_weights = (settings.RETRIEVAL_SCORE_SIMILARITY_WEIGHT, settings.RETRIEVAL_SCORE_RECENCY_WEIGHT, settings.RETRIEVAL_SCORE_IMPORTANCE_WEIGHT)
    settings.RETRIEVAL_SCORE_SIMILARITY_WEIGHT = 0.6
    settings.RETRIEVAL_SCORE_RECENCY_WEIGHT = 0.3
    settings.RETRIEVAL_SCORE_IMPORTANCE_WEIGHT = 0.1
    settings.RETRIEVAL_RECENCY_DECAY_HOURS = 24.0 # Tau for recency

    mock_neo4j = MockNeo4jAdapter()
    mock_embed = MockEmbeddingClient()
    
    retrieval_service = RetrievalService(neo4j_adapter=mock_neo4j, embedding_client=mock_embed)
    
    test_query = "Tell me about recent important memories."
    retrieved_memories = retrieval_service.retrieve_relevant_memories(test_query, k_value=3)
    
    logger.info(f"Retrieved {len(retrieved_memories)} memories for query: '{test_query}'")
    for i, mem in enumerate(retrieved_memories):
        logger.info(f"Memory {i+1}: UUID={mem.get('uuid')}, FinalScore={mem.get('final_score'):.4f}, AgeHours={mem.get('calculated_age_hours'):.2f}, SimScore={mem.get('similarity_score')}, Importance={mem.get('importance')}, Content='{mem.get('content')}'")

    # Basic assertions
    assert len(retrieved_memories) <= 3
    if len(retrieved_memories) > 1:
        assert retrieved_memories[0].get('final_score',0) >= retrieved_memories[1].get('final_score',0)

    # Test with empty query
    logger.info("Testing with empty query...")
    empty_retrieval = retrieval_service.retrieve_relevant_memories("", k_value=3)
    assert len(empty_retrieval) == 0
    logger.info("Empty query test successful.")

    # Restore original settings if they were changed for test
    settings.RETRIEVAL_SCORE_SIMILARITY_WEIGHT, settings.RETRIEVAL_SCORE_RECENCY_WEIGHT, settings.RETRIEVAL_SCORE_IMPORTANCE_WEIGHT = original_weights

    logger.info("RetrievalService test completed.")
