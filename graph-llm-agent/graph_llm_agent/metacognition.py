import logging
import json
import re # For parsing JSON from LLM response
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
import asyncio # For async operations

from graph_llm_agent.config import settings
from graph_llm_agent.llm_client import LLMClient
from graph_llm_agent.neo4j_adapter import Neo4jAdapter
from graph_llm_agent.embedding_client import EmbeddingClient # For embedding extracted facts
from graph_llm_agent.memory_schema import MetaMemory, SemanticMemory, NodeLabel, RelationshipType # Pydantic models

logger = logging.getLogger(__name__)

class MetacognitionService:
    def __init__(self, 
                 neo4j_adapter: Neo4jAdapter, 
                 llm_client: LLMClient, 
                 embedding_client: EmbeddingClient):
        self.neo4j_adapter = neo4j_adapter
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        logger.info("MetacognitionService initialized.")

    async def _parse_llm_reflection_response(self, llm_response_text: str) -> Optional[Dict[str, Any]]:
        try:
            json_match = re.search(r'```json\s*([\s\S]+?)\s*```|({[\s\S]+})', llm_response_text)
            if json_match:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
                data = json.loads(json_str)
            else:
                data = json.loads(llm_response_text)

            if not isinstance(data, dict) or                'summary' not in data or not isinstance(data['summary'], str) or                'facts' not in data or not isinstance(data['facts'], list) or                'salience_score' not in data or not isinstance(data['salience_score'], (float, int)):
                logger.error(f"LLM reflection response JSON missing required fields or has wrong types: {data}")
                return None
            
            data['salience_score'] = float(data['salience_score'])
            if not (0.0 <= data['salience_score'] <= 1.0):
                logger.warning(f"Salience score {data['salience_score']} out of range [0,1]. Clamping.")
                data['salience_score'] = max(0.0, min(1.0, data['salience_score']))
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM reflection response: {e}. Response text: '{llm_response_text[:200]}...'")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM reflection response: {e}", exc_info=True)
            return None

    async def reflect_on_episodes(self, episodes_data: List[Dict[str, Any]]) -> Optional[UUID]:
        if not episodes_data:
            logger.info("No episodes provided for reflection.")
            return None
        logger.info(f"Starting reflection on {len(episodes_data)} episodes.")

        conversation_excerpt = "\n".join([f"{e.get('role', 'unknown')}: {e.get('content', '')}" for e in episodes_data])
        reflection_prompt = (
            f"You are a helpful AI assistant performing a reflection task. "
            f"Analyze the following conversation excerpt. Provide a concise summary of the main topics and flow. "
            f"Extract any new factual statements or key decisions made. These facts should be simple, declarative statements. "
            f"Finally, rate the overall importance or salience of this entire excerpt on a scale of 0.0 (trivial) to 1.0 (highly significant and memorable).\n\n"
            f"Respond ONLY with a JSON object with the following keys:\n"
            f'- "summary": (string) Your concise summary of the excerpt.\n'
            f'- "facts": (list of strings) A list of extracted factual statements. If no new facts, provide an empty list.\n'
            f'- "salience_score": (float) Your rating of the overall salience from 0.0 to 1.0.\n\n'
            f"Conversation Excerpt:\n"""\n{conversation_excerpt}\n"""\n\n"
            f"JSON Response:"
        )
        
        try:
            raw_llm_response, confidence = self.llm_client.generate_response(
                prompt=reflection_prompt, temperature=0.4, max_tokens=500
            )
            if "Error:" in raw_llm_response or confidence == 0.0:
                logger.error(f"LLM client failed to generate reflection: {raw_llm_response}")
                return None
        except Exception as e:
            logger.error(f"LLM call for reflection failed: {e}", exc_info=True)
            return None

        parsed_data = await self._parse_llm_reflection_response(raw_llm_response)
        if not parsed_data: return None

        summary_text = parsed_data['summary']
        facts_list = parsed_data['facts']
        salience_score = parsed_data['salience_score']
        meta_memory_node_uuid = uuid4()

        try:
            original_episode_node_ids_str = [str(e.get("uuid")) for e in episodes_data if e.get("uuid")]
            meta_memory_obj = MetaMemory(
                uuid=meta_memory_node_uuid,
                summary_text=summary_text,
                # salience_score=salience_score, # This should be part of MetaMemory Pydantic model
                covered_episode_uuids=[UUID(uid_str) for uid_str in original_episode_node_ids_str], # Ensure UUID objects
                reflection_type="episodic_reflection"
            )
            # Add 'salience_score' to MetaMemory Pydantic model if not present
            # For now, assume it will be part of model_dump if in the object.
            # If MetaMemory model doesn't have 'salience_score', add it to props_to_store like this:
            props_to_store = meta_memory_obj.model_dump(exclude_none=True)
            # if 'salience_score' not in MetaMemory.model_fields: # This check might be problematic if model_fields is not directly comparable
            props_to_store['salience_score'] = salience_score # Add it directly for now, ensure MetaMemory model is updated


            self.neo4j_adapter.add_node(NodeLabel.META_MEMORY, props_to_store)
            logger.info(f"Created MetaMemory node {meta_memory_node_uuid} with salience {salience_score:.2f}.")

            for episode_uuid_str in original_episode_node_ids_str:
                self.neo4j_adapter.add_relationship(
                    meta_memory_node_uuid, UUID(episode_uuid_str),
                    NodeLabel.META_MEMORY, NodeLabel.EPISODE, RelationshipType.COVERS
                )
            logger.info(f"Linked MetaMemory node {meta_memory_node_uuid} to {len(original_episode_node_ids_str)} episodes.")

            for fact_statement in facts_list:
                if not fact_statement.strip(): continue
                fact_embedding = self.embedding_client.get_embedding(fact_statement) or                                  [0.0] * (settings.EMBEDDING_DIMENSIONS or 384)

                semantic_memory_obj = SemanticMemory(statement=fact_statement, embedding=fact_embedding)
                semantic_props = semantic_memory_obj.model_dump(exclude_none=True)
                # Add custom property linking to this reflection
                semantic_props['source_reflection_uuid'] = str(meta_memory_node_uuid)
                
                semantic_node_uuid = self.neo4j_adapter.add_node(NodeLabel.SEMANTIC_MEMORY, semantic_props)
                logger.info(f"Created SemanticMemory node {semantic_node_uuid} for fact: '{fact_statement[:50]}...'")
                self.neo4j_adapter.add_relationship(
                    meta_memory_node_uuid, semantic_node_uuid,
                    NodeLabel.META_MEMORY, NodeLabel.SEMANTIC_MEMORY, RelationshipType.CAPTURES # Changed from DERIVED_FROM to CAPTURES
                )
            logger.info(f"Processed {len(facts_list)} facts for MetaMemory {meta_memory_node_uuid}.")

        except Exception as e:
            logger.error(f"Error during graph writes for reflection {meta_memory_node_uuid}: {e}", exc_info=True)
            return None

        logger.info(f"Propagating salience score {salience_score:.2f} to {len(original_episode_node_ids_str)} episodes.")
        for episode_uuid_str in original_episode_node_ids_str:
            try:
                new_importance_value = min(1.0, 0.5 + (salience_score / 2.0)) # Simple blending
                update_query = f"MATCH (e:{NodeLabel.EPISODE} {{uuid: $uuid}}) SET e.importance = $importance"
                self.neo4j_adapter._execute_query(update_query, {"uuid": episode_uuid_str, "importance": new_importance_value})
            except Exception as e:
                logger.error(f"Failed to update importance for episode {episode_uuid_str}: {e}", exc_info=True)
        
        logger.info(f"Reflection completed. MetaMemory node: {meta_memory_node_uuid}")
        return meta_memory_node_uuid

if __name__ == "__main__":
    # Imports moved inside main_test or here for clarity if needed by mocks
    import re 
    from uuid import UUID, uuid4 # Ensure these are available for the test code
    from graph_llm_agent.memory_schema import Episode # For mock episode data creation

    class MockLLMClient(LLMClient):
        def __init__(self): self.mode = "mock"; logger.info("MockLLMClient for Metacognition")
        def generate_response(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, float]:
            response_json = {"summary": "Mock summary.", "facts": ["Fact A.", "Fact B."], "salience_score": 0.8}
            return json.dumps(response_json), 0.99

    class MockNeo4jAdapter(Neo4jAdapter):
        def __init__(self): 
            self._driver = True; self.nodes_added = {}; self.rels_added = []
            logger.info("MockNeo4jAdapter for Metacognition")
        def add_node(self, label: str, props: Dict[str, Any]) -> UUID:
            uid = UUID(str(props.get("uuid", uuid4())))
            self.nodes_added[uid] = {"label": label, "props": props}
            return uid
        def add_relationship(self, s, e, sl, el, rt, p=None): self.rels_added.append({"from":s,"to":e,"type":rt})
        def _execute_query(self, q: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
            if "SET e.importance" in q:
                uid = UUID(str(params.get('uuid')))
                if uid in self.nodes_added and self.nodes_added[uid]['label'] == NodeLabel.EPISODE:
                    self.nodes_added[uid]['props']['importance'] = params.get('importance')
            return []
        def close(self): pass

    class MockEmbeddingClient(EmbeddingClient):
        def __init__(self): 
            self.dim = settings.EMBEDDING_DIMENSIONS or 384
            logger.info(f"MockEmbeddingClient for Metacognition, dim {self.dim}")
        def get_embedding(self, text: str) -> List[float]: return [0.3] * self.dim
        def get_embedding_dimensionality(self) -> Optional[int]: return self.dim

    logging.basicConfig(level=logging.INFO)
    if not settings.EMBEDDING_DIMENSIONS: settings.EMBEDDING_DIMENSIONS = 384

    mock_llm, mock_neo4j, mock_embed = MockLLMClient(), MockNeo4jAdapter(), MockEmbeddingClient()
    service = MetacognitionService(mock_neo4j, mock_llm, mock_embed)

    ep1_id, ep2_id = uuid4(), uuid4()
    # Create mock Episode Pydantic models for more realistic data structure
    # This isn't strictly needed if episodes_data is just dicts, but good for consistency.
    # However, episodes_data is defined as List[Dict[str,Any]] in reflect_on_episodes.
    test_eps_data = [
        {"uuid": str(ep1_id), "role": "user", "content": "Msg 1", "importance": 0.5},
        {"uuid": str(ep2_id), "role": "assistant", "content": "Msg 2", "importance": 0.5},
    ]
    mock_neo4j.nodes_added[ep1_id] = {"label": NodeLabel.EPISODE, "props": test_eps_data[0]}
    mock_neo4j.nodes_added[ep2_id] = {"label": NodeLabel.EPISODE, "props": test_eps_data[1]}

    async def main_test():
        logger.info("\n--- Testing MetacognitionService ---")
        meta_uuid = await service.reflect_on_episodes(test_eps_data)
        assert meta_uuid is not None
        assert meta_uuid in mock_neo4j.nodes_added
        assert mock_neo4j.nodes_added[meta_uuid]["props"]["salience_score"] == 0.8
        
        num_facts = len(json.loads(mock_llm.generate_response("",0,0)[0])["facts"]) # Get from mock response
        assert sum(1 for r in mock_neo4j.rels_added if r["type"] == RelationshipType.CAPTURES) == num_facts
        assert sum(1 for r in mock_neo4j.rels_added if r["type"] == RelationshipType.COVERS) == len(test_eps_data)
        
        expected_importance = min(1.0, 0.5 + (0.8 / 2.0))
        assert mock_neo4j.nodes_added[ep1_id]["props"]["importance"] == expected_importance
        logger.info("MetacognitionService tests passed.")

    asyncio.run(main_test())

# Final check: Ensure MetaMemory Pydantic model includes 'salience_score: float'
# and 'reflection_type: str'. If not, the props_to_store approach is a workaround.
# The line `if 'salience_score' not in MetaMemory.model_fields:` was commented out
# as `MetaMemory.model_fields` is the correct Pydantic v2 way to check fields.
# The direct assignment `props_to_store['salience_score'] = salience_score` is kept.
# RelationshipType.CAPTURES is used for MetaMemory -> SemanticMemory.
# The prompt mentioned "propagating salience score...to the importance score of the original episodes".
# This is now implemented by setting e.importance = 0.5 + (salience_score / 2.0) (capped at 1.0).
# Previously this was just `salience_score`, which might be too high if salience is e.g. 0.9.
# The blending provides a more moderated update.
