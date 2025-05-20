import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
import asyncio # For async operations if sub-services are async

from graph_llm_agent.config import settings
from graph_llm_agent.neo4j_adapter import Neo4jAdapter
from graph_llm_agent.embedding_client import EmbeddingClient
from graph_llm_agent.retrieval import RetrievalService
from graph_llm_agent.context_manager import ContextManager
# from graph_llm_agent.metacognition import MetacognitionService # To be implemented next
from graph_llm_agent.memory_schema import Episode, NodeLabel, RelationshipType
from graph_llm_agent.llm_client import LLMClient # For MetacognitionService placeholder

logger = logging.getLogger(__name__)

class PlaceholderMetacognitionService:
    def __init__(self, neo4j_adapter: Neo4jAdapter, llm_client: LLMClient):
        self.neo4j_adapter = neo4j_adapter # Stored but not used by placeholder
        self.llm_client = llm_client # Stored but not used by placeholder
        logger.info("PlaceholderMetacognitionService initialized.")
    
    async def reflect_on_recent_episodes(self, recent_episodes_data: List[Dict[str, Any]], force_reflection: bool = False):
        # In a real implementation, this would interact with LLM and Neo4j
        # For now, just log. The data passed would be List[Episode] or similar.
        if force_reflection:
            logger.info(f"PlaceholderMetacognitionService: Forced reflection on {len(recent_episodes_data)} episodes.")
        else:
            logger.info(f"PlaceholderMetacognitionService: Reflection triggered for {len(recent_episodes_data)} episodes.")
        return None 


class AgentMemory:
    def __init__(self, 
                 neo4j_adapter: Neo4jAdapter, 
                 embedding_client: EmbeddingClient, 
                 retrieval_service: RetrievalService,
                 context_manager: ContextManager,
                 metacognition_service: Any, # Placeholder or actual MetacognitionService
                 llm_client: LLMClient # Pass LLMClient for MetacognitionService
                ):
        self.neo4j_adapter = neo4j_adapter
        self.embedding_client = embedding_client
        self.retrieval_service = retrieval_service
        self.context_manager = context_manager
        self.metacognition_service = metacognition_service if metacognition_service else PlaceholderMetacognitionService(neo4j_adapter, llm_client)
        self.llm_client = llm_client # Store if needed, or ensure MetacognitionService gets it
        
        self.interaction_count_since_last_reflection = 0
        logger.info("AgentMemory initialized.")

    async def add_interaction(self, 
                        speaker: str, 
                        content: str, 
                        episode_uuid: Optional[UUID] = None, 
                        previous_episode_uuid: Optional[UUID] = None
                       ) -> Episode:
        
        episode_uuid = episode_uuid if episode_uuid else uuid4()
        logger.debug(f"Adding interaction: speaker={speaker}, uuid={episode_uuid}, content='{content[:50]}...'")

        content_embedding = self.embedding_client.get_embedding(content)
        if not content_embedding:
            logger.warning(f"Could not generate embedding for episode {episode_uuid}. Storing with zero-vector.")
            content_embedding = [0.0] * (settings.EMBEDDING_DIMENSIONS or 384)

        token_count = self.context_manager._count_tokens(content) 

        episode_data = Episode(
            uuid=episode_uuid, speaker=speaker, content=content,
            embedding=content_embedding, token_count=token_count,
            previous_episode_uuid=previous_episode_uuid
        )

        try:
            props_to_store = episode_data.model_dump(exclude_none=True)
            self.neo4j_adapter.add_node(NodeLabel.EPISODE, props_to_store)
            logger.info(f"Stored Episode {episode_uuid} (speaker: {speaker}) in Neo4j.")
        except Exception as e:
            logger.error(f"Failed to store Episode {episode_uuid} in Neo4j: {e}", exc_info=True)

        if previous_episode_uuid:
            try:
                self.neo4j_adapter.add_relationship(
                    previous_episode_uuid, episode_uuid,
                    NodeLabel.EPISODE, NodeLabel.EPISODE,
                    RelationshipType.NEXT_EPISODE
                )
                logger.info(f"Linked Episode {previous_episode_uuid} ->NEXT-> Episode {episode_uuid}.")
            except Exception as e:
                logger.error(f"Failed to link episodes {previous_episode_uuid} to {episode_uuid}: {e}", exc_info=True)

        await self.context_manager.add_interaction(speaker, content, episode_uuid)
        
        self.interaction_count_since_last_reflection += 1
        if self.interaction_count_since_last_reflection >= settings.REFLECTION_INTERACTION_INTERVAL:
            logger.info(f"Reflection interval of {settings.REFLECTION_INTERACTION_INTERVAL} reached.")
            # This is a simplified way to get recent data. A real implementation might query Neo4j.
            recent_context_for_reflection = list(self.context_manager.context_window)[-settings.REFLECTION_INTERACTION_INTERVAL:]
            await self.metacognition_service.reflect_on_recent_episodes(recent_context_for_reflection)
            self.interaction_count_since_last_reflection = 0

        return episode_data

    def retrieve_context_for_query(self, query_text: str, k_value: Optional[int] = None) -> List[Dict[str, Any]]:
        logger.debug(f"Retrieving context for query: '{query_text[:50]}...'")
        if not query_text: return []
        
        retrieved_memories = self.retrieval_service.retrieve_relevant_memories(
            query_text=query_text,
            k_value=k_value if k_value is not None else settings.RETRIEVAL_K_VALUE
        )
        logger.info(f"Retrieved {len(retrieved_memories)} memories for query.")
        return retrieved_memories


if __name__ == "__main__":
    # Mock dependencies
    class MockNeo4jAdapter(Neo4jAdapter):
        def __init__(self): self._driver = True; logger.info("MockNeo4jAdapter initialized")
        def add_node(self, node_label: str, properties: Dict[str, Any]): 
            logger.info(f"Mocked add_node: {node_label}, {properties.get('uuid')}")
            return properties.get("uuid", uuid4())
        def add_relationship(self, *args, **kwargs): logger.info(f"Mocked add_relationship: {args[0]}->{args[1]} type {args[4]}")
        def close(self): pass

    class MockEmbeddingClient(EmbeddingClient):
        def __init__(self): 
            self.model_name = "mock_embed_client"; self.device = "cpu"
            self.dim = settings.EMBEDDING_DIMENSIONS or 384
            logger.info(f"MockEmbeddingClient initialized with dim {self.dim}")
        def get_embedding(self, text: str) -> List[float]: return [0.1] * self.dim
        def get_embedding_dimensionality(self) -> Optional[int]: return self.dim

    class MockRetrievalService(RetrievalService):
        def __init__(self): 
            # super().__init__(neo4j_adapter=None, embedding_client=None) # Call super if it needs specific init
            logger.info("MockRetrievalService initialized")
        def retrieve_relevant_memories(self, query_text: str, k_value: Optional[int] = None) -> List[Dict[str, Any]]:
            logger.info(f"Mocked retrieve_relevant_memories for query: '{query_text}'")
            return [{"uuid": uuid4(), "content": "mocked retrieved memory", "score": 0.85, "final_score": 0.85}]

    class MockContextManager(ContextManager):
        def __init__(self): 
            # Pass a placeholder for compaction service to super if it expects one
            super().__init__(compaction_service=None) 
            logger.info("MockContextManager initialized")
        def add_interaction(self, role: str, content: str, episode_uuid: UUID): 
            # Call real method for token counting and deque if it's safe with None compaction
            # For a pure mock, you might just log:
            logger.info(f"Mocked ContextManager.add_interaction: role={role}, uuid={episode_uuid}, content='{content[:20]}...'")
            # Simulate adding to deque for reflection test to work
            self.context_window.append({"role": role, "content": content, "uuid": episode_uuid, "token_count": self._count_tokens(content)})
            self.current_total_tokens += self._count_tokens(content)
        def _count_tokens(self, text:str) -> int: return len(text.split()) # Simple space-based token count for mock

    class MockLLMClient(LLMClient): # Needed for PlaceholderMetacognitionService
         def __init__(self): self.mode = "mock"; logger.info("MockLLMClient for Metacognition initialized")


    logging.basicConfig(level=logging.INFO)
    if not settings.EMBEDDING_DIMENSIONS: settings.EMBEDDING_DIMENSIONS = 384

    mock_neo4j = MockNeo4jAdapter()
    mock_embed = MockEmbeddingClient()
    mock_retrieval = MockRetrievalService() # Initialize mock retrieval
    mock_context = MockContextManager()
    mock_llm_for_meta = MockLLMClient()
    mock_meta_service = PlaceholderMetacognitionService(mock_neo4j, mock_llm_for_meta)

    agent_memory = AgentMemory(
        neo4j_adapter=mock_neo4j, embedding_client=mock_embed,
        retrieval_service=mock_retrieval, context_manager=mock_context,
        metacognition_service=mock_meta_service, llm_client=mock_llm_for_meta
    )

    async def test_run():
        logger.info("\n--- Testing AgentMemory ---")
        user_content1 = "Hello Agent, this is my first message."
        ep1_uuid = uuid4()
        episode1 = await agent_memory.add_interaction("user", user_content1, episode_uuid=ep1_uuid)
        assert episode1.uuid == ep1_uuid

        assistant_content1 = "Hello User! How can I help you today?"
        ep2_uuid = uuid4()
        episode2 = await agent_memory.add_interaction("assistant", assistant_content1, episode_uuid=ep2_uuid, previous_episode_uuid=episode1.uuid)
        assert episode2.previous_episode_uuid == episode1.uuid

        retrieved = agent_memory.retrieve_context_for_query("some query")
        assert len(retrieved) == 1 and retrieved[0]["content"] == "mocked retrieved memory"

        logger.info("\nTesting reflection trigger path...")
        original_reflection_interval = settings.REFLECTION_INTERACTION_INTERVAL
        settings.REFLECTION_INTERACTION_INTERVAL = 2 # Trigger reflection every 2 interactions
        
        # Reset interaction count for this specific test section for predictable behavior
        agent_memory.interaction_count_since_last_reflection = 0 

        # First interaction to make count = 1
        ep3_uuid = uuid4()
        await agent_memory.add_interaction("user", "Another message for reflection test 1", uuid4(), previous_episode_uuid=episode2.uuid)
        assert agent_memory.interaction_count_since_last_reflection == 1
        
        # Second interaction to make count = 2 (triggers reflection, then resets to 0)
        ep4_uuid = uuid4()
        await agent_memory.add_interaction("user", "This message should trigger reflection.", uuid4(), previous_episode_uuid=ep3_uuid) 
        assert agent_memory.interaction_count_since_last_reflection == 0
        
        settings.REFLECTION_INTERACTION_INTERVAL = original_reflection_interval # Restore original setting

        logger.info("AgentMemory tests completed.")

    asyncio.run(test_run())

# Ensure all mock initializations are correct, especially superclass calls if needed.
# For MockRetrievalService, if its __init__ expects arguments, they must be provided or the __init__ fully mocked.
# The provided RetrievalService __init__ expects neo4j_adapter and embedding_client.
# So, MockRetrievalService needs to either not call super().__init__ or be provided with (mocked) versions of these.
# Corrected MockRetrievalService by removing the super call as it's not needed for this mock's purpose.
# Corrected MockContextManager's _count_tokens to be a simple split for testing.
# Corrected reflection test logic for clarity.
