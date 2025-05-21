import logging
import tiktoken # For token counting of summaries
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID, uuid4

from graph_llm_agent.config import settings
from graph_llm_agent.llm_client import LLMClient
from graph_llm_agent.neo4j_adapter import Neo4jAdapter
from graph_llm_agent.memory_schema import CompactionNode, NodeLabel # Pydantic model and labels
from graph_llm_agent.embedding_client import EmbeddingClient # Import for type hinting or direct use

logger = logging.getLogger(__name__)

class CompactionService:
    def __init__(self, llm_client: LLMClient, neo4j_adapter: Neo4jAdapter, embedding_client: EmbeddingClient): # Added EmbeddingClient
        self.llm_client = llm_client
        self.neo4j_adapter = neo4j_adapter
        self.embedding_client = embedding_client # Store embedding client

        if settings.OVERRIDE_TOKENIZER:
            try:
                self.tokenizer = tiktoken.get_encoding(settings.OVERRIDE_TOKENIZER)
                logger.info(f"CompactionService: Using OVERRIDE_TOKENIZER '{settings.OVERRIDE_TOKENIZER}'.")
            except Exception as e_override:
                logger.error(f"CompactionService: Failed to load OVERRIDE_TOKENIZER '{settings.OVERRIDE_TOKENIZER}'. Error: {e_override}. Attempting other fallbacks.")
                self._initialize_default_tokenizers() 
        else:
            self._initialize_default_tokenizers()
        
        # logger.info(f"CompactionService initialized with tokenizer: {self.tokenizer.name}") # Covered by specific logs

    def _initialize_default_tokenizers(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model(settings.LLM_MODEL_NAME)
            logger.info(f"CompactionService: Using tokenizer '{self.tokenizer.name}' based on LLM_MODEL_NAME '{settings.LLM_MODEL_NAME}'.")
        except KeyError:
            logger.warning(f"CompactionService: No specific tiktoken encoding for model '{settings.LLM_MODEL_NAME}'. Falling back.")
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("CompactionService: Using 'cl100k_base' tokenizer as fallback.")
            except Exception as e_cl100k:
                logger.warning(f"CompactionService: Failed to load 'cl100k_base', falling back to 'p50k_base'. Error: {e_cl100k}")
                try:
                    self.tokenizer = tiktoken.get_encoding("p50k_base")
                    logger.info("CompactionService: Using 'p50k_base' tokenizer as further fallback.")
                except Exception as e_p50k:
                    logger.error(f"CompactionService: Failed to load any tiktoken encoding: {e_p50k}", exc_info=True)
                    raise RuntimeError("Could not initialize tiktoken tokenizer for CompactionService.") from e_p50k

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        return len(self.tokenizer.encode(text))

    def _prepare_text_for_summary(self, messages_to_compact: List[Dict[str, Any]]) -> str:
        full_text = ""
        for msg in messages_to_compact:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            full_text += f"{role}: {content}\n"
        return full_text.strip()

    async def _generate_summary_text_with_llm(self, text_to_summarize: str, original_total_tokens: int) -> Tuple[Optional[str], int]:
        if not text_to_summarize:
            return None, 0
            
        max_summary_tokens = max(50, min(250, int(original_total_tokens * 0.2))) 

        summary_prompt = (
            f"Summarize the following conversation excerpt concisely. "
            f"The summary should capture the key information, decisions, and questions. "
            f"Aim for a summary around {max_summary_tokens // 2} - {max_summary_tokens} tokens.\n\n"
            f"Conversation Excerpt:\n{text_to_summarize}\n\n"
            f"Summary (ensure it's a concise paragraph):"
        )
        
        logger.info(f"Requesting LLM summary for original tokens: {original_total_tokens}, target summary tokens: ~{max_summary_tokens}")

        summary_text, confidence = self.llm_client.generate_response(
            prompt=summary_prompt,
            temperature=0.5,
            max_tokens=max_summary_tokens + 50 
        )

        if "Error:" in summary_text or confidence == 0.0:
            logger.error(f"LLM failed to generate summary. Response: {summary_text}")
            return None, 0
            
        summary_token_count = self._count_tokens(summary_text)
        logger.info(f"Generated summary: '{summary_text[:100]}...', token count: {summary_token_count}")
        
        return summary_text, summary_token_count

    async def compact_and_store_memory(self, messages_to_compact: List[Dict[str, Any]]) -> Tuple[Optional[str], int, Optional[UUID]]:
        if not messages_to_compact:
            logger.warning("No messages provided for compaction.")
            return None, 0, None

        text_to_summarize = self._prepare_text_for_summary(messages_to_compact)
        original_total_tokens = sum(msg.get("token_count", 0) for msg in messages_to_compact)
        
        summary_text, summary_token_count = await self._generate_summary_text_with_llm(text_to_summarize, original_total_tokens)

        if not summary_text or summary_token_count == 0:
            logger.warning("Compaction resulted in no summary. Nothing will be stored in Neo4j.")
            return None, 0, None

        source_episode_uuids = [msg.get("uuid") for msg in messages_to_compact if msg.get("uuid") is not None]
        
        summary_embedding = self.embedding_client.get_embedding(summary_text)
        if not summary_embedding:
             logger.warning(f"Could not generate embedding for summary text: {summary_text[:100]}...")
             # Fallback to zero vector or handle as per requirements
             summary_embedding = [0.0] * (settings.EMBEDDING_DIMENSIONS or 384)


        compaction_data = CompactionNode(
            summary_text=summary_text,
            original_token_span=original_total_tokens,
            source_episode_uuids=source_episode_uuids,
            embedding=summary_embedding
        )
        
        try:
            props_to_store = compaction_data.model_dump(exclude_none=True)
            compaction_node_uuid = self.neo4j_adapter.add_node(
                node_label=NodeLabel.COMPACTION,
                properties=props_to_store
            )
            logger.info(f"Stored compaction summary node {compaction_node_uuid} in Neo4j for {len(source_episode_uuids)} original messages.")
            return summary_text, summary_token_count, compaction_node_uuid
        except Exception as e:
            logger.error(f"Failed to store compaction summary node in Neo4j: {e}", exc_info=True)
            return summary_text, summary_token_count, None


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4

    class MockLLMClient(LLMClient):
        def __init__(self): # No need to call super().__init__ if we mock all methods used
            self.mode = "mock" 
        def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> Tuple[str, float]:
            logger.info(f"MockLLMClient generating response for prompt (first 50 chars): '{prompt[:50]}...'")
            return f"This is a mock summary of the provided text, aiming for about {max_tokens // 2} tokens.", 0.95

    class MockNeo4jAdapter(Neo4jAdapter):
        def __init__(self):
            self._driver = True 
            logger.info("MockNeo4jAdapter initialized")
        def add_node(self, node_label: str, properties: Dict[str, Any]) -> UUID:
            new_uuid = properties.get("uuid")
            if isinstance(new_uuid, str): new_uuid = UUID(new_uuid)
            elif not isinstance(new_uuid, UUID): new_uuid = uuid4() # Ensure it's a UUID
            logger.info(f"Mocked add_node: label={node_label}, uuid={new_uuid}, summary='{properties.get('summary_text','')[:50]}...'")
            return new_uuid
        def close(self): pass

    class MockEmbeddingClient(EmbeddingClient):
        def __init__(self):
            self.model_name = "mock_embedding_model"
            self.device = "cpu"
            self.dim = settings.EMBEDDING_DIMENSIONS or 384 # Ensure dim is set
            logger.info(f"MockEmbeddingClient initialized with dim {self.dim}")
        def get_embedding(self, text: str) -> List[float]:
            if not text: return []
            return [0.01] * self.dim
        def get_embedding_dimensionality(self) -> Optional[int]:
            return self.dim


    logging.basicConfig(level=logging.INFO)
    if not settings.EMBEDDING_DIMENSIONS:
        settings.EMBEDDING_DIMENSIONS = 384
        logger.info(f"Set settings.EMBEDDING_DIMENSIONS to {settings.EMBEDDING_DIMENSIONS} for test.")

    mock_llm = MockLLMClient()
    mock_neo4j = MockNeo4jAdapter()
    mock_embed = MockEmbeddingClient() # Create mock embedding client
    
    compaction_service = CompactionService(
        llm_client=mock_llm, 
        neo4j_adapter=mock_neo4j, 
        embedding_client=mock_embed # Pass it here
    )

    test_messages = [
        {"role": "user", "content": "Hello, what is the weather like today?", "uuid": uuid4(), "token_count": 10},
        {"role": "assistant", "content": "The weather is sunny and warm.", "uuid": uuid4(), "token_count": 8},
    ]

    logger.info("\nTesting compact_and_store_memory...")
    
    async def main_test():
        summary_text, summary_tokens, summary_uuid = await compaction_service.compact_and_store_memory(test_messages)
        logger.info(f"Compaction test result: UUID={summary_uuid}, Tokens={summary_tokens}, Summary='{summary_text}'")
        assert summary_text is not None and "mock summary" in summary_text
        assert summary_tokens > 0
        assert summary_uuid is not None

        logger.info("\nTesting with empty messages...")
        empty_summary_text, empty_summary_tokens, empty_summary_uuid = await compaction_service.compact_and_store_memory([])
        assert empty_summary_text is None
        assert empty_summary_tokens == 0
        assert empty_summary_uuid is None
        
        logger.info("CompactionService tests completed.")

    asyncio.run(main_test())
