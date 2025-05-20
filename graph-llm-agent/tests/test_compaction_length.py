import pytest
import asyncio
import tiktoken # For token counting in test
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

from graph_llm_agent.compaction import CompactionService
from graph_llm_agent.llm_client import LLMClient # For mocking
from graph_llm_agent.neo4j_adapter import Neo4jAdapter # For mocking
from graph_llm_agent.embedding_client import EmbeddingClient # For mocking
from graph_llm_agent.config import settings 
from graph_llm_agent.memory_schema import NodeLabel # For mock data

# --- Mocks ---
class MockLLMClientCompaction(LLMClient):
    def __init__(self):
        self.mode = "mock" 
        self.expected_summary_max_tokens = 250 

    def generate_response(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, float]:
        # max_tokens here is what CompactionService passes (e.g., target_summary_len + buffer)
        # CompactionService calculates its internal target_summary_len (e.g. 250 for 5k input)
        # This mock should generate text that has that internal target_summary_len tokens.
        # For a 5k input, CompactionService's target_summary_len is 250. It calls LLM with max_tokens=300.
        # So, this mock should generate 250 tokens.
        num_tokens_to_generate = max_tokens - 50 # This should be CompactionService's internal target
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        example_token_sequence = tokenizer.encode("sample summary content ") # ensure it's a few tokens
        
        num_sequences_needed = (num_tokens_to_generate + len(example_token_sequence) -1) // len(example_token_sequence)
        mock_summary_tokens = []
        for _ in range(num_sequences_needed):
            mock_summary_tokens.extend(example_token_sequence)
        
        mock_summary_tokens = mock_summary_tokens[:num_tokens_to_generate]
        mock_summary_text = tokenizer.decode(mock_summary_tokens)
        
        return mock_summary_text, 0.95

class MockNeo4jAdapterCompaction(Neo4jAdapter):
    def __init__(self):
        self._driver = True
    def add_node(self, node_label: str, properties: Dict[str, Any]) -> UUID:
        return properties.get("uuid", uuid4())
    def ensure_schema(self): pass
    def close(self): pass

class MockEmbeddingClientCompaction(EmbeddingClient):
    def __init__(self):
        self.dim = settings.EMBEDDING_DIMENSIONS or 384
        if not settings.EMBEDDING_DIMENSIONS: settings.EMBEDDING_DIMENSIONS = self.dim
    def get_embedding(self, text: str) -> List[float]:
        if not text: return []
        return [0.1] * self.dim
    def get_embedding_dimensionality(self) -> Optional[int]:
        return self.dim

@pytest.fixture
def mock_llm_client_compaction_fixture() -> MockLLMClientCompaction: # Renamed fixture
    return MockLLMClientCompaction()

@pytest.fixture
def mock_neo4j_adapter_compaction_fixture() -> MockNeo4jAdapterCompaction: # Renamed fixture
    return MockNeo4jAdapterCompaction()

@pytest.fixture
def mock_embedding_client_compaction_fixture() -> MockEmbeddingClientCompaction: # Renamed fixture
    return MockEmbeddingClientCompaction()

@pytest.fixture
def compaction_service_fixture(
    mock_llm_client_compaction_fixture: MockLLMClientCompaction, 
    mock_neo4j_adapter_compaction_fixture: MockNeo4jAdapterCompaction,
    mock_embedding_client_compaction_fixture: MockEmbeddingClientCompaction
) -> CompactionService:
    return CompactionService(
        llm_client=mock_llm_client_compaction_fixture, 
        neo4j_adapter=mock_neo4j_adapter_compaction_fixture,
        embedding_client=mock_embedding_client_compaction_fixture
    )

def generate_long_transcript(target_tokens: int) -> List[Dict[str, Any]]:
    messages = []
    current_tokens = 0
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    sentence = "This is a test sentence for generating a long transcript content item part. " 
    sentence_tokens = len(tokenizer.encode(sentence))
    if sentence_tokens == 0: sentence_tokens = 1 # Avoid division by zero for empty sentence_tokens

    num_messages_needed = (target_tokens + sentence_tokens - 1) // sentence_tokens
    
    for i in range(num_messages_needed):
        role = "user" if i % 2 == 0 else "assistant"
        # Construct slightly varying content to avoid identical embeddings if that mattered
        content = sentence + f"ID {i}." 
        actual_tokens = len(tokenizer.encode(content))
        messages.append({
            "role": role, "content": content,
            "uuid": uuid4(), "token_count": actual_tokens 
        })
        current_tokens += actual_tokens
        if current_tokens >= target_tokens: break
            
    # print(f"DEBUG: Generated transcript with {current_tokens} tokens from {len(messages)} messages for target {target_tokens}.")
    return messages

@pytest.mark.asyncio 
async def test_compaction_summary_length(compaction_service_fixture: CompactionService):
    target_transcript_tokens = 5000
    # CompactionService aims for max_summary_tokens = max(50, min(250, int(original_total_tokens * 0.2)))
    # For 5000 tokens, this is 250. MockLLM is set to return this many tokens.
    expected_summary_tokens_from_mock = 250 
    
    long_transcript_messages = generate_long_transcript(target_transcript_tokens)
    
    summary_text, summary_token_count, summary_uuid = await compaction_service_fixture.compact_and_store_memory(long_transcript_messages)

    assert summary_text is not None, "Compaction failed to produce a summary text."
    assert summary_token_count > 0, "Summary token count is zero."
    
    # print(f"DEBUG: Original transcript tokens (summed): {sum(m['token_count'] for m in long_transcript_messages)}")
    # print(f"DEBUG: Summary text (first 100 chars): '{summary_text[:100]}...'")
    # print(f"DEBUG: Actual summary token count from service: {summary_token_count}")

    assert summary_token_count <= 300, f"Summary token count {summary_token_count} exceeded 300 (user requirement)."
    # Check if it's close to what the mock was designed to produce (CompactionService's internal target)
    assert abs(summary_token_count - expected_summary_tokens_from_mock) <= 5,         f"Summary token count {summary_token_count} significantly different from mock's target {expected_summary_tokens_from_mock}."

```
