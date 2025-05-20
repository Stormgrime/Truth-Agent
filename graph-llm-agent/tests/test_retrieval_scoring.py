import pytest
import math
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta

# Ensure src modules can be imported. If running pytest from root, this might need adjustments
# to sys.path or using `python -m pytest`. For now, assuming direct import works.
from src.retrieval import RetrievalService
from src.neo4j_adapter import Neo4jAdapter # For mocking
from src.embedding_client import EmbeddingClient # For mocking
from src.config import settings
from src.memory_schema import NodeLabel # For mock data

# --- Mocks ---
class MockNeo4jAdapter(Neo4jAdapter):
    def __init__(self):
        self._driver = True # Mock connected state
        self.mock_vector_query_results: List[Dict[str, Any]] = []
        # Mock the _embedding_client_for_compaction_fallback if it's part of the __init__
        # For this test, it's not directly used by RetrievalService, so can be omitted if not in base __init__
        # self._embedding_client_for_compaction_fallback = None 


    def query_vector_nodes(self, index_name: str, knn_embedding: List[float], k_value: int) -> List[Dict[str, Any]]:
        # Return a slice of the mock results based on k_value
        # The RetrievalService might ask for k_value*2, so ensure mock can handle that
        return self.mock_vector_query_results[:k_value] 
    
    def ensure_schema(self): pass # Mock
    def close(self): pass # Mock

class MockEmbeddingClient(EmbeddingClient):
    def __init__(self):
        self.model_name = "mock_model"
        self.device = "cpu"
        # Ensure settings.EMBEDDING_DIMENSIONS is set for tests, RetrievalService might check it.
        self.dim = settings.EMBEDDING_DIMENSIONS or 384 
        if not settings.EMBEDDING_DIMENSIONS:
            settings.EMBEDDING_DIMENSIONS = self.dim


    def get_embedding(self, text: str) -> List[float]:
        return [0.1] * self.dim
    
    def get_embedding_dimensionality(self) -> Optional[int]:
        return self.dim

@pytest.fixture
def mock_embedding_client_fixture() -> MockEmbeddingClient: # Renamed to avoid conflict if there's a global one
    return MockEmbeddingClient()

@pytest.fixture
def mock_neo4j_adapter_fixture() -> MockNeo4jAdapter: # Renamed
    return MockNeo4jAdapter()

@pytest.fixture
def retrieval_service_fixture(mock_neo4j_adapter_fixture: MockNeo4jAdapter, mock_embedding_client_fixture: MockEmbeddingClient) -> RetrievalService:
    return RetrievalService(neo4j_adapter=mock_neo4j_adapter_fixture, embedding_client=mock_embedding_client_fixture)

def create_mock_episode_data(uuid_val: UUID, content: str, hours_ago: float, 
                             importance: float, similarity_score: float) -> Dict[str, Any]:
    timestamp = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "uuid": uuid_val, 
        "content": content, 
        "speaker": "user", 
        "timestamp": timestamp, 
        "importance": importance, 
        "labels": [NodeLabel.EPISODE], 
        "score": similarity_score 
    }

def test_retrieval_scoring_order(retrieval_service_fixture: RetrievalService, mock_neo4j_adapter_fixture: MockNeo4jAdapter):
    mem1_uuid = uuid4()
    mem2_uuid = uuid4()
    mem3_uuid = uuid4()
    mem4_uuid = uuid4()

    mock_neo4j_adapter_fixture.mock_vector_query_results = [
        create_mock_episode_data(mem1_uuid, "Memory 1", hours_ago=1, importance=0.6, similarity_score=0.9),      # Expected final ~0.8877
        create_mock_episode_data(mem2_uuid, "Memory 2", hours_ago=24, importance=0.9, similarity_score=0.7),     # Expected final ~0.6203
        create_mock_episode_data(mem3_uuid, "Memory 3", hours_ago=100, importance=0.2, similarity_score=0.5),    # Expected final ~0.3246
        create_mock_episode_data(mem4_uuid, "Memory 4", hours_ago=200, importance=0.9, similarity_score=0.9),    # Expected final ~0.6301
    ]
    # RetrievalService asks for k_value*2 from neo4j_adapter.query_vector_nodes.
    # So if k_value=4, it asks for 8. Our mock_vector_query_results has 4 items.
    # The mock query_vector_nodes returns [:k_value], so it will respect the k_value*2.

    retrieved_memories = retrieval_service_fixture.retrieve_relevant_memories("test query", k_value=4)

    assert len(retrieved_memories) == 4 # Should return all 4 after scoring
    
    final_scores = [mem['final_score'] for mem in retrieved_memories]
    retrieved_uuids_ordered = [mem['uuid'] for mem in retrieved_memories]

    for mem in retrieved_memories:
        print(f"UUID: {mem['uuid']}, Final Score: {mem['final_score']:.4f}, Sim: {mem['similarity_score']:.2f}, Age: {mem['calculated_age_hours']:.2f}h, Imp: {mem['importance']:.2f}")

    for i in range(len(final_scores) - 1):
        assert final_scores[i] >= final_scores[i+1], "Memories are not sorted correctly by final_score"

    # Expected order based on manual calculation: Mem1 > Mem4 > Mem2 > Mem3
    assert retrieved_uuids_ordered[0] == mem1_uuid # Highest score
    assert retrieved_uuids_ordered[1] == mem4_uuid # Second highest
    assert retrieved_uuids_ordered[2] == mem2_uuid # Third highest
    assert retrieved_uuids_ordered[3] == mem3_uuid # Lowest score


def test_retrieval_empty_query(retrieval_service_fixture: RetrievalService):
    retrieved_memories = retrieval_service_fixture.retrieve_relevant_memories("")
    assert len(retrieved_memories) == 0

def test_retrieval_no_results_from_db(retrieval_service_fixture: RetrievalService, mock_neo4j_adapter_fixture: MockNeo4jAdapter):
    mock_neo4j_adapter_fixture.mock_vector_query_results = []
    retrieved_memories = retrieval_service_fixture.retrieve_relevant_memories("test query")
    assert len(retrieved_memories) == 0

def test_retrieval_k_value_respected(retrieval_service_fixture: RetrievalService, mock_neo4j_adapter_fixture: MockNeo4jAdapter):
    results = [create_mock_episode_data(uuid4(), f"Mem {i}", i*10 + 1, 0.5, 0.8-(i*0.05)) for i in range(5)] # 5 mock results
    mock_neo4j_adapter_fixture.mock_vector_query_results = results 
    # RetrievalService will ask for k_value*2 from the mock adapter's query_vector_nodes.
    # The mock adapter's query_vector_nodes will return [:k_value_asked_by_retrieval_service]
    # So if k_value=2 for retrieve_relevant_memories, RetrievalService asks for 4 from DB. Mock DB returns min(4, len(results)).
    # Then RetrievalService itself slices to original k_value.

    retrieved_k2 = retrieval_service_fixture.retrieve_relevant_memories("test query", k_value=2)
    assert len(retrieved_k2) == 2
    
    retrieved_k4 = retrieval_service_fixture.retrieve_relevant_memories("test query", k_value=4)
    assert len(retrieved_k4) == 4 
    
    retrieved_k10 = retrieval_service_fixture.retrieve_relevant_memories("test query", k_value=10)
    # RetrievalService asks for 20. Mock DB returns all 5. RetrievalService then returns min(10, 5) = 5.
    assert len(retrieved_k10) == 5
