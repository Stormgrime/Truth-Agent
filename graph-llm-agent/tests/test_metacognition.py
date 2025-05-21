import pytest
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

from graph_llm_agent.metacognition import MetacognitionService
from graph_llm_agent.llm_client import LLMClient
from graph_llm_agent.neo4j_adapter import Neo4jAdapter
from graph_llm_agent.embedding_client import EmbeddingClient
from graph_llm_agent.config import settings
from graph_llm_agent.memory_schema import NodeLabel, RelationshipType, MetaMemory, SemanticMemory # For asserting node types and rels

# --- Mocks ---
class MockLLMMeta(LLMClient):
    def __init__(self):
        self.mode = "mock"
        self.salience_score_to_return = 0.75
        self.facts_to_return = ["Fact one from mock.", "Fact two from mock."]
        self.summary_to_return = "This is a mock reflection summary from LLM."

    def generate_response(self, prompt: str, temperature: float, max_tokens: int) -> Tuple[str, float]:
        response_json = {
            "summary": self.summary_to_return,
            "facts": self.facts_to_return,
            "salience_score": self.salience_score_to_return
        }
        return json.dumps(response_json), 0.98

class MockNeo4jAdapterMeta(Neo4jAdapter):
    def __init__(self):
        self._driver = True
        self.nodes_added: Dict[UUID, Dict[str, Any]] = {}
        self.rels_added: List[Dict[str, Any]] = []

    def add_node(self, node_label: str, properties: Dict[str, Any]) -> UUID:
        new_uuid = UUID(str(properties.get("uuid", uuid4()))) # Ensure it's a UUID object
        self.nodes_added[new_uuid] = {"label": node_label, "props": properties}
        return new_uuid

    def add_relationship(self, start_node_uuid: UUID, end_node_uuid: UUID, 
                         start_node_label: str, end_node_label: str, 
                         rel_type: str, properties: Optional[Dict[str, Any]] = None):
        self.rels_added.append({
            "from": start_node_uuid, "to": end_node_uuid, 
            "s_label": start_node_label, "e_label": end_node_label, 
            "type": rel_type, "props": properties or {}
        })
    
    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Any]:
        # For importance update
        if "SET e.importance" in query or "SET ep.importance" in query : # Adjusted to match metacognition.py
            node_uuid_to_update = UUID(str(parameters.get('uuid')))
            if node_uuid_to_update in self.nodes_added and self.nodes_added[node_uuid_to_update]['props'].get('uuid') == node_uuid_to_update: # Check if it's an episode
                 self.nodes_added[node_uuid_to_update]['props']['importance'] = parameters.get('importance')
            return [] 
        return []
    
    def ensure_schema(self): pass
    def close(self): pass


class MockEmbeddingClientMeta(EmbeddingClient):
    def __init__(self):
        self.dim = settings.EMBEDDING_DIMENSIONS or 384
        if not settings.EMBEDDING_DIMENSIONS: settings.EMBEDDING_DIMENSIONS = self.dim
    def get_embedding(self, text: str) -> List[float]:
        if not text: return []
        return [0.3] * self.dim
    def get_embedding_dimensionality(self) -> Optional[int]:
        return self.dim

@pytest.fixture
def mock_llm_meta_fixture() -> MockLLMMeta:
    return MockLLMMeta()

@pytest.fixture
def mock_neo4j_meta_fixture() -> MockNeo4jAdapterMeta:
    return MockNeo4jAdapterMeta()

@pytest.fixture
def mock_embedding_meta_fixture() -> MockEmbeddingClientMeta:
    return MockEmbeddingClientMeta()

@pytest.fixture
def metacognition_service_fixture(
    mock_neo4j_meta_fixture: MockNeo4jAdapterMeta,
    mock_llm_meta_fixture: MockLLMMeta,
    mock_embedding_meta_fixture: MockEmbeddingClientMeta
) -> MetacognitionService:
    return MetacognitionService(
        neo4j_adapter=mock_neo4j_meta_fixture,
        llm_client=mock_llm_meta_fixture,
        embedding_client=mock_embedding_meta_fixture
    )

@pytest.mark.asyncio
async def test_reflect_on_episodes_creates_nodes_and_relationships(
    metacognition_service_fixture: MetacognitionService, 
    mock_neo4j_meta_fixture: MockNeo4jAdapterMeta,
    mock_llm_meta_fixture: MockLLMMeta # To access facts_to_return for assertion count
):
    # Prepare 3 dummy episodes data
    ep1_uuid, ep2_uuid, ep3_uuid = uuid4(), uuid4(), uuid4()
    test_episodes_data = [
        {"uuid": str(ep1_uuid), "role": "user", "content": "Episode 1 content", "token_count": 5, "importance": 0.4},
        {"uuid": str(ep2_uuid), "role": "assistant", "content": "Episode 2 content", "token_count": 5, "importance": 0.5},
        {"uuid": str(ep3_uuid), "role": "user", "content": "Episode 3 content", "token_count": 5, "importance": 0.6},
    ]
    # Pre-populate mock_neo4j_meta_fixture.nodes_added if importance update logic reads before writing
    for ep_data in test_episodes_data:
         mock_neo4j_meta_fixture.nodes_added[UUID(ep_data["uuid"])] = {"label": NodeLabel.EPISODE, "props": ep_data.copy()}


    meta_mem_uuid = await metacognition_service_fixture.reflect_on_episodes(test_episodes_data)

    assert meta_mem_uuid is not None, "Reflection did not return a MetaMemory UUID."

    # 1. Check MetaMemory node creation
    assert meta_mem_uuid in mock_neo4j_meta_fixture.nodes_added
    meta_node_info = mock_neo4j_meta_fixture.nodes_added[meta_mem_uuid]
    assert meta_node_info["label"] == NodeLabel.META_MEMORY
    assert meta_node_info["props"]["summary_text"] == mock_llm_meta_fixture.summary_to_return
    assert meta_node_info["props"]["salience_score"] == mock_llm_meta_fixture.salience_score_to_return
    assert len(meta_node_info["props"]["covered_episode_uuids"]) == len(test_episodes_data)

    # 2. Check SemanticMemory node creation (for extracted facts)
    expected_facts_count = len(mock_llm_meta_fixture.facts_to_return)
    created_semantic_nodes = 0
    for node_id, node_details in mock_neo4j_meta_fixture.nodes_added.items():
        if node_details["label"] == NodeLabel.SEMANTIC_MEMORY:
            created_semantic_nodes += 1
            # Check if it's linked from the MetaMemory node via :CAPTURES
            assert any(
                rel["from"] == meta_mem_uuid and rel["to"] == node_id and rel["type"] == RelationshipType.CAPTURES 
                for rel in mock_neo4j_meta_fixture.rels_added
            ), f"SemanticMemory node {node_id} not linked via :CAPTURES from MetaMemory {meta_mem_uuid}"
            assert node_details["props"]["source_reflection_uuid"] == str(meta_mem_uuid)

    assert created_semantic_nodes == expected_facts_count,         f"Expected {expected_facts_count} SemanticMemory nodes, found {created_semantic_nodes}."

    # 3. Check :COVERS relationships from MetaMemory to Episodes
    covers_rels_count = sum(
        1 for rel in mock_neo4j_meta_fixture.rels_added 
        if rel["type"] == RelationshipType.COVERS and rel["from"] == meta_mem_uuid
    )
    assert covers_rels_count == len(test_episodes_data),         f"Expected {len(test_episodes_data)} :COVERS relationships, found {covers_rels_count}."

    # 4. Check Salience Propagation (importance update on original episodes)
    expected_new_importance = min(1.0, 0.5 + (mock_llm_meta_fixture.salience_score_to_return / 2.0))
    for ep_data in test_episodes_data:
        ep_uuid = UUID(ep_data["uuid"])
        assert mock_neo4j_meta_fixture.nodes_added[ep_uuid]["props"]["importance"] == expected_new_importance,             f"Importance for episode {ep_uuid} not updated correctly."

```
