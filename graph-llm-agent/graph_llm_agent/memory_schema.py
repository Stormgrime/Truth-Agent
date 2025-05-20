from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime, timezone

class BaseNode(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5 # Default importance score
    embedding: Optional[List[float]] = None # Optional because it might be computed after node creation

class Episode(BaseNode):
    speaker: str # "user", "assistant", "system", "tool"
    content: str
    token_count: Optional[int] = None # Number of tokens in the content

    # For linking episodes in a conversation sequence
    previous_episode_uuid: Optional[UUID] = None
    next_episode_uuid: Optional[UUID] = None

class SemanticMemory(BaseNode):
    statement: str # The factual statement
    source_episode_uuids: List[UUID] = Field(default_factory=list) # UUIDs of Episodes it was derived from
    source_reflection_uuid: Optional[UUID] = None # UUID of the MetaMemory reflection it was derived from
    # Example: "Alice works at Acme Corp"
    # We can add more structured fields like subject, predicate, object if needed later

class MetaMemory(BaseNode):
    # For reflections, summaries of multiple episodes, or critiques
    summary_text: str
    covered_episode_uuids: List[UUID] = Field(default_factory=list) # Episodes this meta memory covers/summarizes
    reflection_type: Optional[str] = None # e.g., "daily_summary", "interaction_summary", "self_critique"
    salience_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

class CompactionNode(BaseNode):
    # Stores the summary of a compacted part of the context window
    summary_text: str
    original_token_span: int # How many tokens from original context this summary represents
    source_episode_uuids: List[UUID] = Field(default_factory=list) # Episodes that were part of the compacted context

# Relationships can also be modeled if we need to store properties on edges,
# or use a more formal OGM. For now, relationships are primarily managed by
# linking UUIDs within the node models or by direct Cypher queries in the adapter.

# Example of a relationship model (if using an OGM or more detailed edge properties)
# class Relationship(BaseModel):
#     start_node_uuid: UUID
#     end_node_uuid: UUID
#     type: str # e.g., "MENTIONS", "SUMMARIZES", "SUPERSEDED_BY"
#     properties: Optional[Dict[str, Any]] = None


# Enum for Node Labels (consistent strings for Neo4j)
class NodeLabel:
    EPISODE = "Episode"
    SEMANTIC_MEMORY = "SemanticMemory"
    META_MEMORY = "MetaMemory"
    COMPACTION = "Compaction" # As per prompt for compaction node
    # Add other node types as needed, e.g., "Entity" if we extract entities

# Enum for Relationship Types (consistent strings for Neo4j)
class RelationshipType:
    NEXT_EPISODE = "NEXT_EPISODE" # Connects sequential episodes
    MENTIONS = "MENTIONS" # Episode mentions a SemanticMemory or Entity
    DERIVED_FROM = "DERIVED_FROM" # SemanticMemory derived from Episode(s)
    SUMMARIZES = "SUMMARIZES" # MetaMemory summarizes Episode(s) or CompactionNode summarizes Episode(s)
    COVERS = "COVERS" # MetaMemory covers Episode(s) (as per prompt for metacognition)
    RELATED_TO = "RELATED_TO" # Generic relationship between any two nodes
    SUPERSEDED_BY = "SUPERSEDED_BY" # Indicates a node's information is outdated by another
    COMPACTED_FROM = "COMPACTED_FROM" # CompactionNode was created from these Episodes

if __name__ == "__main__":
    # Test creating instances of the models
    user_episode = Episode(speaker="user", content="Hello, agent!")
    print("User Episode:", user_episode.model_dump_json(indent=2))

    agent_response = Episode(
        speaker="assistant",
        content="Hello, user! How can I help you today?",
        previous_episode_uuid=user_episode.uuid
    )
    user_episode.next_episode_uuid = agent_response.uuid # Manually link back for this example
    print("\nAgent Response:", agent_response.model_dump_json(indent=2))

    fact = SemanticMemory(
        statement="The sky is blue.",
        source_episode_uuids=[user_episode.uuid, agent_response.uuid],
        importance=0.8,
        source_reflection_uuid=None # Example, could be a real UUID if generated from a reflection
    )
    print("\nSemantic Memory:", fact.model_dump_json(indent=2))

    reflection = MetaMemory(
        summary_text="User and agent exchanged greetings. Agent offered help.",
        covered_episode_uuids=[user_episode.uuid, agent_response.uuid],
        reflection_type="interaction_summary",
        salience_score=0.75
    )
    print("\nMeta Memory:", reflection.model_dump_json(indent=2))

    compaction_summary = CompactionNode(
        summary_text="Greetings exchanged.",
        original_token_span=20,
        source_episode_uuids=[user_episode.uuid, agent_response.uuid]
    )
    print("\nCompaction Node:", compaction_summary.model_dump_json(indent=2))

    print(f"\nNode label for Episode: {NodeLabel.EPISODE}")
    print(f"Relationship type for NEXT_EPISODE: {RelationshipType.NEXT_EPISODE}")
