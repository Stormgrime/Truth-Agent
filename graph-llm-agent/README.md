# Graph-LLM Agent

**A Large Language Model (LLM) agent with persistent, graph-based memory.**

This project implements an LLM agent that leverages a graph database (Neo4j) to store its interactions and learned knowledge, enabling long-term memory and recall beyond the limitations of a fixed context window. The agent uses retrieval-augmented generation (RAG) with a hybrid scoring mechanism (semantic similarity, recency, importance) to fetch relevant memories for providing context to the LLM.

## Core Features

- **Persistent Graph Memory:** Uses Neo4j to store episodic, semantic, and metacognitive memories.
- **Hybrid Retrieval:** Combines vector similarity search with recency and importance scores.
- **Context Management:** Employs a sliding window for conversation history with LLM-based compaction for older turns.
- **Uncertainty Arbitration:** Uses a two-pass LLM generation (with and without memory) to decide on response confidence and strategy (answer, hedge, clarify, IDK).
- **Metacognition:** Periodically reflects on interactions to generate summaries and extract new knowledge, storing these as `MetaMemory` nodes.
- **Local First:** Designed to run locally using Poetry for dependency management. No Docker required for core functionality.
- **Configurable LLM:** Supports configurable LLM models (default Qwen family, tested with Hugging Face `transformers` or vLLM-compatible API).

## Architecture Overview (Conceptual)

```
+---------------------+   Retrieves   +----------------------+   Provides   +---------------------+
|      User Query     |------------->|   Retrieval Service  |------------->|    Context Manager  |
+---------------------+              |(Vector Search, Hybrid|              |(Manages prompt size,|
                                     | Scoring: Sim,Rec,Imp)|              | calls Compaction)  |
         ^                           +----------------------+              +---------------------+
         |                                       ^                                      |
         | Stores Interaction                    | Stores Memories                      | Assembles Prompt
         | (User)                                | (Episodes, Facts)                    | (System, Memories, History)
         |                                       |                                      v
+---------------------+              +----------------------+              +---------------------+
|    Agent Memory     |<-------------|    Neo4j Adapter     |<-------------|      LLM Client     |
|(Orchestrates memory |   Writes To   |(Handles DB I/O,      |   Gets Resp.  |(Interface to local |
| operations,        |<-------------| Vector Indexes)      |   from        | or remote LLM)     |
| triggers reflection)|              +----------------------+              +---------------------+
+---------------------+                                       ^                                      |
         |                                       |                                      | Stores Interaction
         | Stores Interaction                    | Reflects Using                       | (Assistant)
         | (Assistant Output)                    |                                      v
         v                           +----------------------+              +---------------------+
+---------------------+              | Metacognition Service|------------->| Uncertainty Arbiter |
|   User Sees Reply   |<-------------|(Summaries, Fact Extr.)|   Decides    |(Chooses best answer |
+---------------------+              +----------------------+   Final Resp. | based on confidence)|
                                                                           +---------------------+
```

**Key Flow:**
1. User sends a query.
2. `AgentMemory` stores the user interaction (Episode node).
3. `RetrievalService` fetches relevant memories from Neo4j (vector search + scoring).
4. `ContextManager` assembles the prompt using system instructions, retrieved memories, and recent conversation history (from its deque). It triggers `CompactionService` if context window grows too large.
5. `CompactionService` summarizes old messages using the LLM and stores a `CompactionNode`.
6. `UncertaintyArbiter` uses the `LLMClient` for two passes (with/without memories) to decide on the best response and policy.
7. `LLMClient` generates text based on the prompt.
8. The final answer is presented to the user.
9. `AgentMemory` stores the assistant's response (Episode node).
10. `MetacognitionService` periodically reflects on stored episodes, creating `MetaMemory` and `SemanticMemory` nodes.

## Getting Started

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- A running Neo4j instance (Community Edition 4.4 or 5.x recommended for vector index support).
    - Ensure the Neo4j instance has the APOC and Graph Data Science (GDS) plugins installed if you plan to use advanced graph algorithms or vector indexes via GDS (though native vector indexes in 5.x are preferred). The current `Neo4jAdapter` uses native vector indexes.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/graph-llm-agent.git # Replace with actual repo URL
    cd graph-llm-agent
    ```

2.  **Set up environment variables:**
    Copy the sample environment file and customize it with your settings (especially Neo4j credentials):
    ```bash
    cp .env.sample .env
    nano .env # Or your favorite editor
    ```
    Ensure `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` are correct. You might also want to set `LLM_MODEL_NAME` if you don't want the default.

3.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
    This will create a virtual environment (if one isn't active) and install all necessary packages.

### Running the Agent

1.  **Activate the virtual environment (if not already active):**
    ```bash
    poetry shell
    ```

2.  **Run the smoke test (recommended first run):**
    This initializes services, performs a basic interaction with Neo4j and the LLM, and then exits. It helps verify that essential components are working.
    ```bash
    python -m src.main smoke
    ```
    You can specify a smaller model for the smoke test if needed (especially for CI or resource-constrained environments):
    ```bash
    python -m src.main smoke --model "hf-internal-testing/tiny-random-gpt2"
    ```

3.  **Run unit tests:**
    ```bash
    pytest -q
    ```

4.  **Start the interactive chat REPL:**
    ```bash
    python -m src.main chat
    ```
    You can override the default model or API endpoint:
    ```bash
    python -m src.main chat --model "mistralai/Mistral-7B-Instruct-v0.1"
    # Or for a vLLM endpoint:
    # python -m src.main chat --llm-api-base "http://localhost:8000/v1" --model "meta-llama/Llama-2-7b-chat-hf" 
    ```

## Design Principles

- **Modularity:** Components are designed to be as independent as possible.
- **Local First:** Prioritizes running on local hardware without mandatory cloud dependencies (except for model downloads).
- **Append-Only Memory:** Knowledge is accumulated; old information is marked as superseded rather than deleted, preserving history.
- **Explicit Metacognition:** Reflection and self-summary are distinct processes that enrich the agent's memory graph.
- **Configurability:** Key parameters (model names, thresholds, etc.) are managed via `src/config.py` and environment variables.

## Licensing

- The code within this repository is licensed under the **Apache-2.0 License**. See the `LICENSE` file (to be added) for more details.
- **Neo4j Community Edition:** Please be aware that Neo4j Community Edition is licensed under the **GPLv3**. If you use Neo4j CE, ensure compliance with its terms. This project interacts with Neo4j as an external, user-installed dependency.

## Development & Contributing

(Placeholder for future development roadmap, contribution guidelines, etc.)

- Key areas for future work:
    - More sophisticated retrieval strategies (e.g., querying multiple indexes, knowledge graph traversals).
    - Advanced metacognitive routines (e.g., learning from feedback, automated fact validation).
    - Fine-tuning LLMs on conversation history or extracted knowledge.
    - UI/UX improvements beyond the CLI.
```
