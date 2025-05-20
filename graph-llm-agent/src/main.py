import logging
import asyncio 
from typing import List, Dict, Optional, Any
from uuid import uuid4, UUID

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.panel import Panel

from src.config import settings
from src.embedding_client import EmbeddingClient
from src.llm_client import LLMClient
from src.neo4j_adapter import Neo4jAdapter
from src.retrieval import RetrievalService
from src.context_manager import ContextManager
from src.compaction import CompactionService
from src.metacognition import MetacognitionService
from src.uncertainty import UncertaintyArbiter
from src.agent_memory import AgentMemory

logging.basicConfig(level=settings.LOG_LEVEL.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, no_args_is_help=True)
console = Console()

neo4j_adapter: Optional[Neo4jAdapter] = None
embedding_client: Optional[EmbeddingClient] = None
llm_client: Optional[LLMClient] = None
retrieval_service: Optional[RetrievalService] = None
compaction_service: Optional[CompactionService] = None
context_manager: Optional[ContextManager] = None
metacognition_service: Optional[MetacognitionService] = None
uncertainty_arbiter: Optional[UncertaintyArbiter] = None
agent_memory: Optional[AgentMemory] = None

def initialize_services(model_override: Optional[str] = None, llm_api_base_url_override: Optional[str] = None) -> bool:
    global neo4j_adapter, embedding_client, llm_client, retrieval_service
    global compaction_service, context_manager, metacognition_service
    global uncertainty_arbiter, agent_memory

    logger.info("Initializing services...")
    try:
        neo4j_adapter = Neo4jAdapter()
        neo4j_adapter.ensure_schema()

        embedding_client = EmbeddingClient()
        
        current_llm_model = model_override if model_override else settings.LLM_MODEL_NAME
        current_llm_api_base = llm_api_base_url_override if llm_api_base_url_override else settings.LLM_API_BASE_URL
        
        llm_client = LLMClient(model_name=current_llm_model, llm_api_base_url=current_llm_api_base)
        
        retrieval_service = RetrievalService(neo4j_adapter, embedding_client)
        compaction_service = CompactionService(llm_client, neo4j_adapter, embedding_client)
        context_manager = ContextManager(compaction_service)
        metacognition_service = MetacognitionService(neo4j_adapter, llm_client, embedding_client)
        uncertainty_arbiter = UncertaintyArbiter(llm_client)
        
        agent_memory = AgentMemory(
            neo4j_adapter=neo4j_adapter, embedding_client=embedding_client,
            retrieval_service=retrieval_service, context_manager=context_manager,
            metacognition_service=metacognition_service, llm_client=llm_client
        )
        logger.info("All services initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Error initializing services:[/bold red]{{e}}", title="Initialization Error", border_style="red"))
        return False

def format_messages_to_prompt_string(messages: List[Dict[str, str]]) -> str:
    formatted_prompt = ""
    for msg in messages:
        role = msg.get("role", "system") 
        content = msg.get("content", "")
        formatted_prompt += f"{role.capitalize()}: {content}\n"
    return formatted_prompt.strip()

@app.command()
def chat(
    model: Optional[str] = typer.Option(settings.LLM_MODEL_NAME, "--model", "-m", help=f"LLM model name. Overrides config."),
    llm_api_base_url: Optional[str] = typer.Option(settings.LLM_API_BASE_URL, "--llm-api-base", help="Base URL for vLLM or OpenAI-compatible API. Overrides config.")
):
    """Interactive REPL (chat loop) with the Graph-LLM Agent."""
    effective_model = model if model != settings.LLM_MODEL_NAME else settings.LLM_MODEL_NAME
    effective_api_base = llm_api_base_url if llm_api_base_url != settings.LLM_API_BASE_URL else settings.LLM_API_BASE_URL
    
    if not initialize_services(model_override=effective_model, llm_api_base_url_override=effective_api_base):
        raise typer.Exit(code=1)

    console.print(Panel("[bold green]Graph-LLM Agent Chat Initialized[/bold green]", title="Welcome!", expand=False))
    console.print(f"Using LLM: [cyan]{llm_client.model_name if llm_client else 'Unknown'}[/cyan]")
    if llm_client and llm_client.api_base_url:
        console.print(f"LLM API Endpoint: [cyan]{llm_client.api_base_url}[/cyan]")
    console.print("Type 'exit' or 'quit' to end the chat.")
    
    last_episode_uuid: Optional[UUID] = None

    while True:
        try:
            user_query = Prompt.ask(Text("You", style="bold blue"))
            if user_query.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Exiting chat. Goodbye![/bold yellow]")
                break
            if not user_query.strip(): continue

            current_user_episode_uuid = uuid4()
            asyncio.run(agent_memory.add_interaction(
                "user", user_query, current_user_episode_uuid, last_episode_uuid
            ))
            last_episode_uuid = current_user_episode_uuid

            retrieved_memories = agent_memory.retrieve_context_for_query(user_query)
            if retrieved_memories:
                console.print(Text(f"Retrieved {len(retrieved_memories)} memories for consideration.", style="italic dim"))

            prompt_A_messages = context_manager.get_current_context_for_prompt(retrieved_memories=None)
            prompt_A_str = format_messages_to_prompt_string(prompt_A_messages)
            
            prompt_B_str: Optional[str] = None
            if retrieved_memories:
                prompt_B_messages = context_manager.get_current_context_for_prompt(retrieved_memories=retrieved_memories)
                prompt_B_str = format_messages_to_prompt_string(prompt_B_messages)

            final_answer, policy, conf_a, conf_b, raw_a, raw_b = asyncio.run(
                uncertainty_arbiter.arbitrate(prompt_A_str, prompt_B_str)
            )
            
            console.print(Text(f"Agent ({policy}):", style="bold green"), Text(final_answer))
            logger.info(f"Policy: {policy}, ConfA: {conf_a:.2f}, ConfB: {conf_b:.2f}")

            assistant_episode_uuid = uuid4()
            asyncio.run(agent_memory.add_interaction(
                "assistant", final_answer, assistant_episode_uuid, current_user_episode_uuid
            ))
            last_episode_uuid = assistant_episode_uuid

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting chat. Goodbye![/bold yellow]")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            console.print(f"[bold red]An error occurred:[/bold red] {e}")
    if neo4j_adapter: neo4j_adapter.close()

@app.command()
def smoke(
    model: Optional[str] = typer.Option(settings.LLM_MODEL_NAME, "--model", "-m", help="LLM model for smoke test."),
    llm_api_base_url: Optional[str] = typer.Option(settings.LLM_API_BASE_URL, "--llm-api-base", help="LLM API base URL for smoke test.")
):
    """Run a self-check: initialize services, simple DB R/W, LLM call, then exits."""
    console.print(Panel("[bold yellow]Smoke Test Initiated[/bold yellow]", expand=False))
    effective_model = model if model != settings.LLM_MODEL_NAME else settings.LLM_MODEL_NAME
    effective_api_base = llm_api_base_url if llm_api_base_url != settings.LLM_API_BASE_URL else settings.LLM_API_BASE_URL
    
    if not initialize_services(model_override=effective_model, llm_api_base_url_override=effective_api_base):
        console.print("[bold red]Smoke Test Failed: Initialization error.[/bold red]")
        raise typer.Exit(code=1)
    try:
        console.print(f"Smoke test using LLM: [cyan]{llm_client.model_name if llm_client else 'Unknown'}[/cyan]")
        if llm_client and llm_client.api_base_url:
             console.print(f"LLM API Endpoint: [cyan]{llm_client.api_base_url}[/cyan]")

        console.print("Step 1: Adding user interaction...")
        smoke_user_uuid = uuid4()
        asyncio.run(agent_memory.add_interaction("user", "Hello, this is a smoke test message.", smoke_user_uuid))
        console.print(f"  Added user episode: {smoke_user_uuid}")

        console.print("Step 2: Retrieving context...")
        memories = agent_memory.retrieve_context_for_query("smoke test message")
        assert memories is not None, "Smoke Test Failed: Retrieving memories returned None."
        console.print(f"  Retrieved {len(memories)} memories.") 

        console.print("Step 3: Generating LLM response (direct call, not full arbitration)...")
        current_ctx_messages = context_manager.get_current_context_for_prompt()
        smoke_prompt_str = format_messages_to_prompt_string(current_ctx_messages)
        smoke_prompt_str += "\nAssistant (respond very briefly, e.g. 'Okay.'):" 
        
        assistant_reply, confidence = llm_client.generate_response(smoke_prompt_str, max_tokens=15)
        assert "Error:" not in assistant_reply and confidence >= 0.0, f"Smoke Test Failed: LLM error - '{assistant_reply}' with conf {confidence}"
        console.print(f"  LLM generated (conf: {confidence:.2f}): '{assistant_reply}'")

        console.print("Step 4: Adding assistant interaction...")
        smoke_assistant_uuid = uuid4()
        asyncio.run(agent_memory.add_interaction("assistant", assistant_reply, smoke_assistant_uuid, smoke_user_uuid))
        console.print(f"  Added assistant episode: {smoke_assistant_uuid}")
        
        console.print("[bold green]Smoke Test Passed Successfully![/bold green]")
    except AssertionError as e:
        logger.error(f"Smoke Test Assertion Failed: {e}", exc_info=True)
        console.print(f"[bold red]Smoke Test Failed: {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Smoke Test Failed: {e}", exc_info=True)
        console.print(f"[bold red]Smoke Test Failed: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        if neo4j_adapter: neo4j_adapter.close()

@app.command()
def reflect(force: bool = typer.Option(False, "--force", help="Force reflection.")): 
    """Manually trigger metacognitive reflection."""
    if not initialize_services(): raise typer.Exit(code=1)
    console.print(Panel("[bold yellow]Manual Reflection Triggered[/bold yellow]", expand=False))
    try:
        if len(context_manager.context_window) > 0:
            episodes_to_reflect = [dict(item) for item in context_manager.context_window]
            console.print(f"Reflecting on {len(episodes_to_reflect)} episodes from context window...")
            meta_uuid = asyncio.run(metacognition_service.reflect_on_episodes(episodes_to_reflect))
            if meta_uuid:
                console.print(f"[bold green]Reflection complete. MetaMemory: {meta_uuid}[/bold green]")
            else:
                console.print("[bold red]Reflection did not complete or create MetaMemory.[/bold red]")
        else:
            console.print("[yellow]No episodes in context window to reflect upon for this manual trigger.[/yellow]")
    except Exception as e:
        logger.error(f"Manual reflection failed: {e}", exc_info=True)
        console.print(f"[bold red]Manual Reflection Failed: {e}[/bold red]")
    finally:
        if neo4j_adapter: neo4j_adapter.close()

if __name__ == "__main__":
    app()
