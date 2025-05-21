import logging
import tiktoken
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from graph_llm_agent.config import settings
# from graph_llm_agent.compaction import CompactionService # Will be implemented next
# from graph_llm_agent.llm_client import LLMClient # For tokenizer, if not using tiktoken directly

logger = logging.getLogger(__name__)

# Placeholder for CompactionService until it's implemented
class PlaceholderCompactionService:
    def compact_messages_and_get_summary_text(self, messages_to_compact: List[Dict[str, Any]]) -> Tuple[Optional[str], int]:
        logger.warning("Using PlaceholderCompactionService. Compaction will not actually occur.")
        if not messages_to_compact:
            return None, 0
        # Simulate a summary
        simulated_summary_text = f"Summary of {len(messages_to_compact)} messages (total tokens: {sum(m.get('token_count',0) for m in messages_to_compact)})."
        # Simulate summary token count (e.g. 10% of original, or fixed)
        simulated_summary_token_count = max(10, int(sum(m.get('token_count',0) for m in messages_to_compact) * 0.1))
        return simulated_summary_text, simulated_summary_token_count

    async def compact_and_store_memory(self, messages_to_compact: List[Dict[str, Any]]) -> Tuple[Optional[str], int, Optional[UUID]]:
        summary_text, token_count = self.compact_messages_and_get_summary_text(messages_to_compact)
        # This placeholder doesn't store anything, so returns None for summary_uuid
        return summary_text, token_count, None


class ContextManager:
    def __init__(self, compaction_service: Any # PlaceholderCompactionService or actual CompactionService
                ):
        self.context_window: deque[Dict[str, Any]] = deque()
        self.current_total_tokens: int = 0
        self.compaction_service = compaction_service if compaction_service else PlaceholderCompactionService()
        
        if settings.OVERRIDE_TOKENIZER:
            try:
                self.tokenizer = tiktoken.get_encoding(settings.OVERRIDE_TOKENIZER)
                logger.info(f"ContextManager: Using OVERRIDE_TOKENIZER '{settings.OVERRIDE_TOKENIZER}'.")
            except Exception as e_override:
                logger.error(f"ContextManager: Failed to load OVERRIDE_TOKENIZER '{settings.OVERRIDE_TOKENIZER}'. Error: {e_override}. Attempting other fallbacks.")
                # Fallthrough to model-specific or default logic if override fails
                self._initialize_default_tokenizers() 
        else:
            self._initialize_default_tokenizers()
        
        # logger.info(f"ContextManager initialized with tokenizer: {self.tokenizer.name}") # Covered by specific logs above

    def _initialize_default_tokenizers(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model(settings.LLM_MODEL_NAME)
            logger.info(f"ContextManager: Using tokenizer '{self.tokenizer.name}' based on LLM_MODEL_NAME '{settings.LLM_MODEL_NAME}'.")
        except KeyError:
            logger.warning(f"ContextManager: No specific tiktoken encoding for model '{settings.LLM_MODEL_NAME}'. Falling back.")
            # ... (rest of the existing fallback logic for cl100k_base, then p50k_base) ...
            # Ensure the final RuntimeError is within this method or handled appropriately.
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("ContextManager: Using 'cl100k_base' tokenizer as fallback.")
            except Exception as e_cl100k:
                logger.warning(f"ContextManager: Failed to load 'cl100k_base', falling back to 'p50k_base'. Error: {e_cl100k}")
                try:
                    self.tokenizer = tiktoken.get_encoding("p50k_base")
                    logger.info("ContextManager: Using 'p50k_base' tokenizer as further fallback.")
                except Exception as e_p50k:
                    logger.error(f"ContextManager: Failed to load any tiktoken encoding: {e_p50k}", exc_info=True)
                    raise RuntimeError("Could not initialize tiktoken tokenizer for ContextManager.") from e_p50k

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        return len(self.tokenizer.encode(text))

    async def add_interaction(self, role: str, content: str, episode_uuid: UUID):
        token_count = self._count_tokens(content)
        interaction = {
            "role": role,
            "content": content,
            "uuid": episode_uuid,
            "token_count": token_count
        }
        self.context_window.append(interaction)
        self.current_total_tokens += token_count
        logger.debug(f"Added interaction to context: role={role}, tokens={token_count}, total_tokens={self.current_total_tokens}")
        
        # Asynchronously check and trigger compaction if needed
        # For now, making it synchronous for simplicity in this step.
        # If this were async: asyncio.create_task(self.check_and_trigger_compaction())
        await self.check_and_trigger_compaction()


    def get_oldest_slice_for_compaction(self, num_tokens_to_remove: int) -> List[Dict[str, Any]]:
        compactable_slice: List[Dict[str, Any]] = []
        tokens_gathered = 0
        
        # Iterate from left (oldest)
        while self.context_window and tokens_gathered < num_tokens_to_remove:
            oldest_interaction = self.context_window[0] # Peek
            # Avoid compacting a single very large message if it alone exceeds num_tokens_to_remove,
            # unless it's the only thing in context (edge case).
            if oldest_interaction["token_count"] > num_tokens_to_remove and len(compactable_slice) > 0 :
                break # Don't take an item larger than remaining needed if we already have some items.
            
            interaction_to_compact = self.context_window.popleft()
            compactable_slice.append(interaction_to_compact)
            tokens_gathered += interaction_to_compact["token_count"]
            self.current_total_tokens -= interaction_to_compact["token_count"] # Adjust total tokens
            if self.current_total_tokens < 0: self.current_total_tokens = 0 # Safety
            
            # If a single message is very large and is the first one popped.
            if oldest_interaction["token_count"] > num_tokens_to_remove and len(compactable_slice) == 1:
                break


        logger.info(f"Extracted {len(compactable_slice)} interactions for compaction, totaling {tokens_gathered} tokens.")
        return compactable_slice

    def add_compaction_summary(self, summary_text: str, summary_token_count: int, original_episode_uuids: List[UUID], summary_uuid: Optional[UUID]):
        if not summary_text or summary_token_count == 0:
            logger.warning("Empty summary or zero tokens, not adding to context window.")
            return

        summary_interaction = {
            "role": "system", # Or a special role like "summary"
            "content": f"Summary of prior conversation (covering {len(original_episode_uuids)} messages): {summary_text}",
            "uuid": summary_uuid if summary_uuid else UUID(int=0), # Placeholder if no DB UUID for summary
            "token_count": summary_token_count,
            "is_summary": True,
            "original_episode_uuids": original_episode_uuids
        }
        # Add summary to the beginning of the context window (oldest part)
        self.context_window.appendleft(summary_interaction)
        self.current_total_tokens += summary_token_count
        logger.info(f"Added compaction summary to context: tokens={summary_token_count}, total_tokens={self.current_total_tokens}")


    async def check_and_trigger_compaction(self):
        if self.current_total_tokens > settings.COMPACTION_TRIGGER_TOKENS:
            logger.info(f"Total tokens {self.current_total_tokens} exceeded trigger {settings.COMPACTION_TRIGGER_TOKENS}. Initiating compaction.")
            
            tokens_to_remove = self.current_total_tokens - (settings.COMPACTION_TRIGGER_TOKENS - settings.COMPACTION_OLDEST_TOKENS) # Aim to get below trigger
            if tokens_to_remove < settings.COMPACTION_OLDEST_TOKENS : # Ensure we try to remove a substantial amount
                tokens_to_remove = settings.COMPACTION_OLDEST_TOKENS

            if tokens_to_remove <= 0:
                logger.info("Compaction triggered, but calculated tokens to remove is zero or negative. Skipping.")
                return

            messages_to_compact = self.get_oldest_slice_for_compaction(tokens_to_remove)
            
            if messages_to_compact:
                # In a real scenario, this might be async:
                # summary_text, summary_token_count, summary_uuid = await self.compaction_service.compact_and_store_memory(messages_to_compact)
                
                # Using placeholder synchronously for now
                # For the placeholder, we call it directly. If it were a real async method called
                # from a sync method like this, one would need asyncio.run() or similar,
                # but check_and_trigger_compaction itself would ideally be async.
                # For this step, direct call is fine as placeholder is simple.
                # summary_text, summary_token_count, summary_uuid = self.compaction_service.compact_messages_and_get_summary_text(messages_to_compact)
                summary_text, summary_token_count, summary_uuid = await self.compaction_service.compact_and_store_memory(messages_to_compact)
                # summary_uuid would be None from placeholder's compact_and_store_memory if we called that.
                # Let's align with the async nature for future, but call the sync part of placeholder for now.
                # summary_text, summary_token_count, summary_uuid = asyncio.run(self.compaction_service.compact_and_store_memory(messages_to_compact)) # This would be for a real async service

                if summary_text and summary_token_count > 0:
                    original_uuids = [m.get("uuid") for m in messages_to_compact if m.get("uuid")]
                    self.add_compaction_summary(summary_text, summary_token_count, original_uuids, summary_uuid)
                else:
                    logger.warning("Compaction resulted in no summary text or zero tokens. Original messages removed but not replaced by summary.")
                    # Decide if removed messages should be added back if summary fails. For now, they are kept removed.
            else:
                logger.info("No messages were extracted for compaction, though trigger was met. Possibly due to large single messages.")


    def get_current_context_for_prompt(self, 
                                     system_prompt: Optional[str] = "You are a helpful AI assistant.",
                                     retrieved_memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        
        prompt_messages: List[Dict[str, str]] = []
        available_tokens = settings.CONTEXT_WINDOW_SIZE

        # 1. Add System Prompt
        if system_prompt:
            system_token_count = self._count_tokens(system_prompt)
            if available_tokens >= system_token_count:
                prompt_messages.append({"role": "system", "content": system_prompt})
                available_tokens -= system_token_count
            else:
                logger.warning("Not enough tokens for system prompt.")
                return [{"role": "user", "content": "Error: Context too small for system prompt."}] # Should not happen

        # 2. Add Retrieved Memories (if any, and if space allows)
        # Format memories appropriately. Keep it simple for now.
        if retrieved_memories:
            memories_text_parts = []
            for mem in reversed(retrieved_memories): # Add newest retrieved memories first
                mem_content = mem.get('content', '')
                # Could add speaker, timestamp if desired: e.g., f"Recalled ({mem.get('speaker', 'unknown')} at {mem.get('timestamp')}): {mem_content}"
                mem_str = f"Recalled from memory: {mem_content}"
                mem_tokens = self._count_tokens(mem_str)
                
                if available_tokens >= mem_tokens:
                    memories_text_parts.append(mem_str)
                    available_tokens -= mem_tokens
                else:
                    break # Stop adding memories if no more space
            
            if memories_text_parts:
                # Consolidate memories into a single system-like message or assistant pre-fill.
                # Using a system message here to provide context before current conversation.
                full_memories_text = "\n".join(reversed(memories_text_parts)) # Newest first in this block
                # This could also be a user message like "Here's some relevant context:"
                prompt_messages.append({"role": "system", "content": f"Relevant information from memory:\n{full_memories_text}"})


        # 3. Add current conversation from deque (newest first, up to available_tokens)
        temp_conversation_store: List[Dict[str, str]] = []
        for interaction in reversed(self.context_window):
            interaction_content = interaction.get("content", "")
            # For summaries, the content might already be formatted.
            # For regular interactions, prepend role if not already part of content.
            # Assuming 'content' is the pure text from user/assistant for non-summary.
            # For this example, let's assume 'content' is what LLM expects per role.
            
            # Use interaction's specific token_count if available and accurate
            token_count = interaction.get("token_count", self._count_tokens(interaction_content))
            
            if available_tokens >= token_count:
                temp_conversation_store.append({"role": interaction["role"], "content": interaction_content})
                available_tokens -= token_count
            else:
                # Try to partially include the last message if it's too long
                if available_tokens > 20 : # Arbitrary minimum to include part of a message
                    try:
                        # This is a simplification; precise truncation needs careful handling of token boundaries.
                        truncated_content = self.tokenizer.decode(self.tokenizer.encode(interaction_content)[:available_tokens-5]) + "..."
                        temp_conversation_store.append({"role": interaction["role"], "content": truncated_content})
                        available_tokens = 0 # No more space
                    except Exception:
                        logger.warning(f"Could not truncate content for interaction {interaction.get('uuid')}")
                break # No more space, or couldn't truncate

        prompt_messages.extend(reversed(temp_conversation_store)) # Add to main prompt in correct order

        logger.info(f"Assembled prompt with {len(prompt_messages)} messages, using {settings.CONTEXT_WINDOW_SIZE - available_tokens} tokens.")
        return prompt_messages


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from uuid import uuid4

    # Mock CompactionService for testing ContextManager in isolation
    mock_compaction_service = PlaceholderCompactionService()
    
    manager = ContextManager(compaction_service=mock_compaction_service)
    async def run_add_interaction_test():
        # Test adding interactions
        await manager.add_interaction("user", "Hello, this is my first message.", uuid4())
        await manager.add_interaction("assistant", "Hi there! Nice to meet you.", uuid4())
        await manager.add_interaction("user", "I want to talk about LLMs.", uuid4())
    
    asyncio.run(run_add_interaction_test()) # Encapsulate async calls for test block

    logger.info(f"Current context window items: {len(manager.context_window)}")
    logger.info(f"Current total tokens: {manager.current_total_tokens}")

    # Test getting context for prompt
    prompt = manager.get_current_context_for_prompt(
        system_prompt="You are a friendly conversational AI.",
        retrieved_memories=[
            {"content": "User previously expressed interest in Python programming.", "speaker": "system_memory", "timestamp": "2023-01-01T10:00:00Z"},
            {"content": "Agent helped user debug a script.", "speaker": "system_memory", "timestamp": "2023-01-01T10:05:00Z"}
        ]
    )
    logger.info("\nAssembled Prompt:")
    for msg in prompt:
        logger.info(f"  Role: {msg['role']}, Content: '{msg['content'][:100]}...'")
    
    assert len(prompt) > 0
    assert manager.current_total_tokens > 0

    # Test compaction trigger
    logger.info("\nTesting compaction trigger...")
    # Override settings for test
    settings.COMPACTION_TRIGGER_TOKENS = 50 
    settings.COMPACTION_OLDEST_TOKENS = 20 # Compact oldest 20 tokens when going over 50

    # Add more interactions to exceed trigger tokens
    async def run_compaction_trigger_test():
        for i in range(10):
            await manager.add_interaction("user", f"This is user message number {i+4}. It adds some more tokens to our growing conversation history.", uuid4())
            await manager.add_interaction("assistant", f"This is assistant response {i+4}. I am also adding tokens.", uuid4())
            if manager.current_total_tokens < settings.COMPACTION_TRIGGER_TOKENS and len(manager.context_window) > 5 : # check if compaction happened by summary being added
                # if a summary was added, it means compaction occurred
                # This check is a bit indirect for a placeholder.
                pass # Pass for now, actual check is below
    
    asyncio.run(run_compaction_trigger_test()) # Encapsulate async calls

    logger.info(f"After many interactions: Context items: {len(manager.context_window)}, Total tokens: {manager.current_total_tokens}")
            # A real check would be if compaction_service.compact_and_store_memory was called.
            pass


    logger.info(f"After many interactions: Context items: {len(manager.context_window)}, Total tokens: {manager.current_total_tokens}")
    # Check if a summary message exists (PlaceholderCompactionService adds one)
    has_summary = any(item.get("is_summary", False) for item in manager.context_window)
    logger.info(f"Context contains a summary message: {has_summary}")
    # With PlaceholderCompactionService, if current_total_tokens exceeded COMPACTION_TRIGGER_TOKENS,
    # and then messages were popped and a summary added, this should be true.
    # The exact token count will vary based on placeholder summary logic.
    assert manager.current_total_tokens < (settings.COMPACTION_TRIGGER_TOKENS + 50) # Should be roughly around trigger + summary_tokens - compacted_tokens

    logger.info("ContextManager tests completed.")
