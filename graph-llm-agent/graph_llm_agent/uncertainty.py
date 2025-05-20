import logging
from typing import Tuple, Optional, List, Dict, Any
import math # For isnan checks if needed

from graph_llm_agent.config import settings
from graph_llm_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class UncertaintyArbiter:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.CONF_THRESHOLD_HIGH = getattr(settings, 'UNCERTAINTY_CONF_THRESHOLD_HIGH', 0.75)
        self.CONF_THRESHOLD_LOW = getattr(settings, 'UNCERTAINTY_CONF_THRESHOLD_LOW', 0.4)
        self.CONF_DIFF_MARGIN = getattr(settings, 'UNCERTAINTY_CONF_DIFF_MARGIN', 0.15)

    async def arbitrate(
        self,
        prompt_A: str, 
        prompt_B: Optional[str] = None
    ) -> Tuple[str, str, float, float, str, str]:
        logger.info("Arbitrating response based on uncertainty...")

        answer_A_raw, confidence_A = self.llm_client.generate_response(prompt_A)
        if "Error:" in answer_A_raw:
            logger.error(f"LLMClient returned an error for Pass A: {answer_A_raw}")
            return "I'm currently unable to generate a response. Please try again later.", "error_pass_a", 0.0, 0.0, answer_A_raw, ""
        
        logger.info(f"Pass A: Answer='{answer_A_raw[:50]}...', Confidence={confidence_A:.2f}")

        answer_B_raw = ""
        confidence_B = 0.0

        if prompt_B:
            logger.debug(f"Pass B prompt (first 100 chars): {prompt_B[:100]}...")
            answer_B_raw, confidence_B = self.llm_client.generate_response(prompt_B)
            if "Error:" in answer_B_raw:
                logger.warning(f"LLMClient returned an error for Pass B: {answer_B_raw}. Proceeding with Pass A result only.")
                prompt_B = None 
                answer_B_raw = "" 
                confidence_B = 0.0
            else:
                logger.info(f"Pass B: Answer='{answer_B_raw[:50]}...', Confidence={confidence_B:.2f}")
        else:
            logger.info("No prompt_B provided. Skipping Pass B.")

        final_answer_text: str
        policy_taken: str

        conf_A_norm = confidence_A if not math.isnan(confidence_A) else 0.0
        conf_B_norm = confidence_B if not math.isnan(confidence_B) else 0.0

        if not prompt_B or not answer_B_raw or confidence_B == 0.0: # Only Pass A was successful or run
            if conf_A_norm >= self.CONF_THRESHOLD_HIGH:
                policy_taken = "answered_confidently_no_mem"
                final_answer_text = answer_A_raw
            elif conf_A_norm >= self.CONF_THRESHOLD_LOW:
                policy_taken = "answered_cautiously_no_mem"
                final_answer_text = f"I think the answer might be: {answer_A_raw} (I'm not entirely sure based on current information)."
            else:
                policy_taken = "clarify_or_idk_no_mem"
                final_answer_text = "I'm not sure how to respond to that. Could you please provide more details or rephrase your question?"
        else: # Both Pass A and Pass B ran successfully
            if conf_B_norm >= self.CONF_THRESHOLD_HIGH and (conf_B_norm > conf_A_norm + self.CONF_DIFF_MARGIN):
                policy_taken = "answered_confidently_with_mem"
                final_answer_text = answer_B_raw
            elif conf_A_norm >= self.CONF_THRESHOLD_HIGH and (conf_A_norm > conf_B_norm + self.CONF_DIFF_MARGIN):
                policy_taken = "answered_confidently_no_mem_preferred"
                final_answer_text = answer_A_raw
            elif conf_A_norm >= self.CONF_THRESHOLD_HIGH and conf_B_norm >= self.CONF_THRESHOLD_HIGH:
                policy_taken = "answered_confidently_with_mem_both_high"
                final_answer_text = answer_B_raw 
            elif conf_B_norm >= self.CONF_THRESHOLD_LOW and conf_B_norm > conf_A_norm:
                policy_taken = "answered_cautiously_with_mem"
                final_answer_text = f"Based on relevant memories, I believe: {answer_B_raw}"
            elif conf_A_norm >= self.CONF_THRESHOLD_LOW and conf_A_norm > conf_B_norm:
                policy_taken = "answered_cautiously_no_mem_preferred"
                final_answer_text = f"My understanding is: {answer_A_raw}"
            elif conf_A_norm < self.CONF_THRESHOLD_LOW and conf_B_norm < self.CONF_THRESHOLD_LOW:
                policy_taken = "clarify_or_idk_both_low"
                final_answer_text = "I'm finding it difficult to give a confident answer, even with retrieved memories. Could you clarify or provide more context?"
            elif conf_B_norm > conf_A_norm:
                policy_taken = "answered_cautiously_with_mem_mixed"
                final_answer_text = f"Considering available information, it seems: {answer_B_raw}"
            elif conf_A_norm > conf_B_norm: # Catches A > B when B is not zero
                policy_taken = "answered_cautiously_no_mem_mixed"
                final_answer_text = f"My current thought is: {answer_A_raw}"
            else: 
                policy_taken = "answered_cautiously_no_mem_fallback" # Equal confidences, not high
                final_answer_text = f"This is what I found: {answer_A_raw}"
                
        logger.info(f"Arbitration decision: Policy='{policy_taken}', Final Answer='{final_answer_text[:50]}...'")
        return final_answer_text, policy_taken, confidence_A, confidence_B, answer_A_raw, answer_B_raw

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For direct execution, ensure LLMClient can be imported or is defined locally for the mock
    from graph_llm_agent.llm_client import LLMClient 

    class MockLLMClient(LLMClient):
        def __init__(self):
            self.mode = "mock"
            self.responses = {}

        def set_response(self, prompt_key: str, answer: str, confidence: float):
            self.responses[prompt_key] = (answer, confidence)

        def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> Tuple[str, float]:
            if "PROMPT_A_KEY" in prompt:
                return self.responses.get("A", ("Error: Mock for A not set", 0.0))
            elif "PROMPT_B_KEY" in prompt:
                return self.responses.get("B", ("Error: Mock for B not set", 0.0))
            return "Error: Unknown prompt for mock", 0.0

    mock_llm = MockLLMClient()
    arbiter = UncertaintyArbiter(llm_client=mock_llm)

    async def run_test_scenario(scenario_name: str, prompt_a_text: str, prompt_b_text: Optional[str], 
                                resp_a: Tuple[str, float], resp_b: Optional[Tuple[str, float]]):
        logger.info(f"--- Scenario: {scenario_name} ---")
        mock_llm.set_response("A", resp_a[0], resp_a[1])
        if resp_b and prompt_b_text: # Ensure prompt_b_text is not None for setting response B
            mock_llm.set_response("B", resp_b[0], resp_b[1])
        
        # Pass prompt_b_text to arbitrate
        final_ans, pol, c_a, c_b, r_a, r_b = await arbiter.arbitrate(prompt_a_text, prompt_b_text)
        logger.info(f"  Policy: {pol}")
        logger.info(f"  Conf A: {c_a:.2f}, Conf B: {c_b:.2f}")
        logger.info(f"  Raw A: '{r_a[:30]}...'")
        if prompt_b_text: logger.info(f"  Raw B: '{r_b[:30]}...'")
        logger.info(f"  Final Answer: '{final_ans[:60]}...'")
        logger.info("--- End Scenario ---")
        return pol

    async def main():
        s1 = await run_test_scenario("Confident A, No B", "PROMPT_A_KEY: What is 2+2?", None, ("4", 0.9), None)
        assert s1 == "answered_confidently_no_mem"

        s2 = await run_test_scenario("Low Conf A, No B", "PROMPT_A_KEY: Explain quantum physics simply.", None, ("It's about small stuff.", 0.3), None)
        assert s2 == "clarify_or_idk_no_mem"
        
        s3 = await run_test_scenario("Confident B > A", "PROMPT_A_KEY: Capital of France?", "PROMPT_B_KEY: Capital of France with memory hint?", ("Lyon", 0.6), ("Paris is the capital.", 0.95))
        assert s3 == "answered_confidently_with_mem"

        s4 = await run_test_scenario("Confident A > B", "PROMPT_A_KEY: Who painted Mona Lisa?", "PROMPT_B_KEY: Who painted Mona Lisa? (Memory: Van Gogh)", ("Leonardo da Vinci", 0.9), ("Van Gogh perhaps?", 0.5))
        assert s4 == "answered_confidently_no_mem_preferred"

        s5 = await run_test_scenario("Both Low Conf", "PROMPT_A_KEY: Future of AI?", "PROMPT_B_KEY: Future of AI with generic memory?", ("It's evolving.", 0.35), ("It will be significant.", 0.4))
        assert s5 == "clarify_or_idk_both_low" # Original logic for both low.
        
        # Test for Pass B failure specifically
        logger.info(f"--- Scenario: Pass B Fails (Manual Setup) ---")
        mock_llm_b_fails = MockLLMClient()
        arbiter_b_fails = UncertaintyArbiter(llm_client=mock_llm_b_fails)
        mock_llm_b_fails.set_response("A", "Answer from Pass A", 0.8)
        # Simulate LLMClient returning an error string for Pass B
        mock_llm_b_fails.set_response("B", "Error: LLM B exploded", 0.0) 
        final_ans, pol, c_a, c_b, r_a, r_b = await arbiter_b_fails.arbitrate("PROMPT_A_KEY: Test query", "PROMPT_B_KEY: Test query with mems")
        logger.info(f"  Policy: {pol}, Final Answer: '{final_ans[:60]}...'")
        assert pol == "answered_confidently_no_mem"
        assert c_b == 0.0
        assert r_b == "" # Raw answer B should be empty string if it effectively failed

    import asyncio
    asyncio.run(main())
