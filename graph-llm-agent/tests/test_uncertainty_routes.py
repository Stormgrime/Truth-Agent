import pytest
import asyncio
from typing import Tuple, Optional

from graph_llm_agent.uncertainty import UncertaintyArbiter
from graph_llm_agent.llm_client import LLMClient # For mocking
from graph_llm_agent.config import settings # To potentially override thresholds for testing

# --- Mock LLMClient ---
class MockLLMClientUncertainty(LLMClient):
    def __init__(self):
        # Bypassing super().__init__() as it might try to load models
        self.mode = "mock" 
        self.responses: dict[str, Tuple[str, float]] = {}

    def set_response_for_prompt_key(self, prompt_key: str, answer: str, confidence: float):
        self.responses[prompt_key] = (answer, confidence)

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> Tuple[str, float]:
        if "PROMPT_A_KEY" in prompt: 
            return self.responses.get("A", ("Error: Mock for Pass A not set", 0.0))
        elif "PROMPT_B_KEY" in prompt: 
            return self.responses.get("B", ("Error: Mock for Pass B not set", 0.0))
        # Fallback for any other prompt not matching A or B keys
        return "Error: Unknown prompt type for mock LLMClientUncertainty", 0.0

@pytest.fixture
def mock_llm_client_uncertainty_fixture() -> MockLLMClientUncertainty: # Renamed fixture
    return MockLLMClientUncertainty()

@pytest.fixture
def uncertainty_arbiter_fixture(mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty) -> UncertaintyArbiter:
    return UncertaintyArbiter(llm_client=mock_llm_client_uncertainty_fixture)

PROMPT_A_TEXT = "PROMPT_A_KEY: This is a question without memory."
PROMPT_B_TEXT = "PROMPT_B_KEY: This is a question with memory."


@pytest.mark.asyncio
async def test_route_answered_confidently_no_mem(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "Answer A is correct.", 0.9)
    _final_ans, policy, c_a, c_b, _r_a, _r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, None)
    assert policy == "answered_confidently_no_mem"
    assert c_a == 0.9
    assert c_b == 0.0

@pytest.mark.asyncio
async def test_route_clarify_or_idk_no_mem(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "Not sure about A.", 0.2)
    _final_ans, policy, _c_a, _c_b, _r_a, _r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, None)
    assert policy == "clarify_or_idk_no_mem"

@pytest.mark.asyncio
async def test_route_answered_confidently_with_mem(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "Answer A might be okay.", 0.6)
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("B", "Answer B from memory is definitely correct.", 0.95)
    _final_ans, policy, _c_a, _c_b, _r_a, _r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, PROMPT_B_TEXT)
    assert policy == "answered_confidently_with_mem"

@pytest.mark.asyncio
async def test_route_answered_confidently_no_mem_preferred(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "Answer A is very solid.", 0.92)
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("B", "Memory suggests this, but it's weak.", 0.5)
    _final_ans, policy, _c_a, _c_b, _r_a, _r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, PROMPT_B_TEXT)
    assert policy == "answered_confidently_no_mem_preferred"

@pytest.mark.asyncio
async def test_route_clarify_or_idk_both_low(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "A is very uncertain.", 0.25)
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("B", "B is also very uncertain.", 0.3)
    _final_ans, policy, _c_a, _c_b, _r_a, _r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, PROMPT_B_TEXT)
    assert policy == "clarify_or_idk_both_low"

@pytest.mark.asyncio
async def test_route_pass_b_llm_error(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "Answer A is good.", 0.85)
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("B", "Error: LLM for Pass B failed.", 0.0) 
    _final_ans, policy, c_a, c_b, r_a, r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, PROMPT_B_TEXT)
    assert policy == "answered_confidently_no_mem" 
    assert c_a == 0.85
    assert c_b == 0.0 
    assert r_b == ""  

@pytest.mark.asyncio
async def test_route_both_highly_confident(uncertainty_arbiter_fixture: UncertaintyArbiter, mock_llm_client_uncertainty_fixture: MockLLMClientUncertainty):
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("A", "Answer A very confident.", 0.9)
    mock_llm_client_uncertainty_fixture.set_response_for_prompt_key("B", "Answer B also very confident.", 0.92)
    final_ans, policy, _c_a, _c_b, _r_a, _r_b = await uncertainty_arbiter_fixture.arbitrate(PROMPT_A_TEXT, PROMPT_B_TEXT)
    assert policy == "answered_confidently_with_mem_both_high" 
    assert final_ans == "Answer B also very confident."

```
