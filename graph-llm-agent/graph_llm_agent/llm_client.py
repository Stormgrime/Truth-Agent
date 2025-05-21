import logging
import re
from typing import Tuple, Optional, Dict, Any

import requests # For synchronous HTTP requests to vLLM
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from graph_llm_agent.config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self,
                 model_name: Optional[str] = None,
                 llm_api_base_url: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 device: Optional[str] = None):

        self.model_name = model_name if model_name else settings.LLM_MODEL_NAME
        self.api_base_url = llm_api_base_url if llm_api_base_url else settings.LLM_API_BASE_URL
        self.api_key = llm_api_key if llm_api_key else settings.LLM_API_KEY
        
        self.mode = "http_client" if self.api_base_url else "hf_pipeline"

        if self.mode == "hf_pipeline":
            if device:
                self.device_map = device
            elif torch.cuda.is_available():
                self.device_map = "auto" # Let transformers handle multi-GPU or single GPU
            elif torch.backends.mps.is_available() and not torch.cuda.is_available():
                 self.device_map = "mps"
            else:
                self.device_map = "cpu" # Fallback to CPU

            logger.info(f"Initializing LLMClient in 'hf_pipeline' mode with model: {self.model_name} on device_map: '{self.device_map}'")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device_map,
                    trust_remote_code=True
                )
                logger.info(f"Hugging Face model and tokenizer for {self.model_name} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Hugging Face model/tokenizer for model '{self.model_name}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to load Hugging Face model/tokenizer for model '{self.model_name}'") from e
        else:
            logger.info(f"Initializing LLMClient in 'http_client' mode for endpoint: {self.api_base_url}")
            if not self.api_base_url.endswith("/v1"):
                logger.warning(f"LLM API Base URL '{self.api_base_url}' does not end with '/v1'. Assuming it's the base and will append '/chat/completions'.")
                self.chat_completions_url = self.api_base_url.rstrip('/') + "/v1/chat/completions"
            else: # Ends with /v1
                self.chat_completions_url = self.api_base_url.rstrip('/') + "/chat/completions"


    def _parse_response_with_confidence(self, text_content: str) -> Tuple[str, float]:
        confidence_match = re.search(r"### confidence \(0-100\):\s*(\d+)", text_content, re.IGNORECASE)
        if confidence_match:
            confidence_score = float(confidence_match.group(1)) / 100.0
            text_before_confidence = text_content[:confidence_match.start()].strip()
            return text_before_confidence, confidence_score
        else:
            logger.warning("Confidence score not found in LLM response. Using default confidence 1.0.")
            return text_content.strip(), 1.0


    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> Tuple[str, float]:
        if settings.TEST_MODE:
            logger.info("LLMClient is in TEST_MODE. Returning canned response.")
            return "This is a canned test mode response from LLMClient.", 0.99
        
        logger.debug(f"Generating response for prompt (first 100 chars): {prompt[:100]}...")
        
        if self.mode == "hf_pipeline":
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                # Ensure decoding starts after the prompt tokens
                prompt_input_ids_length = inputs.input_ids.shape[1]
                generated_ids = outputs[0][prompt_input_ids_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                return self._parse_response_with_confidence(generated_text)

            except Exception as e:
                logger.error(f"Error during Hugging Face model generation: {e}", exc_info=True)
                return "Error: Could not generate response.", 0.0
        
        elif self.mode == "http_client":
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            try:
                response = requests.post(self.chat_completions_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                
                data = response.json()
                generated_text = data['choices'][0]['message']['content'].strip()
                return self._parse_response_with_confidence(generated_text)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP request failed for LLM generation: {e}", exc_info=True)
                return f"Error: API request failed - {e}", 0.0
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse LLM API response: {e}. Response: {response.text}", exc_info=True)
                return "Error: Invalid API response.", 0.0
        
        return "Error: LLMClient not configured properly.", 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- Testing Hugging Face Pipeline Mode ---")
    test_hf_model = settings.LLM_MODEL_NAME
    
    if not settings.LLM_API_BASE_URL:
        try:
            print(f"Attempting to load HF model: {test_hf_model}")
            force_cpu = not (torch.cuda.is_available() or torch.backends.mps.is_available())
            client_hf = LLMClient(model_name=test_hf_model, device="cpu" if force_cpu else None)
            
            prompt_hf_test = f"Write a sentence about a cat. Then, on a new line, write '### confidence (0-100): 75'"
            response_hf, confidence_hf = client_hf.generate_response(prompt_hf_test, temperature=0.7, max_tokens=60)
            print(f"HF Response: '{response_hf}'")
            print(f"HF Confidence: {confidence_hf}")
            
            if "Error:" not in response_hf:
                assert "### confidence (0-100): 75" not in response_hf
                assert confidence_hf == 0.75
            else:
                print("HF test encountered an error, skipping assertions.")

        except RuntimeError as e:
            print(f"HF Pipeline test failed for {test_hf_model}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during HF pipeline test: {e}", exc_info=True)
    else:
        print("Skipping Hugging Face Pipeline Mode test as LLM_API_BASE_URL is set.")

    print("\n--- Testing HTTP Client Mode (vLLM OpenAI Compatible) ---")
    if settings.LLM_API_BASE_URL:
        try:
            client_http = LLMClient(model_name="served-model-name", llm_api_base_url=settings.LLM_API_BASE_URL)
            prompt_http = "What is the capital of France? On a new line, write '### confidence (0-100): 90'"
            response_http, confidence_http = client_http.generate_response(prompt_http, temperature=0.1, max_tokens=50)
            
            print(f"HTTP Response: '{response_http}'")
            print(f"HTTP Confidence: {confidence_http}")

            if "Error:" not in response_http:
                 assert "### confidence (0-100): 90" not in response_http
                 assert confidence_http == 0.90
            else:
                print("HTTP Client test encountered an error, skipping assertions.")
        except Exception as e:
            print(f"HTTP Client test failed: {e}. Ensure vLLM (or compatible) server is running at {settings.LLM_API_BASE_URL}.")
    else:
        print("Skipping HTTP Client Mode test as LLM_API_BASE_URL is not set in config.")
