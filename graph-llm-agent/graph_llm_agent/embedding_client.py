from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
import logging

from graph_llm_agent.config import settings

logger = logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name if model_name else settings.EMBEDDING_MODEL_NAME
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available(): # For Apple Silicon
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Initializing EmbeddingClient with model: {self.model_name} on device: {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # If EMBEDDING_DIMENSIONS is not set in config, try to get it from the model
            if settings.EMBEDDING_DIMENSIONS is None:
                dim = self.model.get_sentence_embedding_dimension()
                if dim is not None:
                    settings.EMBEDDING_DIMENSIONS = dim
                    logger.info(f"Auto-detected embedding dimensions: {settings.EMBEDDING_DIMENSIONS}")
                else:
                    # Fallback if model doesn't provide it directly (should be rare for SBERT)
                    # For "sentence-transformers/all-MiniLM-L6-v2" it's 384
                    # For "Qwen/Qwen-embed-large" it's 1024 (this is an example, actual Qwen model name might differ)
                    if "all-MiniLM-L6-v2" in self.model_name:
                         settings.EMBEDDING_DIMENSIONS = 384
                    elif "Qwen-embed-large" in self.model_name: # Placeholder, adjust if using Qwen embedder
                         settings.EMBEDDING_DIMENSIONS = 1024
                    else: # Default to a common value if undetectable and not in known list
                         settings.EMBEDDING_DIMENSIONS = 768 
                    logger.warning(
                        f"Could not auto-detect embedding dimensions. Falling back to {settings.EMBEDDING_DIMENSIONS}. "
                        f"Please set EMBEDDING_DIMENSIONS in config if this is incorrect."
                    )
            elif self.model.get_sentence_embedding_dimension() != settings.EMBEDDING_DIMENSIONS:
                logger.warning(
                    f"Configured EMBEDDING_DIMENSIONS ({settings.EMBEDDING_DIMENSIONS}) "
                    f"does not match model's dimension ({self.model.get_sentence_embedding_dimension()}). "
                    f"Using model's dimension."
                )
                settings.EMBEDDING_DIMENSIONS = self.model.get_sentence_embedding_dimension()


            logger.info(f"Embedding model {self.model_name} loaded successfully on {self.device}.")
            logger.info(f"Embedding dimensions: {self.get_embedding_dimensionality()}")

        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
            # Consider whether to raise the error or handle it by disabling embedding capabilities
            raise RuntimeError(f"Failed to load SentenceTransformer model '{self.model_name}'") from e

    def get_embedding(self, text: str) -> List[float]:
        if settings.TEST_MODE:
            logger.info("EmbeddingClient is in TEST_MODE. Returning dummy vector.")
            dim = self.get_embedding_dimensionality() or 384 
            return [0.01] * dim
            
        if not text or not isinstance(text, str):
            logger.warning("Received empty or invalid text for embedding, returning empty list.")
            return []
        try:
            embedding = self.model.encode(text, convert_to_tensor=False) # Returns numpy array
            return embedding.tolist() # Convert to list of floats
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:100]}...': {e}", exc_info=True)
            return [] # Return empty list on error

    def get_embedding_dimensionality(self) -> Optional[int]:
        if settings.EMBEDDING_DIMENSIONS:
            return settings.EMBEDDING_DIMENSIONS
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return None

# Global instance (optional, can be managed by a dependency injection framework later)
# embedding_client = EmbeddingClient()

if __name__ == "__main__":
    # Configure basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test with default model from config
    print(f"Testing with model: {settings.EMBEDDING_MODEL_NAME}")
    try:
        client = EmbeddingClient()
        example_text = "This is a test sentence for the embedding client."
        embedding = client.get_embedding(example_text)
        
        if embedding:
            print(f"Successfully generated embedding for: '{example_text}'")
            print(f"Embedding (first 10 dims): {embedding[:10]}")
            print(f"Embedding dimensions: {client.get_embedding_dimensionality()}")
            assert len(embedding) == client.get_embedding_dimensionality(), "Embedding length mismatch!"
        else:
            print(f"Failed to generate embedding for: '{example_text}'")

        # Test with a potentially different model (if you have one downloaded)
        # You might need to change this to a model you have access to or expect to work
        # For example, if Qwen/Qwen-embed-large is a valid SentenceTransformer model:
        # print("\nTesting with specific model: Qwen/Qwen-embed-large (example)")
        # try:
        #     qwen_client = EmbeddingClient(model_name="Qwen/Qwen-embed-large") # Fictitious name for testing
        #     qwen_embedding = qwen_client.get_embedding("Another test.")
        #     if qwen_embedding:
        #         print(f"Qwen embedding (first 10 dims): {qwen_embedding[:10]}")
        #         print(f"Qwen embedding dimensions: {qwen_client.get_embedding_dimensionality()}")
        # except Exception as e:
        #     print(f"Could not test with Qwen/Qwen-embed-large: {e}")


        # Test empty string
        print("\nTesting with empty string:")
        empty_embedding = client.get_embedding("")
        if not empty_embedding:
            print("Correctly returned empty list for empty string.")
        else:
            print(f"Error: Expected empty list for empty string, got {empty_embedding}")

    except RuntimeError as e:
        print(f"Error during EmbeddingClient test: {e}")
    except ImportError:
        print("SentenceTransformers library not found. Please install it: pip install sentence-transformers")
