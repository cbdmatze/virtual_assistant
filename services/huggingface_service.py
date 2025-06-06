
from transformers import pipeline
import logging
from typing import Tuple, Optional

from config import HUGGINGFACE_MODEL


logger = logging.getLogger(__name__)

# Global variable for the text generation pipeline
generator = None


def init_huggingface():
    """Initialize the HuggingFace text generation pipeline"""
    global generator

    try:
        logger.info(f"Initializing HuggingFace pipeline with model: {HUGGINGFACE_MODEL}...")
        generator = pipeline("text-generation", model=HUGGINGFACE_MODEL)
        logger.info("HuggingFace pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing HuggingFace pipeline: {e}")
        generator = None


def check_huggingface_status() -> bool:
    """Check if HuggingFace model is available"""
    return generator is not None


def generate_huggingface_response(prompt: str, max_length: int = 1000) -> Tuple[str,str]:
    """
    Generate a response using the HuggingFace pipeline
    
    Args:
        prompt: The user's input prompt
        max_length: The maximum length of the generated response
    
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    global generator
    if generator is None:
        raise Exception("HuggingFace pipeline is not available.", HUGGINGFACE_MODEL)
    
    try:
        logger.info(f"Using HuggingFace model: {HUGGINGFACE_MODEL}")
        outputs = generator(prompt, max_length=max_length, num_return_sequences=1)
        return outputs[0]["generated_text"], HUGGINGFACE_MODEL
    except Exception as e:
        logger.error(f"Error generating text with HuggingFace model: {e}")
        return f"Error geenrating text with HuggingFace model: {str(e)}", HUGGINGFACE_MODEL
