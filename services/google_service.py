import google.generativeai as genai
import logging
from typing import Tuple

from config import GOOGLE_API_KEY, VALID_GOOGLE_MODELS


logger = logging.getLogger(__name__)

# Initialize Google API
genai.configure(api_key=GOOGLE_API_KEY)


def validate_google_model(model_name: str) -> str:
    """Validates and returns the correct model name format for Google API."""
    if model_name in VALID_GOOGLE_MODELS:
        return model_name
    
    # Default model if not found
    logger.warning(f"Google model '{model_name}' not found. Using default model.")
    return "gemini-1.5-pro"


async def generate_google_response(prompt: str, model: str, temperature: float) -> Tuple[str,str]:
    """
    Generate a response using Google's Gemini models
    
    Args:
        prompt: The user's input prompt
        model: The Gemini model to use
        temperature: Creativity parameter: (-1- +2)
        
        Returns:
        Tuple[str,str]: The generated response and the model used
        """
    # Validate the model name
    validated_model = validate_google_model(model)
    logger.info(f"Using Google model: {validated_model} (requested: {model})")

    try:
        # Create a GenerativeModel instance
        model = genai.GenerativeModel(validated_model)

        # Generate content with the given temperature
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
            )
        )

        content = response.text
        return content, validated_model
    except Exception as e:
        logger.error(f"Error with Google API call: {e}")
        raise
