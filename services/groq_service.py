import groq
import logging
from typing import Tuple

from config import GROQ_API_KEY, VALID_GROQ_MODELS


logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = groq.Groq(api_key=GROQ_API_KEY)


def validate_groq_model(model_name: str) -> str:
    """Validates and returns the correct model name format for Groq API."""
    if model_name in VALID_GROQ_MODELS:
        return model_name
    
    # Default model if not found
    logger.warning(f"Groq model: '{model_name}' not found. Using default model.")
    return "llama-3.2-90b-vision-preview"


def generate_groq_response(prompt: str, model: str, temperature: float) -> Tuple[str,str]:
    """
    Generate a response using Groq API
    
    Args:
        prompt: The user's input prompt
        model: The Groq model to use
        temperature: Creativity parameter: (-1 - +2)
        
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    # Validate the model name
    validated_model = validate_groq_model(model)
    logger.info(f"Using Groq model: {validated_model} (requested: {model})")

    # System message to format code snippets properly
    system_message = "You are a helpful assistant. When providing code snippets, use triple backticks (```) to format the code blocks with proper indentation and syntax highlighting."

    try:
        # Generate a response using the Groq API
        response = groq_client.chat.completions.create(
            model=validated_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )

        content = response.choices[0].message.content
        return content, validated_model
    except Exception as e:
        logger.error(f"Error with Groq API call: {e}")
        raise
