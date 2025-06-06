import anthropic
import logging
from typing import Tuple

from config import ANTHROPIC_API_KEY, VALID_ANTHROPIC_MODELS


logger = logging.getLogger(__name__)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def validate_anthropic_model(model_name: str) -> str:
    """Validates and returns the correct model name format for Anthropic API."""
    if model_name in VALID_ANTHROPIC_MODELS:
        return model_name
    
    # Handle "latest" versions by mapping to the most recent version
    if model_name == "claude-3-5-sonnet-latest":
        return "claude-3-5-sonnet-20241022"  # Use the latest version

    # Other mappings as needed
    model_mappings = {
        "claude-3-7-sonnet-20250219": "claude-3-5-sonnet-20241022",  # Fallback if it is a typo
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    if model_name in model_mappings:
        return model_mappings(model_name)
    
    # If not match found, return a default model
    logger.warning(f"Model: '{model_name}' not found. Using default model.")
    return "claude-3-5-sonnet-20241022"


async def generate_anthropic_response(prompt: str,  model: str, temperature: float) -> Tuple[str,str]:
    """
    Generate a response using Anthropic API
    
    Args:
        prompt: The user's input prompt
        model: The Claude model to use
        temperature: Creativity parameter: (0 - 1)
        
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    # Validate the model name
    validated_model = validate_anthropic_model(model)
    logger.info(f"Using Anthropic model: {validated_model} (requested: {model})")

    # Define the system message
    system_message = "You are a helpful assistant. When providing code snippets, use triple backticks (```) to format the code blocks with proper indentation and syntax highlighting."
     
    # Create the message payload
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        # Generate a response using the Anthropic API
        response = anthropic_client.messages.create(
            model=validated_model,
            system=system_message,
            messages=messages,
            max_tokens=1000,
            temperature=temperature
        )

        content = response.content[0].text
        return content, validated_model
    except Exception as e:
        logger.error(f"Error with Anthropic API call: {e}")
        raise
