import openai
import logging
from typing import Tuple, Dict, Any

from config import OPENAI_API_KEY


logger = logging.getLogger(__name__)

# Initialize OpenaI API
openai.api_key = OPENAI_API_KEY

def check_openai_status() -> Tuple[str, bool]:
    """Check OpenAI API status and version"""
    try:
        if hasattr(openai, 'OpenAI'):
            return "v1.x", True
        else:
            return "v0.28.x", True
    except Exception as e:
        logger.error(f"Error checking OpenAI API status: {e}")
        return "unknown", False
   

async def generate_openai_response(prompt: str, model: str, temperature: float) -> Tuple[str,str]:
    """
    Generate a response using OpenAI
    
    Args:
        prompt: The user's input prompt
        model: The OpenAI model to use
        temperature: Creativity parameter: (-1 - +2)
        
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    logger.info(f"Using OpenAI model: {model}")

    try:
        context_prompt = (
            "You are a helpful assistant. When providing code snippets, start a new line ensure "
            "correct indentation and syntax highlighting in python. "
            "Use triple backticks (```) to format the code blocks.\n\n"
            f"User: {prompt}\n"
            "Assistant:"
        )
        
        content = get_openai_completion(
            model, 
            [{"role": "user", "content": context_prompt}],
            1000,
            temperature
        )
        return content, model
    except Exception as e:
        logger.error(f"Error with OpenAI API call: {e}")
        raise


def get_openai_completion(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    """
    Compatibility function that works with both v1.x and v0.28.x OpenAI API versions
    """
    try:
        if hasattr(openai, 'OpenAI'): # v1.0.0x
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        else: # v0.28.x
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message['content']
    except Exception as e:
        logger.error(f"Error with OpenAI API call: {e}")
        raise

