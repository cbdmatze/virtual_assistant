import google.generativeai as genai
import logging
import re
from typing import Tuple, Dict, Any, Optional

from config import GOOGLE_API_KEY, VALID_GOOGLE_MODELS

logger = logging.getLogger(__name__)

# Initialize Google API
genai.configure(api_key=GOOGLE_API_KEY)

# Input threshold for switching to direct mode (characters)
LARGE_INPUT_THRESHOLD = 10000

# Comprehensive system instruction for detailed responses
COMPREHENSIVE_SYSTEM_MESSAGE = """You are an expert AI assistant that provides detailed, comprehensive answers.
Your responses should be well-structured with examples, technical details, and multiple perspectives."""

# Short system instruction for large inputs
LARGE_INPUT_SYSTEM_MESSAGE = """You are a helpful assistant. Format code with triple backticks."""

# Standard system message for regular responses
STANDARD_SYSTEM_MESSAGE = "You are a helpful assistant. Format code with triple backticks."

def validate_google_model(model_name: str) -> str:
    """Validates and returns the correct model name format for Google API."""
    if model_name in VALID_GOOGLE_MODELS:
        return model_name
    
    # Default model if not found
    logger.warning(f"Google model '{model_name}' not found. Using default model.")
    return "gemini-1.5-pro"

def estimate_token_count(text: str) -> int:
    """Roughly estimate token count (4 chars ≈ 1 token)"""
    return len(text) // 4

async def generate_google_response(
    prompt: str, 
    model: str = "gemini-1.5-pro", 
    temperature: float = 0.7,
    comprehensive: bool = True,
    max_output_tokens: int = 8192
) -> Tuple[str, str]:
    """
    Generate a response using Google's Gemini models with option for comprehensive responses
    
    Args:
        prompt: The user's input prompt
        model: The Gemini model to use
        temperature: Creativity parameter (0.0-1.0)
        comprehensive: Whether to generate a comprehensive response
        max_output_tokens: Maximum number of tokens in the response
        
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    # Validate the model name
    validated_model = validate_google_model(model)
    
    # Check input size to determine processing mode
    input_size = len(prompt)
    is_large_input = input_size > LARGE_INPUT_THRESHOLD
    
    # Adjust mode based on input size
    use_comprehensive = comprehensive and not is_large_input
    
    logger.info(f"Using Google model: {validated_model} (requested: {model}) with comprehensive mode: {use_comprehensive} (input size: {input_size} chars)")

    try:
        if use_comprehensive:
            # For comprehensive mode with smaller inputs, use the thinking process
            response = await generate_comprehensive_google_response(prompt, validated_model, temperature, max_output_tokens)
            return response, f"{validated_model}-Comprehensive"
        else:
            # For large inputs or standard mode, use direct approach with appropriate system message
            system_msg = LARGE_INPUT_SYSTEM_MESSAGE if is_large_input else STANDARD_SYSTEM_MESSAGE
            
            # For very large inputs, adjust max_output_tokens to match input size
            if is_large_input:
                estimated_tokens = estimate_token_count(prompt)
                max_output_tokens = max(max_output_tokens, min(estimated_tokens * 2, 30000))
                logger.info(f"Adjusted max_output_tokens to {max_output_tokens} for large input")
            
            gen_model = genai.GenerativeModel(
                validated_model,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                ),
                system_instruction=system_msg
            )

            response = gen_model.generate_content(prompt)
            content = response.text
            
            # For large inputs in comprehensive mode, add a note about size
            if is_large_input and comprehensive:
                content = f"Note: Your input was large (approx. {estimate_token_count(prompt)} tokens), so I used direct processing mode.\n\n{content}"
                return content, f"{validated_model}-LargeInput"
            
            return content, validated_model
    except Exception as e:
        logger.error(f"Error with Google API call: {e}")
        
        # If the error is token-related, try again with direct mode
        error_str = str(e).lower()
        if "too large" in error_str or "token" in error_str or "limit" in error_str:
            logger.info("Token limit hit, attempting direct mode with minimal context")
            try:
                # Minimal context approach for token limit errors
                minimal_model = genai.GenerativeModel(
                    validated_model,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens
                    )
                )
                
                # Use a simple prompt without system message
                response = minimal_model.generate_content(prompt)
                return response.text, f"{validated_model}-DirectMode"
            except Exception as direct_error:
                logger.error(f"Direct mode also failed: {direct_error}")
                
        raise

async def generate_comprehensive_google_response(
    prompt: str,
    model_name: str,
    temperature: float,
    max_output_tokens: int
) -> str:
    """
    Generate a comprehensive response using Google Gemini with a two-step thinking process
    
    Args:
        prompt: The user's input prompt
        model_name: The Google model to use
        temperature: Creativity parameter
        max_output_tokens: Maximum tokens for the response
        
    Returns:
        str: The comprehensive response
    """
    try:
        # Get prompt length to adjust token allocation
        prompt_length = len(prompt)
        
        # Calculate thinking tokens - use less for longer prompts
        if prompt_length < 1000:
            thinking_tokens = max_output_tokens // 3  # Use 1/3 for short prompts
        elif prompt_length < 5000:
            thinking_tokens = max_output_tokens // 4  # Use 1/4 for medium prompts
        else:
            thinking_tokens = max_output_tokens // 5  # Use 1/5 for longer prompts
        
        # Adjust thinking prompt for efficiency
        thinking_prompt = f"""Analyze this query to plan a detailed response: "{prompt}"
        
Consider:
1. Key aspects and necessary background
2. Helpful examples or analogies
3. Different perspectives to include
4. Technical details to cover
5. Best structure for the response
"""

        thinking_system = "Plan your response step by step."
        
        thinking_model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=thinking_tokens
            ),
            system_instruction=thinking_system
        )
        
        thinking_response = thinking_model.generate_content(thinking_prompt)
        thinking = thinking_response.text
        logger.info("Generated thinking step for comprehensive Google response")
        
        # Create an efficient enhanced prompt that doesn't repeat the full original prompt
        enhanced_prompt = f"""Based on this analysis: 
{thinking}

Provide a comprehensive response to: "{prompt.strip()[:200]}..." 
Include examples, different perspectives, and technical details in a well-structured format.
"""

        comprehensive_model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            ),
            system_instruction=COMPREHENSIVE_SYSTEM_MESSAGE
        )
        
        final_response = comprehensive_model.generate_content(enhanced_prompt)
        return final_response.text
        
    except Exception as e:
        logger.error(f"Error generating comprehensive Google response: {e}")
        # Fall back to a standard response
        return await fallback_google_response(prompt, model_name, temperature, max_output_tokens)

async def fallback_google_response(prompt: str, model_name: str, temperature: float, max_output_tokens: int) -> str:
    """Fallback method if the comprehensive approach fails"""
    try:
        # Use minimal system instructions to save tokens
        fallback_model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        )
        
        fallback_response = fallback_model.generate_content(prompt)
        
        return fallback_response.text
    except Exception as fallback_error:
        logger.error(f"Fallback response also failed: {fallback_error}")
        return f"I'm sorry, but I encountered an error while generating a response. Please try again later."

async def expand_google_response(initial_response: str, model_name: str = "gemini-1.5-pro") -> str:
    """
    Expand an initial response to add more details and depth
    
    Args:
        initial_response: The initial response to expand
        model_name: The Google model to use
        
    Returns:
        str: The expanded response
    """
    try:
        # Skip expansion for very long responses
        if len(initial_response) > 10000:
            return initial_response + "\n\n(Response already comprehensive; expansion skipped due to length.)"
            
        expansion_prompt = f"""Expand this response with more details and examples:

{initial_response}
"""

        expansion_model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=4096
            )
        )
        
        expanded_response = expansion_model.generate_content(expansion_prompt)
        expanded_content = expanded_response.text
        
        final_response = (
            f"{initial_response}\n\n"
            f"--- Additional Details ---\n\n"
            f"{expanded_content}"
        )
        
        return final_response
    except Exception as e:
        logger.error(f"Error expanding Google response: {e}")
        return initial_response

