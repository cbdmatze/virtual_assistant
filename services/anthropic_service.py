import anthropic
import logging
import time
from typing import Tuple, Dict, Any, Optional

from config import ANTHROPIC_API_KEY, VALID_ANTHROPIC_MODELS

logger = logging.getLogger(__name__)

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Comprehensive system instruction that encourages detailed responses
COMPREHENSIVE_SYSTEM_MESSAGE = """You are an expert AI assistant that provides extremely detailed, comprehensive answers.
Your responses should:
- Explain concepts thoroughly with multiple examples and analogies
- Explore different perspectives on each topic
- Include relevant technical details, research findings, or historical context
- For technical topics, provide code examples when appropriate
- Structure your response with clear sections
- Be educational and insightful with well-organized information
- Think step by step before providing your final answer

Always aim to be thorough and exceed expectations in the depth and breadth of your responses.
"""

# Standard system message for regular responses
STANDARD_SYSTEM_MESSAGE = "You are a helpful assistant. When providing code snippets, use triple backticks (```) to format the code blocks with proper indentation and syntax highlighting."

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
        return model_mappings[model_name]  # Fixed: was using function call syntax instead of dictionary lookup
    
    # If no match found, return a default model
    logger.warning(f"Model: '{model_name}' not found. Using default model.")
    return "claude-3-5-sonnet-20241022"

async def generate_anthropic_response(
    prompt: str,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    comprehensive: bool = True,
    max_tokens: int = 4096
) -> Tuple[str, str]:
    """
    Generate a response using Anthropic API with option for comprehensive responses
    
    Args:
        prompt: The user's input prompt
        model: The Claude model to use
        temperature: Creativity parameter: (0 - 1)
        comprehensive: Whether to generate a comprehensive response
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    # Validate the model name
    validated_model = validate_anthropic_model(model)
    logger.info(f"Using Anthropic model: {validated_model} (requested: {model}) with comprehensive mode: {comprehensive}")

    try:
        if comprehensive:
            # For comprehensive mode, use the thinking process and detailed response
            response = await generate_comprehensive_claude_response(prompt, validated_model, temperature, max_tokens)
            return response, f"{validated_model}-Comprehensive"
        else:
            # For standard mode, use the basic approach
            system_message = STANDARD_SYSTEM_MESSAGE
            
            # Create the message payload
            messages = [
                {"role": "user", "content": prompt}
            ]

            # Generate a response using the Anthropic API
            response = anthropic_client.messages.create(
                model=validated_model,
                system=system_message,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            content = response.content[0].text
            return content, validated_model
    except Exception as e:
        logger.error(f"Error with Anthropic API call: {e}")
        raise

async def generate_comprehensive_claude_response(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Generate a comprehensive response using Claude with a two-step thinking process
    
    Args:
        prompt: The user's input prompt
        model: The Claude model to use
        temperature: Creativity parameter
        max_tokens: Maximum tokens for the response
        
    Returns:
        str: The comprehensive response
    """
    try:
        # Step 1: First, have the model think about the response
        thinking_prompt = f"""I need to provide a comprehensive response to this query: "{prompt}"
        
Let me think through this step by step before answering:
1. What are the key aspects of this question?
2. What background information would be helpful?
3. What examples, analogies, or case studies would illustrate this well?
4. What different perspectives should I consider?
5. What technical details or research findings should I include?
6. How should I structure my response for clarity?
"""

        thinking_system = "You are an expert thinking through a problem step by step. Be thorough in your analysis."
        
        thinking_response = anthropic_client.messages.create(
            model=model,
            system=thinking_system,
            messages=[{"role": "user", "content": thinking_prompt}],
            max_tokens=max_tokens // 3,  # Use 1/3 of tokens for thinking
            temperature=temperature
        )
        
        thinking = thinking_response.content[0].text
        logger.info("Generated thinking step for comprehensive Claude response")
        
        # Step 2: Now generate the comprehensive response using the thinking
        enhanced_prompt = f"""Based on the following analysis, please provide an extremely comprehensive, 
detailed response to this question: "{prompt}"

Analysis:
{thinking}

Your response should be well-structured with clear sections, include multiple examples or case studies,
explore different perspectives, and provide deep insights. Make your response educational and thorough.
"""

        final_response = anthropic_client.messages.create(
            model=model,
            system=COMPREHENSIVE_SYSTEM_MESSAGE,
            messages=[{"role": "user", "content": enhanced_prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return final_response.content[0].text
        
    except Exception as e:
        logger.error(f"Error generating comprehensive Claude response: {e}")
        # Fall back to a standard response
        return await fallback_response(prompt, model, temperature, max_tokens)

async def fallback_response(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Fallback method if the comprehensive approach fails"""
    try:
        # Use a simpler approach with the comprehensive system message
        response = anthropic_client.messages.create(
            model=model,
            system=COMPREHENSIVE_SYSTEM_MESSAGE,
            messages=[{"role": "user", "content": f"Please provide a detailed, comprehensive answer to: {prompt}"}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.content[0].text
    except Exception as fallback_error:
        logger.error(f"Fallback response also failed: {fallback_error}")
        return f"I'm sorry, but I encountered an error while generating a response. Please try again later."

async def expand_anthropic_response(initial_response: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    """
    Expand an initial response to add more details and depth
    
    Args:
        initial_response: The initial response to expand
        model: The Claude model to use
        
    Returns:
        str: The expanded response
    """
    try:
        expansion_prompt = f"""The following is a response to a user query:

{initial_response}

Please expand on this response, adding more details, examples, and depth.
Make it more comprehensive while maintaining accuracy and readability.
Add specific examples, technical details, and different perspectives where appropriate.
"""

        expansion_response = anthropic_client.messages.create(
            model=model,
            system=COMPREHENSIVE_SYSTEM_MESSAGE,
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=2048,
            temperature=0.7
        )
        
        expanded_content = expansion_response.content[0].text
        
        final_response = (
            f"{initial_response}\n\n"
            f"--- Additional Details ---\n\n"
            f"{expanded_content}"
        )
        
        return final_response
    except Exception as e:
        logger.error(f"Error expanding Claude response: {e}")
        return initial_response

