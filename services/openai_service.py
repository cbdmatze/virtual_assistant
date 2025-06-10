import openai
import logging
import time
from typing import Tuple, Dict, Any, Optional

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

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

async def generate_openai_response(
    prompt: str, 
    model: str = "gpt-4o", 
    temperature: float = 0.8,
    comprehensive: bool = True,
    max_tokens: int = 4000
) -> Tuple[str, str]:
    """
    Generate a comprehensive response using OpenAI
    
    Args:
        prompt: The user's input prompt
        model: The OpenAI model to use
        temperature: Creativity parameter (0.0-2.0), higher values produce more diverse outputs
        comprehensive: Whether to use the comprehensive response mode
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        Tuple[str,str]: The generated response and the model used
    """
    logger.info(f"Using OpenAI model: {model} with comprehensive mode: {comprehensive}")

    try:
        if comprehensive:
            # For comprehensive mode, use a two-step process with thinking
            response = await generate_comprehensive_response(prompt, model, temperature, max_tokens)
            return response, f"{model}-Comprehensive"
        else:
            # For standard mode, use the basic approach
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
                max_tokens,
                temperature
            )
            return content, model
    except Exception as e:
        logger.error(f"Error with OpenAI API call: {e}")
        raise

async def generate_comprehensive_response(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Generate a comprehensive response using a two-step process
    
    Args:
        prompt: The user's input prompt
        model: The OpenAI model to use
        temperature: Creativity parameter
        max_tokens: Maximum tokens for the response
        
    Returns:
        str: The comprehensive response
    """
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
    
    try:
        thinking_messages = [
            {"role": "system", "content": "You are an expert thinking through a problem step by step. Be thorough in your analysis."},
            {"role": "user", "content": thinking_prompt}
        ]
        
        thinking = get_openai_completion(
            model, 
            thinking_messages,
            max_tokens // 2,  # Use half the tokens for thinking
            temperature
        )
        
        logger.info("Generated thinking step for comprehensive response")
        
        # Step 2: Now generate the comprehensive response using the thinking
        enhanced_prompt = f"""Based on the following analysis, please provide an extremely comprehensive, 
detailed response to this question: "{prompt}"

Analysis:
{thinking}

Your response should be well-structured with clear sections, include multiple examples or case studies,
explore different perspectives, and provide deep insights. Make your response educational and thorough.
"""

        response_messages = [
            {"role": "system", "content": COMPREHENSIVE_SYSTEM_MESSAGE},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        comprehensive_response = get_openai_completion(
            model,
            response_messages,
            max_tokens,
            temperature
        )
        
        return comprehensive_response
    except Exception as e:
        logger.error(f"Error generating comprehensive response: {e}")
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

async def expand_openai_response(initial_response: str, model: str = "gpt-4o", temperature: float = 0.7) -> str:
    """
    Expand an initial response to add more details and depth
    
    Args:
        initial_response: The initial response to expand
        model: The OpenAI model to use
        temperature: Creativity parameter
        
    Returns:
        str: The expanded response
    """
    try:
        expansion_prompt = f"""The following is a response to a user query:

{initial_response}

Please expand on this response, adding more details, examples, and depth.
Make it more comprehensive while maintaining accuracy and readability.
"""

        expansion_messages = [
            {"role": "system", "content": COMPREHENSIVE_SYSTEM_MESSAGE},
            {"role": "user", "content": expansion_prompt}
        ]
        
        expanded_content = get_openai_completion(
            model,
            expansion_messages,
            2000,  # Use 2000 tokens for expansion
            temperature
        )
        
        final_response = (
            f"{initial_response}\n\n"
            f"--- Additional Details ---\n\n"
            f"{expanded_content}"
        )
        
        return final_response
    except Exception as e:
        logger.error(f"Error expanding OpenAI response: {e}")
        return initial_response
a
