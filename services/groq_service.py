import groq
import logging
import asyncio
from typing import Tuple, Dict, Any, Optional

# Assuming config.py exists and contains GROQ_API_KEY and VALID_GROQ_MODELS
# Ensure VALID_GROQ_MODELS in config.py contains the exact, current names
# for the models you want to use.
from config import GROQ_API_KEY, VALID_GROQ_MODELS

logger = logging.getLogger(__name__)

# Initialize Groq Async client
# Use AsyncGroq for non-blocking calls within async functions
groq_client = groq.AsyncGroq(api_key=GROQ_API_KEY)

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

def validate_groq_model(model_name: str) -> str:
    """
    Validates and returns a valid model name for Groq API.
    Defaults to the first model in VALID_GROQ_MODELS if the provided
    model name is not valid or if VALID_GROQ_MODELS is empty.
    Raises a ValueError if VALID_GROQ_MODELS is empty.
    """
    if not VALID_GROQ_MODELS:
         logger.error("VALID_GROQ_MODELS list in config.py is empty. Cannot validate or select a default Groq model.")
         # It's better to raise an error here than proceed with no valid model
         raise ValueError("VALID_GROQ_MODELS list is empty in config.py. Please configure at least one valid Groq model.")

    # Use the first model in the list as the default fallback if the requested model is invalid
    default_model = VALID_GROQ_MODELS[0]

    if model_name in VALID_GROQ_MODELS:
        return model_name

    logger.warning(f"Groq model: '{model_name}' not found in VALID_GROQ_MODELS ({VALID_GROQ_MODELS}). Using default model: {default_model}")
    return default_model

async def generate_groq_response(
    prompt: str,
    # Updated default model to a known valid one
    model: str = "llama3-8b-8192",
    temperature: float = 0.7,
    comprehensive: bool = True,
    max_tokens: int = 4096
) -> Tuple[str, str]:
    """
    Generate a response using Groq API with option for comprehensive responses

    Args:
        prompt: The user's input prompt
        model: The Groq model to use (must be in VALID_GROQ_MODELS or will default)
        temperature: Creativity parameter (0.0-1.0)
        comprehensive: Whether to generate a comprehensive response
        max_tokens: Maximum number of tokens in the response

    Returns:
        Tuple[str,str]: The generated response content and the model name used
    """
    # Validate the model name - this function handles defaulting
    validated_model = validate_groq_model(model)
    logger.info(f"Using Groq model: {validated_model} (requested: {model}) with comprehensive mode: {comprehensive}")

    try:
        if comprehensive:
            # For comprehensive mode, use the thinking process and detailed response
            response = await generate_comprehensive_groq_response(prompt, validated_model, temperature, max_tokens)
            # Append "-Comprehensive" suffix to the reported model name for comprehensive mode
            return response, f"{validated_model}-Comprehensive"
        else:
            # For standard mode, use the basic approach
            response = await groq_client.chat.completions.create(
                model=validated_model,
                messages=[
                    {"role": "system", "content": STANDARD_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content
            return content, validated_model
    except Exception as e:
        logger.error(f"Error with Groq API call for model {validated_model}: {e}")
        # Re-raise the exception after logging
        raise

async def generate_comprehensive_groq_response(
    prompt: str,
    model: str, # validated_model is passed here
    temperature: float,
    max_tokens: int
) -> str:
    """
    Generate a comprehensive response using Groq with a two-step thinking process

    Args:
        prompt: The user's input prompt
        model: The Groq model to use (already validated)
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

        # Use await with the async client
        thinking_response = await groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": thinking_system},
                {"role": "user", "content": thinking_prompt}
            ],
            max_tokens=max_tokens // 3,  # Use 1/3 of tokens for thinking
            temperature=temperature
        )

        thinking = thinking_response.choices[0].message.content
        logger.info("Generated thinking step for comprehensive Groq response")

        # Step 2: Now generate the comprehensive response using the thinking
        enhanced_prompt = f"""Based on the following analysis, please provide an extremely comprehensive,
detailed response to this question: "{prompt}"

Analysis:
{thinking}

Your response should be well-structured with clear sections, include multiple examples or case studies,
explore different perspectives, and provide deep insights. Make your response educational and thorough.
"""

        # Use await with the async client
        final_response = await groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPREHENSIVE_SYSTEM_MESSAGE},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return final_response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating comprehensive Groq response with model {model}: {e}")
        # Fall back to a standard response using the fallback function
        # Pass validated_model to fallback
        return await fallback_groq_response(prompt, model, temperature, max_tokens)


async def fallback_groq_response(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Fallback method if the comprehensive approach fails"""
    try:
        logger.warning(f"Attempting fallback response for model {model}")
        # Use a simpler approach with the comprehensive system message
        response = await groq_client.chat.completions.create(
            model=model, # Use the validated model passed in
            messages=[
                {"role": "system", "content": COMPREHENSIVE_SYSTEM_MESSAGE},
                {"role": "user", "content": f"Please provide a detailed, comprehensive answer to: {prompt}"}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content
    except Exception as fallback_error:
        logger.error(f"Fallback response also failed for model {model}: {fallback_error}")
        return f"I'm sorry, but I encountered an error while generating a response. Please try again later."

async def expand_groq_response(
    initial_response: str,
    # Updated default model to a known valid one
    model: str = "qwen/qwen3-32b"
) -> str:
    """
    Expand an initial response to add more details and depth

    Args:
        initial_response: The initial response to expand
        model: The Groq model to use (must be in VALID_GROQ_MODELS or will default)

    Returns:
        str: The expanded response
    """
    # Validate the model name - this function handles defaulting
    validated_model = validate_groq_model(model)
    logger.info(f"Expanding response using Groq model: {validated_model} (requested: {model})")

    try:
        expansion_prompt = f"""The following is a response to a user query:

{initial_response}

Please expand on this response, adding more details, examples, and depth.
Make it more comprehensive while maintaining accuracy and readability.
Add specific examples, technical details, and different perspectives where appropriate.
"""

        # Use await with the async client
        expansion_response = await groq_client.chat.completions.create(
            model=validated_model, # Use the validated model
            messages=[
                {"role": "system", "content": COMPREHENSIVE_SYSTEM_MESSAGE},
                {"role": "user", "content": expansion_prompt}
            ],
            max_tokens=2048, # Note: Max tokens for expansion response
            temperature=0.7
        )

        expanded_content = expansion_response.choices[0].message.content

        # Concatenate initial and expanded content
        final_response = (
            f"{initial_response}\n\n"
            f"--- Additional Details ---\n\n"
            f"{expanded_content}"
        )

        return final_response
    except Exception as e:
        logger.error(f"Error expanding Groq response with model {validated_model}: {e}")
        # Return the original response if expansion fails
        return initial_response

# Example usage (requires an async context, e.g., inside an async main function)
# async def main():
#     # Example using a specific model name you expect to work
#     # Make sure "llama3-70b-8192" is in your config.VALID_GROQ_MODELS list
#     prompt = "Explain the concept of quantum entanglement."
#     try:
#         response_content, used_model = await generate_groq_response(
#             prompt=prompt,
#             model="llama3-70b-8192", # Request a specific Llama model
#             comprehensive=True
#         )
#         print(f"Response ({used_model}):\n{response_content}")
#     except Exception as e:
#         print(f"Failed to generate response: {e}")

# if __name__ == "__main__":
#     # Configure basic logging if not already done
#     logging.basicConfig(level=logging.INFO)
#     # To run the example:
#     # asyncio.run(main())
