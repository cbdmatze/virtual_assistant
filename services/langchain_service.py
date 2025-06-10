import logging
from typing import Tuple, Dict, Any, Optional

from config import GOOGLE_API_KEY, LANGCHAIN_MODEL  # Assuming you have GOOGLE_API_KEY in config
from services.search_service import direct_google_search

logger = logging.getLogger(__name__)

# Global variables for LangChain components
bullseye_llm = None
search_agent = None
verbose_agent = None  # New agent for comprehensive responses

# Flag to check if LangChain imports are available
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.error("LangChain imports failed - some functionality will be limited")
    LANGCHAIN_AVAILABLE = False

# Comprehensive system instruction template
VERBOSE_SYSTEM_INSTRUCTION = """You are an AI assistant that provides extremely detailed, comprehensive answers. 
Always explain concepts thoroughly with multiple examples, analogies, and explore different perspectives on each topic.
For technical topics, include code examples when relevant.
For factual questions, provide context, history, and related information.
Always aim to be educational and insightful with long-form, well-structured responses.
"""

def setup_langchain_components():
    """Set up LangChain components with error handling and diagnostics, now using Google GenAI"""
    global bullseye_llm, search_agent, verbose_agent

    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain components not available - imports failed")
        return None, None

    try:
        logger.info(f"Setting up LangChain with Google GenAI model '{LANGCHAIN_MODEL}'")

        # First, verify the Google API key (basic check - you might need more robust validation)
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set in config.py")
            return None, None
        logger.info("GOOGLE_API_KEY is set (basic check)")

        # Set up LangChain LLM - using ChatGoogleGenerativeAI for Google models
        # With enhanced parameters for more comprehensive responses
        try:
            bullseye_llm = ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY,
                temperature=0.8,  # Increased for more creative responses
                model=LANGCHAIN_MODEL,
                max_output_tokens=8192,  # Increased to maximum
                top_p=0.95,  # Added for more diverse text generation
                top_k=40,     # Added to increase response variety
                system_instruction=VERBOSE_SYSTEM_INSTRUCTION  # Added detailed system instruction
            )
            logger.info(f"ChatGoogleGenerativeAI initialized with model '{LANGCHAIN_MODEL}' and enhanced parameters")
        except Exception as genai_init_error:
            logger.error(f"Error initializing ChatGoogleGenerativeAI: {genai_init_error}")
            return None, None

        # Test LLM directly
        try:
            logger.info("Testing LLM directly")
            llm_result = bullseye_llm.invoke("Hello, this is a test.")
            logger.info("LLM test successful")
        except Exception as llm_error:
            logger.error(f"LLM test failed: {llm_error}")
            return bullseye_llm, None

        # Test Google Search directly before creating the tool
        from services.search_service import test_google_api
        try:
            logger.info("Testing Google Search API directly")
            test_result = test_google_api()
            if not test_result:
                logger.error("Google Search API test failed")
                return bullseye_llm, None
            logger.info("Google Search API test successful")
        except Exception as search_error:
            logger.error(f"Google Search API test failed: {search_error}")
            return bullseye_llm, None

        # Create search tools
        from services.search_service import google_search
        search_tools = [
            Tool(
                name="GoogleSearch",
                func=google_search,
                description="Useful for when you need to search for information online. Input should be a search query."
            )
        ]

        # Create standard agent
        try:
            logger.info("Creating standard search agent")
            memory = ConversationBufferMemory(memory_key="chat_history")
            agent = initialize_agent(
                search_tools,
                bullseye_llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True
            )
            logger.info("Standard search agent created successfully")
            search_agent = agent
            
            # Create verbose agent (for comprehensive responses)
            logger.info("Creating verbose search agent")
            verbose_memory = ConversationBufferMemory(memory_key="chat_history")
            verbose_agent = initialize_agent(
                search_tools,
                bullseye_llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # More detailed, conversational responses
                verbose=True,
                memory=verbose_memory,
                handle_parsing_errors=True
            )
            logger.info("Verbose search agent created successfully")
            
            return bullseye_llm, search_agent
        except Exception as agent_error:
            logger.error(f"Error creating search agent: {agent_error}")
            return bullseye_llm, None

    except Exception as e:
        logger.error(f"Error in LangChain setup: {str(e)}")
        return None, None

def check_langchain_status() -> Tuple[bool, Optional[str]]:
    """Check if LangChain components are available"""
    global bullseye_llm, search_agent, verbose_agent

    if not LANGCHAIN_AVAILABLE:
        return False, "LangChain imports failed"

    if bullseye_llm is None:
        return False, "LLM not initialized"

    if search_agent is None:
        return True, "LLM available but agent not initialized"
        
    if verbose_agent is None:
        return True, "Standard agent available but verbose agent not initialized"

    return True, None

def enhance_prompt(prompt: str) -> str:
    """Enhance the prompt to encourage more detailed responses"""
    enhanced_prompt = (
        f"I need a comprehensive, detailed response to this question: {prompt}\n\n"
        f"Please provide extensive information, multiple examples, and different perspectives. "
        f"Structure your response with clear sections, and don't hesitate to be thorough and educational."
    )
    return enhanced_prompt

async def expand_response(initial_response: str) -> str:
    """Expand an initial response to make it more comprehensive"""
    global bullseye_llm
    
    if not bullseye_llm:
        return initial_response
        
    try:
        # Create a follow-up prompt to expand the initial response
        expansion_prompt = (
            f"The following is a response to a user query:\n\n{initial_response}\n\n"
            f"Please expand on this response, adding more details, examples, and depth. "
            f"Make it more comprehensive while maintaining accuracy."
        )
        
        expanded = bullseye_llm.invoke(expansion_prompt)
        
        # Format the final response
        final_response = (
            f"{initial_response}\n\n"
            f"--- Additional Details ---\n\n"
            f"{expanded.content}"
        )
        
        return final_response
    except Exception as expand_error:
        logger.error(f"Error expanding response: {expand_error}")
        return initial_response

async def generate_langsearch_response(prompt: str, comprehensive: bool = True) -> Tuple[str, str]:
    """
    Generate a response using LangChain Google Search agent or fall back to direct search
    With option for comprehensive responses

    Args:
        prompt: The user's input prompt
        comprehensive: Whether to generate a comprehensive response (default: True)

    Returns:
        Tuple[str, str]: (generated content, model used)
    """
    global search_agent, verbose_agent

    # First try direct search to make sure the Google API works
    try:
        test_search = direct_google_search("quick test")
        if "Error" in test_search:
            logger.error(f"Google Search API is not working: {test_search}")
            return f"Sorry, I couldn't access Google Search at the moment. Error: {test_search}...Please try again later.", "Error-Google-Search"
    except Exception as test_error:
        logger.error(f"Error testing Google Search API: {test_error}")

    # Then try using the agent if available
    if search_agent is None and verbose_agent is None:
        logger.warning("LangChain Google Search agent is not available, using direct search instead.")
        content = direct_google_search(prompt)
        return content, "Direct-Google-Search"

    try:
        # Choose the appropriate agent based on comprehensive flag
        agent_to_use = verbose_agent if comprehensive and verbose_agent is not None else search_agent
        
        # Enhance the prompt for more detailed responses if comprehensive mode is on
        actual_prompt = enhance_prompt(prompt) if comprehensive else prompt
        
        if hasattr(agent_to_use, 'invoke'):
            result = agent_to_use.invoke({"input": actual_prompt})
            response = result['output']
        else:
            response = agent_to_use.run(actual_prompt)
        
        # If comprehensive mode is enabled, expand the response further
        if comprehensive:
            expanded_response = await expand_response(response)
            return expanded_response, f"{LANGCHAIN_MODEL}-Comprehensive"
        else:
            return response, LANGCHAIN_MODEL
            
    except Exception as e:
        logger.error(f"Error using LangChain agent: {e}, falling back to direct search")
        content = direct_google_search(prompt)
        return content, "Direct-Google-Search-Fallback"

async def test_langchain_search(query: str, comprehensive: bool = True) -> Dict[str, Any]:
    """Test LangChain search functionality with option for comprehensive responses"""
    global search_agent, verbose_agent

    # Choose the appropriate agent based on comprehensive flag
    agent_to_use = verbose_agent if comprehensive and verbose_agent is not None else search_agent
    
    if agent_to_use is None:
        return {
            "success": False,
            "method": "langchain_test",
            "query": query,
            "error": "LangChain agent is not available"
        }

    try:
        # Enhance the prompt for more detailed responses if comprehensive mode is on
        actual_query = enhance_prompt(query) if comprehensive else query
        
        if hasattr(agent_to_use, 'invoke'):
            result = agent_to_use.invoke({"input": actual_query})
            agent_result = result.get("output", "No output")
        else:
            agent_result = agent_to_use.run(actual_query)
        
        # If comprehensive mode is enabled, expand the response further
        if comprehensive:
            expanded_result = await expand_response(agent_result)
            return {
                "success": True,
                "method": "langchain_agent_comprehensive",
                "query": query,
                "result": expanded_result
            }
        else:
            return {
                "success": True,
                "method": "langchain_agent",
                "query": query,
                "result": agent_result
            }
            
    except Exception as agent_error:
        logger.error(f"LangChain agent error: {agent_error}")
        return {
            "success": False,
            "method": "langchain_agent",
            "query": query,
            "error": str(agent_error)
        }
