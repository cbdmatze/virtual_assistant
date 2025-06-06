import logging
from typing import Tuple, Dict, Any, Optional

from config import GOOGLE_API_KEY, LANGCHAIN_MODEL  # Assuming you have GOOGLE_API_KEY in config
from services.search_service import direct_google_search

logger = logging.getLogger(__name__)

# Global variables for LangChain components
bullseye_llm = None
search_agent = None

# Flag to check if LangChain imports are available
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    # from langchain_openai import OpenAI, ChatOpenAI  # Removed OpenAI imports
    from langchain_google_genai import ChatGoogleGenerativeAI # Added Google GenAI import
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.error("LangChain imports failed - some functionality will be limited")
    LANGCHAIN_AVAILABLE = False

def setup_langchain_components():
    """Set up LangChain components with error handling and diagnostics, now using Google GenAI"""
    global bullseye_llm, search_agent

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
        try:
            bullseye_llm = ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                model=LANGCHAIN_MODEL,
                max_output_tokens = 4096 # e.g., "gemini-pro"
            )
            logger.info(f"ChatGoogleGenerativeAI initialized with model '{LANGCHAIN_MODEL}'")
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

        # Create agent
        try:
            logger.info("Creating search agent")
            memory = ConversationBufferMemory(memory_key="chat_history")
            agent = initialize_agent(
                search_tools,
                bullseye_llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True
            )
            logger.info("Search agent created successfully")
            search_agent = agent
            return bullseye_llm, search_agent
        except Exception as agent_error:
            logger.error(f"Error creating search agent: {agent_error}")
            return bullseye_llm, None

    except Exception as e:
        logger.error(f"Error in LangChain setup: {str(e)}")
        return None, None

def check_langchain_status() -> Tuple[bool, Optional[str]]:
    """Check if LangChain components are available"""
    global bullseye_llm, search_agent

    if not LANGCHAIN_AVAILABLE:
        return False, "LangChain imports failed"

    if bullseye_llm is None:
        return False, "LLM not initialized"

    if search_agent is None:
        return True, "LLM available but agent not initialized"

    return True, None

async def generate_langsearch_response(prompt: str) -> Tuple[str, str]:
    """
    Generate a response using LangChain Google Search agent or fall back to direct search

    Args:
        prompt: The user's input prompt

    Returns:
        Tuple[str, str]: (generated content, model used)
    """
    global search_agent

    # First try direct search to make sure the Google API works
    try:
        test_search = direct_google_search("quick test")
        if "Error" in test_search:
            logger.error(f"Google Search API is not working: {test_search}")
            return f"Sorry, I couldn't access Google Search at the moment. Error: {test_search}...Please try again later.", "Error-Google-Search"
    except Exception as test_error:
        logger.error(f"Error testing Google Search API: {test_error}")

    # Then try using the agent if available
    if search_agent is None:
        logger.warning("LangChain Google Search agent is not available, using direct search instead.")
        content = direct_google_search(prompt)
        return content, "Direct-Google-Search"

    try:
        if hasattr(search_agent, 'invoke'):
            result = search_agent.invoke({"input": prompt})
            return result['output'], LANGCHAIN_MODEL
        else:
            result = search_agent.run(prompt)
            return result, LANGCHAIN_MODEL
    except Exception as e:
        logger.error(f"Error using LangChain agent: {e}, falling back to direct search")
        content = direct_google_search(prompt)
        return content, "Direct-Google-Search-Fallback"

async def test_langchain_search(query: str) -> Dict[str, Any]:
    """Test LangChain search functionality"""
    global search_agent

    if search_agent is None:
        return {
            "success": False,
            "method": "langchain_test",
            "query": query,
            "error": "LangChain agent is not available"
        }

    try:
        if hasattr(search_agent, 'invoke'):
            result = search_agent.invoke({"input": query})
            agent_result = result.get("output", "No output")
        else:
            agent_result = search_agent.run(query)

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
