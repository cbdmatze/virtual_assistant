import logging
import sys
import time
import os

# API Keys
OPENAI_API_KEY = "Place here your OPENAI_API_KEY"
OPENAI_API_BASE = "https://api.openai.com/v1"  # Default OpenAI API base URL

ANTHROPIC_API_KEY = "Place here your ANTHROPIC_API_KEY"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"  # Default Anthropic API base URL

GROQ_API_KEY = "Place here your GROQ_API_KEY"
GROQ_API_BASE = "https://api.groq.com/v1"  # Default Groq API base URL

GOOGLE_API_KEY = "Place here your GOOGLE_API_KEY"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1"  # Default Google API base URL

GOOGLE_CSE_ID = "Place here your GOOGLE_CSE_ID"
GOOGLE_CSE_BASE_URL = "https://www.googleapis.com/customsearch/v1"  # Default Google CSE base URL

# Specify fixed models for special providers
HUGGINGFACE_MODEL = "gpt2"  # HuggingFace always uses gpt2
# Updated to use a Google Gemini model instead of OpenAI model
LANGCHAIN_MODEL = "gemini-1.5-pro"  # Changed from gpt-3.5-turbo-instruct to gemini-1.5-pro
LANGCHAIN_FRONTEND_MODEL = "google-gemini"  # Updated from google-search to google-gemini

# New LangGraph model identifiers
LANGGRAPH_MODEL = "gemini-1.5-pro"  # Default model for LangGraph
LANGGRAPH_FRONTEND_MODEL = "google-gemini-graph"  # Frontend identifier for LangGraph

# YouTube API Configuration
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_API_ENABLED = True  # Set to False to disable YouTube functionality
YOUTUBE_MAX_RESULTS = 5  # Maximum number of YouTube search results to return
YOUTUBE_SAFE_SEARCH = "moderate"  # Options: "none", "moderate", "strict"

# YouTube Player Configuration
YOUTUBE_PLAYER_WIDTH = 640  # Default width for embedded player
YOUTUBE_PLAYER_HEIGHT = 360  # Default height for embedded player
YOUTUBE_AUTOPLAY = True  # Whether videos should autoplay when opened
YOUTUBE_PLAYER_THEME = "dark"  # Options: "dark", "light"

# Debug and Logging Configuration
DEBUG_MODE = True  # Set to True to enable debug output in console
COLORED_LOGGING = True  # Set to True to enable colored logging output
CHAIN_OF_THOUGHT_VISIBLE = True  # Set to True to show reasoning in responses

# Local Video Player Options (for youtube_play_locally tool)
PREFERRED_PLAYERS = ["mpv", "vlc", "browser"]  # Order of preference for local players

# HTML Player Settings
HTML_PLAYER_TEMP_DIR = os.path.join(os.path.expanduser("~"), "bulls_eye_temp")
if not os.path.exists(HTML_PLAYER_TEMP_DIR):
    try:
        os.makedirs(HTML_PLAYER_TEMP_DIR)
    except Exception:
        HTML_PLAYER_TEMP_DIR = None  # Fall back to system temp directory if creation fails

# Database Configuration
DB_CONFIG = {
    "host": "place your host here",
    "user": "enter your user here",
    "password": "enter your password here",
    "database": "enter your database name here",
}

# RapidAPI Configuration
RAPIDAPI_CONFIG = {
    "key": "Place here your RAPIDAPI_KEY",
    "host": "chatgpt-vision1.p.rapidapi.com",
    "url": "https://chatgpt-vision1.p.rapidapi.com/texttoimage3"
}

# Valid model names
VALID_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2"
]

VALID_GROQ_MODELS = [
    'llama-3.3-70b-versatile',
    'qwen-2.5-32b',
    'llama-3.2-1b-preview',
    'gemma2-9b-it',
    'mixtral-8x7b-32768',
    'deepseek-r1-distill-llama-70b',
    'qwen-2.5-coder-23b',
    'llama3-8b-8192',
    'llama-3.2-11b-vision-preview',
    'llama-3.2-90b-vision-preview'
]

# Define preferred models for Google GenAI integration
PREFERRED_GOOGLE_MODELS = [
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gemini-1.0-pro',
    'gemini-1.5-flash-latest'
]

VALID_GOOGLE_MODELS = [
    'gemini-1.5-flash',
    'gemini-1-5-flash-002',
    'gemini-1.5-flash-8b',
    'gemini-1-5-flash-8b-001',
    'gemini-1.5-flash-8b-latest',
    'gemini-1.5-flash-8b-exp-0827',
    'gemini-1.5-flash-8b-exp-0927',
    'gemini-2.0-flash-exp',
    'gemini-2.0-flash',
    'gemini-1.5-pro-latest',
    'gemini-1.5-pro-001',
    'gemini-1.5-pro-002',
    'gemini-1.5-pro',
    'gemini-1.5-flash-latest',
    'gemini-1.5-flash-001',
    'gemini-1.5-flash-001-tuning',
    'gemini-2.0-flash-001',
    'gemini-2.0-flash-lite-001',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash-lite-preview-02-05',
    'gemini-2.0-flash-lite-preview',
    'gemini-2.0-pro-exp',
    'gemini-2.0-pro-exp-02-05',
    'gemini-exp-1206',
    'gemini-2.0-flash-thinking-exp-01-21',
    'gemini-2.0-flash-thinking-exp-1219',
    'learnlm-1.5-pro-experimental'
]


def setup_logging():
    """Configure and return a logger instance"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if COLORED_LOGGING:
        try:
            # Try to use colorlog if available
            import colorlog
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s' + log_format,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))
            logger = colorlog.getLogger(__name__)
            logger.addHandler(handler)
        except ImportError:
            # Fall back to standard logging
            logging.basicConfig(level=logging.INFO, format=log_format)
            logger = logging.getLogger(__name__)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
        logger = logging.getLogger(__name__)
        
    return logger


def print_with_typing_effect(text, delay=0.03):
    """Print text with a typing effect if enabled in config"""
    if not CHAIN_OF_THOUGHT_VISIBLE:
        return
        
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()
    
