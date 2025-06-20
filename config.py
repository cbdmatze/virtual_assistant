
import logging
import sys
import time
import os

# --- API Keys ---
# IMPORTANT SECURITY NOTE: Replace these with environment variables or a secrets management system
# for production use. Storing keys directly in code is NOT recommended and poses a significant security risk.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "place your Api_Key here") # Example/Placeholder
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "place your Api_Key here") # Example/Placeholder
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "place your Api_Key here") # Example/Placeholder
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "place your Api_Key here") # Example/Placeholder
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "place your CSE_ID here") # Example/Placeholder
# It is highly recommended to fetch API keys using os.getenv() as shown above,
# and store the actual keys in your system's environment variables.


# --- Default Model Selections ---
# Specify fixed models for specific integrations or purposes

HUGGINGFACE_MODEL = "gpt2"  # HuggingFace often requires specific model names

# Default model for Langchain integrations
# User requested to keep this as gemini-1.5-pro
LANGCHAIN_MODEL = "gemini-1.5-pro"
LANGCHAIN_FRONTEND_MODEL = "google-gemini"  # Frontend identifier for Langchain

# Default model for LangGraph integrations
# User requested to keep this as gemini-1.5-pro
LANGGRAPH_MODEL = "gemini-1.5-pro"
LANGGRAPH_FRONTEND_MODEL = "google-gemini-graph"  # Frontend identifier for LangGraph

# Default model specifically for a Google Generative AI model selector UI or logic
# Set as requested by the user based on available models
DEFAULT_GOOGLE_SELECTOR_MODEL = 'gemini-2.0-flash'


# --- Google GenAI Model Lists ---
# List of preferred Google models for common use cases.
# This list is a subset of VALID_GOOGLE_MODELS and might be used for UI dropdowns or default choices.
# Updated to include some of the newer models found.
PREFERRED_GOOGLE_MODELS = [
    'gemini-1.5-pro',          # A powerful, versatile model
    'gemini-1.5-flash-latest', # The latest 1.5 flash model
    'gemini-2.0-flash',        # The requested default for selector, also a good preferred option
    'gemma-3-4b-it',           # Example of a Gemma model
]
# Note: Model names in PREFERRED_GOOGLE_MODELS might sometimes omit the 'models/' prefix
# compared to the full list if the API handles both formats. Using common names here.


# List of ALL Google Generative AI models currently reported as supported by the API.
# This list is dynamically generated from the user's script output.
# IMPORTANT: This list may contain experimental, preview, or time-limited models.
# Models prefixed with 'models/' and those without are included as reported.
VALID_GOOGLE_MODELS = [
    'models/gemini-2.0-flash-exp',
    'models/gemini-1.5-flash-latest',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-flash-002',
    'models/gemini-1.5-flash-8b',
    'models/gemini-1.5-flash-8b-001',
    'models/gemini-1.5-flash-8b-latest',
    'models/gemini-2.5-flash-preview-04-17',
    'models/gemini-2.5-flash-preview-05-20',
    'models/gemini-2.5-flash-preview-04-17-thinking',
    'models/gemini-2.0-flash',
    'models/gemini-2.0-flash-001',
    'models/gemini-2.0-flash-lite-001',
    'models/gemini-2.0-flash-lite',
    'models/gemini-2.0-flash-lite-preview-02-05',
    'models/gemini-2.0-flash-lite-preview',
    'models/gemini-2.0-flash-thinking-exp-01-21',
    'models/gemini-2.0-flash-thinking-exp',
    'models/gemini-2.0-flash-thinking-exp-1219',
    'models/learnlm-2.0-flash-experimental',
    'models/gemma-3-1b-it',
    'models/gemma-3-4b-it',
    'models/gemma-3-12b-it',
    'models/gemma-3-27b-it',
    'models/gemma-3n-e4b-it',
    'gemini-1.5-pro-latest',
    'gemini-1.5-pro-002',
    'gemini-1.5-pro',
]
# It is highly recommended to periodically fetch the list of supported models
# directly from the Google GenAI API (e.g., using genai.list_models())
# to ensure this list is always current.


# --- Other Provider Model Lists ---

# VALID_ANTHROPIC_MODELS: List of Anthropic models.
# Check Anthropic's documentation for the latest supported models.
VALID_ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",     # Most powerful model
    "claude-3-haiku-20240307",    # Fast and cost-effective
    "claude-3-5-sonnet-20240620"  # Latest Sonnet model, excellent balance of capabilities
]


# VALID_GROQ_MODELS: List of Groq models currently confirmed as working.
# This list was updated based on the user's provided manual check.
VALID_GROQ_MODELS = [
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'qwen-qwq-32b',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'deepseek-r1-distill-llama-70b',
    'compound-beta',
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'compound-beta-mini',
    'llama3-70b-8192',
    'qwen/qwen3-32b',
    'llama3-8b-8192',
    'allam-2-7b',
    'gemma2-9b-it',
]
# IMPORTANT: Periodically verify this list against Groq's official API or documentation.


# --- YouTube API Configuration ---
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_API_ENABLED = True  # Set to False to disable YouTube functionality
YOUTUBE_MAX_RESULTS = 10  # Maximum number of YouTube search results to return
YOUTUBE_SAFE_SEARCH = "moderate"  # Options: "none", "moderate", "strict"

# YouTube Player Configuration
YOUTUBE_PLAYER_WIDTH = 640  # Default width for embedded player
YOUTUBE_PLAYER_HEIGHT = 360  # Default height for embedded player
YOUTUBE_AUTOPLAY = True  # Whether videos should autoplay when opened
YOUTUBE_PLAYER_THEME = "dark"  # Options: "dark", "light"

# --- Debug and Logging Configuration ---
DEBUG_MODE = True  # Set to True to enable debug output in console
COLORED_LOGGING = True  # Set to True to enable colored logging output
CHAIN_OF_THOUGHT_VISIBLE = True  # Set to True to show reasoning in responses

# --- Local Video Player Options (for youtube_play_locally tool) ---
PREFERRED_PLAYERS = ["mpv", "vlc", "browser"]  # Order of preference for local players

# --- HTML Player Settings ---
HTML_PLAYER_TEMP_DIR = os.path.join(os.path.expanduser("~"), "bulls_eye_temp")
if not os.path.exists(HTML_PLAYER_TEMP_DIR):
    try:
        os.makedirs(HTML_PLAYER_TEMP_DIR)
    except Exception:
        # Fall back to system temp directory if creation fails
        HTML_PLAYER_TEMP_DIR = None
        logging.warning(f"Could not create temporary directory {HTML_PLAYER_TEMP_DIR}. Falling back to system temp.")


# --- Database Configuration ---
DB_CONFIG = {
    "host": "place your credentials here",
    "user": "place your credentials here",
    "password": "place your credentials here",
    "database": "place your credentials here"
}

# --- RapidAPI Configuration ---
RAPIDAPI_CONFIG = {
    "key": "place your RapidApi_Key here", # Example/Placeholder
    "host": "chatgpt-vision1.p.rapidapi.com",
    "url": "https://chatgpt-vision1.p.rapidapi.com/texttoimage3"
}

# --- Logging Setup Function ---
def setup_logging():
    """Configure and return a logger instance"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(__name__)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

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
            logger.addHandler(handler)
        except ImportError:
            # Fall back to standard logging if colorlog is not installed
            logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
            logger.warning("colorlog not installed. Using standard logging.")
    else:
        # Use standard logging without colors
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)

    # Set logging level based on DEBUG_MODE
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger

# --- Utility Function ---
def print_with_typing_effect(text, delay=0.03):
    """Print text with a typing effect if enabled in config"""
    if not CHAIN_OF_THOUGHT_VISIBLE:
        print(text) # Print normally if typing effect is disabled
        return

    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Initialize logging on import
logger = setup_logging()

# Example usage of the logger (optional, can be removed)
# logger.debug("Debug logging is enabled.")
# logger.info("Configuration loaded successfully.")
# logger.warning("Remember to use environment variables for API keys in production!")
