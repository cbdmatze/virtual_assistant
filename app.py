from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import time

# Import configuration
from config import (
    setup_logging, LANGCHAIN_MODEL, LANGGRAPH_MODEL, 
    YOUTUBE_API_ENABLED, DEBUG_MODE, CHAIN_OF_THOUGHT_VISIBLE
)

# Import API routers
from api.routes import router as api_router
from api.auth import router as auth_router

# Import initialization functions
from services.huggingface_service import init_huggingface
from services.langchain_service import setup_langchain_components
from services.langgraph_service import (
    setup_langgraph_components, 
    test_youtube_api, 
    test_youtube_oembed
)
from services.search_service import test_google_api
from database.connection import init_database

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(title="Bulls AI API", description="API for Bulls AI", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows requests from the frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers (Authorization, etc.)
)

# Include API routers
app.include_router(auth_router)
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Runs initialization code when the application starts"""
    logger.info("Starting Bulls AI API initialization...")
    
    # Initialize database
    logger.info("Initializing database connection...")
    init_database()

    # Initialize HuggingFace model
    logger.info("Loading HuggingFace model...")
    init_huggingface()

    # Test Google API setup
    logger.info("Testing Google API setup...")
    if test_google_api():
        logger.info("✅ Google API setup verified successfully")
    else:
        logger.error("❌ Google API setup verification failed - search functionality may not work")

    # Setup LangChain components
    logger.info(f"Setting up LangChain with Google Gemini model: {LANGCHAIN_MODEL}")
    bullseye_llm, search_agent = setup_langchain_components()
    if bullseye_llm:
        logger.info("✅ LangChain LLM initialized successfully")
    else:
        logger.warning("⚠️ LangChain LLM initialization failed")
        
    if search_agent:
        logger.info("✅ LangChain search agent initialized successfully")
    else:
        logger.warning("⚠️ LangChain search agent initialization failed")
        
    # Setup LangGraph components
    logger.info(f"Setting up LangGraph with Google Gemini model: {LANGGRAPH_MODEL}")
    gemini_llm, langgraph_agent = setup_langgraph_components(LANGGRAPH_MODEL)
    if gemini_llm:
        logger.info("✅ LangGraph LLM initialized successfully")
    else:
        logger.warning("⚠️ LangGraph LLM initialization failed")
        
    if langgraph_agent:
        logger.info("✅ LangGraph agent initialized successfully")
    else:
        logger.warning("⚠️ LangGraph agent initialization failed")
    
    # YouTube API testing
    if YOUTUBE_API_ENABLED:
        logger.info("Testing YouTube API connectivity...")
        youtube_api_working = test_youtube_api()
        if youtube_api_working:
            logger.info("✅ YouTube API connection successful")
        else:
            logger.warning("⚠️ YouTube API connection failed - video search functionality may not work")
            logger.warning("   Make sure YouTube Data API v3 is enabled for your Google API key")
        
        logger.info("Testing YouTube oEmbed API connectivity...")
        oembed_api_working = test_youtube_oembed()
        if oembed_api_working:
            logger.info("✅ YouTube oEmbed API connection successful")
        else:
            logger.warning("⚠️ YouTube oEmbed API connection failed - video embedding may not work")
    else:
        logger.info("YouTube API functionality is disabled in configuration")
    
    # Display debug configuration
    if DEBUG_MODE:
        logger.info("🔍 Debug mode is enabled")
        if CHAIN_OF_THOUGHT_VISIBLE:
            logger.info("💭 Chain-of-thought reasoning is visible")
    
    logger.info("Bulls AI API initialization complete ✨")


if __name__ == "__main__":
    # Add a small delay for logs to display cleanly
    time.sleep(0.1)
    
    # Run the application
    import uvicorn
    logger.info("Starting uvicorn server... 🚀")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
