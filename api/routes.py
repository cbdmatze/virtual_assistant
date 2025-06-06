from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Response
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional, List
import logging
import webbrowser
import os
from pathlib import Path

from api.models import ChatRequest, ImageRequest, YouTubeRequest
from api.auth import oauth2_scheme
from database.crud import (
    execute_query, 
    fetch_all, 
    save_video_to_db, 
    get_video_by_id, 
    link_video_to_conversation,
    get_conversations_with_videos
)
from services.openai_service import generate_openai_response
from services.anthropic_service import generate_anthropic_response
from services.groq_service import generate_groq_response
from services.google_service import generate_google_response
from services.huggingface_service import generate_huggingface_response
from services.langchain_service import generate_langsearch_response
from services.langgraph_service import (
    generate_langgraph_response, 
    test_langgraph_agent,
    youtube_search,
    youtube_video_info,
    youtube_oembed,
    youtube_download,
    youtube_create_html_player,
    test_youtube_api,
    test_youtube_oembed,
    get_youtube_client
)
from services.image_service import generate_image_from_prompt
from services.ocr_service import extract_text_from_image
from config import YOUTUBE_API_ENABLED, YOUTUBE_PLAYER_WIDTH, YOUTUBE_PLAYER_HEIGHT

router = APIRouter(tags=["api"])
logger = logging.getLogger(__name__)

@router.post("/chat")
async def chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    """Handle chat requests to various AI providers"""
    user_id = int(token)
    logger.info(f"Chat request received: provider={request.api_provider}, model={request.model}, temp={request.temperature}")
   
    try:
        # Route to appropriate service based on provider
        if request.api_provider == "anthropic":
            content, used_model = await generate_anthropic_response(request.prompt, request.model, request.temperature)
        elif request.api_provider == "groq":
            content, used_model = await generate_groq_response(request.prompt, request.model, request.temperature)
        elif request.api_provider == "google":
            content, used_model = await generate_google_response(request.prompt, request.model, request.temperature)
        elif request.api_provider == "huggingface":
            content, used_model = generate_huggingface_response(request.prompt)
        elif request.api_provider == "langchain":
            content, used_model = await generate_langsearch_response(request.prompt)
        elif request.api_provider == "langgraph":
            content, used_model = await generate_langgraph_response(request.prompt, request.model)
        else:  # Default to OpenAI
            content, used_model = await generate_openai_response(request.prompt, request.model, request.temperature)
        
        # Check if this is a response with video content
        video_data = None
        video_id = None
        video_type = None
        
        # Parse video metadata if present
        if content.startswith("VIDEO_RESPONSE_TYPE="):
            # Extract video metadata
            meta_end_idx = content.find("|", content.find("|") + 1) + 1
            metadata_str = content[:meta_end_idx]
            clean_content = content[meta_end_idx:]
            
            # Parse the metadata
            video_type = metadata_str.split("VIDEO_RESPONSE_TYPE=")[1].split("|")[0]
            video_id = metadata_str.split("VIDEO_ID=")[1].split("|")[0]
            
            # Replace content with clean version (without metadata)
            content = clean_content
            
            # Process video based on type
            try:
                if video_type == "embedded":
                    # Create embedded player data
                    embed_html = f"""
                    <div class="video-container">
                    <iframe
                        width="{YOUTUBE_PLAYER_WIDTH}"
                        height="{YOUTUBE_PLAYER_HEIGHT}"
                        src="https://www.youtube.com/embed/{video_id}"
                        title="YouTube video player"
                        frameborder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen
                    ></iframe>
                    </div>
                    """
                    
                    # Append the HTML directly to the content to make it more visible
                    content += f"\n\n{embed_html}"
                    
                    embed_data = {
                        "video_id": video_id,
                        "title": f"YouTube Video: {video_id}",
                        "channel": "Unknown Channel",
                        "type": "embedded",
                        "embed_html": embed_html,
                        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                    }
                    
                    # Save video to database
                    video_db_id = save_video_to_db(user_id, embed_data)
                    video_data = embed_data
                    
                elif video_type == "downloaded":
                    # Handle downloaded video (simplified for now)
                    video_db_id = None  # Will be set if properly downloaded
                    
                elif video_type == "reference":
                    # Create reference data
                    ref_data = {
                        "video_id": video_id,
                        "title": f"YouTube Video Reference: {video_id}",
                        "channel": "Unknown Channel",
                        "type": "reference",
                        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                    }
                    
                    video_db_id = save_video_to_db(user_id, ref_data)
                    video_data = ref_data
                
            except Exception as video_error:
                logger.error(f"Error processing video: {str(video_error)}")
                # Continue without video data if there's an error
        
        # Store conversation in database
        conversation_id = execute_query(
            "INSERT INTO conversations (user_id, conversation, model, temperature, api_provider) VALUES (%s, %s, %s, %s, %s)",
            (user_id, content, used_model, request.temperature, request.api_provider),
            return_last_id=True
        )
        
        # If video data is available, link it to the conversation
        if video_data and 'video_db_id' in locals() and video_db_id:
            link_video_to_conversation(conversation_id, video_db_id)
        
        # Include embedded video data in the response
        response_data = {"response": content}
        if video_type == "embedded":
            response_data["video_embed"] = {
                "video_id": video_id,
                "embed_html": embed_html if 'embed_html' in locals() else None
            }
        
        return response_data
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in chat endpoint: {error_message}")
        
        # For LangChain or LangGraph, try direct Google search as fallback
        if request.api_provider in ["langchain", "langgraph"]:
            try:
                logger.info("Attempting direct Google search as a fallback")
                from services.search_service import direct_google_search
                fallback_content = direct_google_search(request.prompt)
                
                # Store conversation with fallback info
                execute_query(
                    "INSERT INTO conversations (user_id, conversation, model, temperature, api_provider) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, fallback_content, f"Emergency-Google-fallback", request.temperature, f"{request.api_provider}-fallback")
                )
                
                return {"response": fallback_content}
            except Exception as fallback_error:
                logger.error(f"Even direct Google search fallback failed: {fallback_error}")
        
        raise HTTPException(status_code=500, detail=f"API error: {error_message}")

@router.post("/generate-image")
async def generate_image(request: ImageRequest, token: str = Depends(oauth2_scheme)):
    """Generate an image from a text prompt"""
    user_id = int(token)
    
    try:
        image_data, status = await generate_image_from_prompt(request.prompt, request.width, request.height, request.steps)
        
        if status["success"]:
            # Store the successful image generation in the database
            image_text = f"Generated the image from prompt: {request.prompt}"
            execute_query(
                "INSERT INTO conversations (user_id, conversation, model, temperature, api_provider, image_data) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, image_text, "Image Generator", 1.0, "rapidapi", image_data)
            )
            
            return {
                "response": image_text,
                "image": image_data,
                "success": True
            }
        else:
            return {
                "response": f"Error generating image: {status['error']}",
                "success": False
            }
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return {
            "response": f"Error generating image: {str(e)}",
            "success": False
        }

@router.get("/conversations")
async def get_conversations(token: str = Depends(oauth2_scheme)):
    """Get all conversations for the authenticated user"""
    user_id = int(token)
    
    # Use the enhanced function that includes video data
    conversations = get_conversations_with_videos(user_id)
    
    return {"conversations": conversations}

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, token: str = Depends(oauth2_scheme)):
    """Delete a specific conversation"""
    user_id = int(token)
    execute_query("DELETE FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id))
    return {"message": "Conversation deleted successfully"}

@router.delete("/conversations")
async def delete_all_conversations(token: str = Depends(oauth2_scheme)):
    """Delete all conversations for the authenticated user"""
    user_id = int(token)
    execute_query("DELETE FROM conversations WHERE user_id = %s", (user_id,))
    return {"message": "All conversations deleted successfully"}

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    """Extract text from an uploaded image using OCR"""
    user_id = int(token)
    
    try:
        ocr_text = await extract_text_from_image(file)
        
        execute_query(
            "INSERT INTO conversations (user_id, conversation, model, api_provider) VALUES (%s, %s, %s, %s)",
            (user_id, ocr_text, "OCR", "local")
        )
        
        return {"message": "Image uploaded and text extracted successfully", "ocr_text": ocr_text}
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/health")
async def health_check():
    """Check the health status of the API and its dependencies"""
    from services.openai_service import check_openai_status
    from services.search_service import test_google_api
    from services.langchain_service import check_langchain_status
    from services.langgraph_service import setup_langgraph_components
    from services.huggingface_service import check_huggingface_status
    from database.connection import check_database_connection
    from config import LANGGRAPH_MODEL, YOUTUBE_API_ENABLED
    
    openai_version, openai_status = check_openai_status()
    langchain_ready, langchain_error = check_langchain_status()
    
    # Quick check for LangGraph status
    langgraph_llm, langgraph_agent = setup_langgraph_components(LANGGRAPH_MODEL)
    langgraph_ready = langgraph_agent is not None
    
    # Check YouTube API if enabled
    youtube_api_ready = False
    youtube_oembed_ready = False
    if YOUTUBE_API_ENABLED:
        youtube_api_ready = test_youtube_api()
        youtube_oembed_ready = test_youtube_oembed()
    
    return {
        "status": "healthy",
        "openai_version": openai_version,
        "anthropic_models": ["claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "..."],
        "groq_models": ["llama-3.3-70b-versatile", "qwen-2.5-32b", "llama-3.2-1b-preview", "..."],
        "google_models": ["gemini-1.5-flash", "gemini-1-5-flash-002", "gemini-1.5-flash-8b", "..."],
        "huggingface_model": "gpt2",
        "langchain_model": "gpt-3.5-turbo-instruct",
        "langgraph_model": LANGGRAPH_MODEL,
        "huggingface_available": check_huggingface_status(),
        "langchain_agent_available": langchain_ready,
        "langchain_error": langchain_error,
        "langgraph_agent_available": langgraph_ready,
        "direct_google_search_available": test_google_api(),
        "youtube_api_enabled": YOUTUBE_API_ENABLED,
        "youtube_api_available": youtube_api_ready,
        "youtube_oembed_available": youtube_oembed_ready,
        "database_connected": check_database_connection(),
        "video_features_enabled": True,
        "fallback_enabled": True
    }

@router.get("/test-google-search")
async def test_google_search(query: str = "test", token: str = Depends(oauth2_scheme)):
    """Test endpoint for Google search functionality"""
    from services.search_service import test_google_api, google_search
    
    try:
        google_api_working = test_google_api()
        if not google_api_working:
            return {
                "success": False,
                "query": query,
                "error": "Google API configuration test failed. Check API key and CSE ID."
            }
        
        result = google_search(query)
        return {
            "success": True,
            "query": query,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }

@router.get("/test-langchain-search")
async def test_langchain_search(query: str = "test", token: str = Depends(oauth2_scheme)):
    """Test endpoint for LangChain search functionality"""
    from services.search_service import test_google_api, direct_google_search
    from services.langchain_service import test_langchain_search
    
    try:
        # Check Google API first
        google_api_working = test_google_api()
        if not google_api_working:
            return {
                "success": False,
                "method": "google_api_test",
                "query": query,
                "error": "Google API configuration test failed. Check API key and CSE ID."
            }
        
        # Try with LangChain
        langchain_result = await test_langchain_search(query)
        if langchain_result["success"]:
            return langchain_result
        
        # Fallback to direct search
        direct_result = direct_google_search(query)
        return {
            "success": True,
            "method": "direct_search",
            "query": query,
            "result": direct_result
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }

@router.get("/test-langgraph-search")
async def test_langgraph_search_endpoint(query: str = "test", token: str = Depends(oauth2_scheme)):
    """Test endpoint for LangGraph search functionality"""
    from services.search_service import test_google_api, direct_google_search
    from services.langgraph_service import test_langgraph_agent
    
    try:
        # Check Google API first
        google_api_working = test_google_api()
        if not google_api_working:
            return {
                "success": False,
                "method": "google_api_test",
                "query": query,
                "error": "Google API configuration test failed. Check API key and CSE ID."
            }
       
        # Try with LangGraph
        langgraph_result = await test_langgraph_agent(query)
        if langgraph_result["success"]:
            return langgraph_result
        
        # Fallback to direct search
        direct_result = direct_google_search(query)
        return {
            "success": True,
            "method": "direct_search",
            "query": query,
            "result": direct_result
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }
    

# NEW ENDPOINTS FOR YOUTUBE FUNCTIONALITY

@router.get("/youtube/search")
async def youtube_search_endpoint(
    query: str, 
    max_results: int = Query(5, ge=1, le=50),
    token: str = Depends(oauth2_scheme)
):
    """Search for videos on YouTube"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled in server configuration")
    
    user_id = int(token)
    try:
        # Parse YouTube search results
        search_response = youtube_search(query)
        
        # Parse the results into a more frontend-friendly format
        results = []
        for video_text in search_response.split("\n\n---\n\n"):
            if not video_text.strip():
                continue
                
            # Extract data from the text
            lines = video_text.strip().split("\n")
            video_data = {}
            
            for line in lines:
                if line.startswith("Title: "):
                    video_data["title"] = line[7:]
                elif line.startswith("Channel: "):
                    video_data["channel"] = line[9:]
                elif line.startswith("URL: "):
                    video_data["url"] = line[5:]
                    # Extract video ID from URL
                    if "v=" in line:
                        video_data["videoId"] = line.split("v=")[1].split("&")[0]
                    elif "youtu.be/" in line:
                        video_data["videoId"] = line.split("youtu.be/")[1]
                elif line.startswith("Thumbnail: "):
                    video_data["thumbnail"] = line[11:]
                elif line.startswith("Description: "):
                    video_data["description"] = line[13:]
            
            if "title" in video_data and "videoId" in video_data:
                results.append(video_data)
        
        # Log search in conversations
        execute_query(
            "INSERT INTO conversations (user_id, conversation, model, api_provider) VALUES (%s, %s, %s, %s)",
            (user_id, f"Searched YouTube for: {query}", "YouTube-Search", "youtube")
        )
        
        return {
            "success": True,
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"YouTube search error: {str(e)}")
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }

@router.get("/youtube/video/{video_id}")
async def youtube_video_endpoint(
    video_id: str,
    token: str = Depends(oauth2_scheme)
):
    """Get detailed information about a YouTube video"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled in server configuration")
    
    try:
        video_info = await youtube_video_info(video_id)
        return {
            "success": True,
            "video_id": video_id,
            "info": video_info
        }
    except Exception as e:
        logger.error(f"YouTube video info error: {str(e)}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e)
        }

@router.get("/youtube/oembed")
async def youtube_oembed_endpoint(
    url: str,
    width: int = Query(YOUTUBE_PLAYER_WIDTH, ge=200, le=1200),
    height: int = Query(YOUTUBE_PLAYER_HEIGHT, ge=200, le=1200),
    token: str = Depends(oauth2_scheme)
):
    """Get oEmbed HTML code for embedding a YouTube video"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled in server configuration")
    
    try:
        oembed_data = await youtube_oembed(url)
        return {
            "success": True,
            "url": url,
            "width": width,
            "height": height,
            "oembed_data": oembed_data
        }
    except Exception as e:
        logger.error(f"YouTube oEmbed error: {str(e)}")
        return {
            "success": False,
            "url": url,
            "error": str(e)
        }

@router.post("/youtube/player")
async def youtube_create_player_endpoint(
    request: YouTubeRequest,
    token: str = Depends(oauth2_scheme)
):
    """Create an HTML file with an embedded YouTube player and return its path"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled in server configuration")
    
    try:
        html_path = await youtube_create_html_player(
            request.video_id, 
            request.width or YOUTUBE_PLAYER_WIDTH, 
            request.height or YOUTUBE_PLAYER_HEIGHT
        )
        
        return {
            "success": True,
            "video_id": request.video_id,
            "player_path": html_path,
            "message": "HTML player created successfully"
        }
    except Exception as e:
        logger.error(f"YouTube player creation error: {str(e)}")
        return {
            "success": False,
            "video_id": request.video_id,
            "error": str(e)
        }
    

@router.get("/youtube/watch/{video_id}")
async def youtube_watch_endpoint(
    video_id: str,
    token: str = Depends(oauth2_scheme)
):
    """Open a YouTube video in the browser"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled in server configuration")
    
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        webbrowser.open(video_url)
        
        user_id = int(token)
        execute_query(
            "INSERT INTO conversations (user_id, conversation, model, api_provider) VALUES (%s, %s, %s, %s)",
            (user_id, f"Watched YouTube video: {video_url}", "YouTube-Browser", "youtube")
        )
        
        return {
            "success": True,
            "video_id": video_id,
            "message": f"Opening YouTube video {video_id} in browser",
            "url": video_url
        }
    except Exception as e:
        logger.error(f"YouTube watch error: {str(e)}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e)
        }

# NEW VIDEO STORAGE AND PLAYBACK ENDPOINTS

@router.post("/videos/{video_id}/embed")
async def create_embedded_player_response(
    video_id: str,
    width: int = Query(YOUTUBE_PLAYER_WIDTH, ge=200, le=1200),
    height: int = Query(YOUTUBE_PLAYER_HEIGHT, ge=200, le=1200),
    token: str = Depends(oauth2_scheme)
):
    """Create an embedded player for a YouTube video and save to history"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled")
    
    user_id = int(token)
    
    try:
        # Get video info
        youtube = get_youtube_client()
        video_info = None
        
        if youtube:
            try:
                video_response = youtube.videos().list(
                    part="snippet",
                    id=video_id
                ).execute()
                
                if video_response.get("items"):
                    video_info = video_response["items"][0]["snippet"]
            except Exception as e:
                logger.error(f"Error fetching video info: {str(e)}")
        
        # Create embed HTML for direct viewing (with autoplay)
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        embed_html_direct = f"""
        <iframe 
            width="{width}" 
            height="{height}" 
            src="{embed_url}?autoplay=1" 
            title="{video_info['title'] if video_info else 'YouTube video'}"
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
        """
        
        # Create embed HTML for history (without autoplay)
        embed_html_history = f"""
        <iframe 
            width="{width}" 
            height="{height}" 
            src="{embed_url}" 
            title="{video_info['title'] if video_info else 'YouTube video'}"
            frameborder="0" 
            allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
        """
        
        # Create video data
        video_data = {
            "video_id": video_id,
            "title": video_info['title'] if video_info else "YouTube Video",
            "channel": video_info['channelTitle'] if video_info else "Unknown channel",
            "type": "embedded",
            "embed_html": embed_html_history,  # Save non-autoplay version to database
            "thumbnail": video_info['thumbnails']['high']['url'] if video_info and 'thumbnails' in video_info else f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "width": width,
            "height": height
        }
        
        # Save to database
        video_db_id = save_video_to_db(user_id, video_data)
        
        if not video_db_id:
            raise HTTPException(status_code=500, detail="Failed to save video to database")
            
        # Save to conversations
        video_text = f"Embedded YouTube video: {video_data['title']}"
        conversation_id = execute_query(
            "INSERT INTO conversations (user_id, conversation, model, api_provider) VALUES (%s, %s, %s, %s)",
            (user_id, video_text, "YouTube", "youtube"),
            return_last_id=True
        )
        
        # Link video to conversation
        link_video_to_conversation(conversation_id, video_db_id)
        
        return {
            "success": True,
            "video_id": video_id,
            "video_db_id": video_db_id,
            "embed_html": embed_html_direct,  # Return the autoplay version for direct viewing
            "title": video_data["title"],
            "thumbnail": video_data["thumbnail"],
            "conversation_added": True
        }
    except Exception as e:
        logger.error(f"Error creating embedded player: {str(e)}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e)
        }

@router.post("/videos/{video_id}/download")
async def download_video_response(
    video_id: str,
    resolution: str = "720p",
    token: str = Depends(oauth2_scheme)
):
    """Download a YouTube video and save to history"""
    if not YOUTUBE_API_ENABLED:
        raise HTTPException(status_code=400, detail="YouTube API is disabled")
    
    user_id = int(token)
    
    try:
        # Download video
        video_download_result = await youtube_download(video_id, resolution)
        
        # Extract video data from the download result
        video_data = None
        
        # This is a complex parsing task - depends on the exact format of youtube_download's response
        # For now, just create basic video info as a fallback
        if not isinstance(video_download_result, dict):
            # Create basic video info
            video_data = {
                "video_id": video_id,
                "title": f"YouTube Video {video_id}",
                "channel": "Unknown channel",
                "type": "downloaded",
                "filepath": f"./storage/videos/{video_id}.mp4",  # Default path
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            }
        else:
            video_data = video_download_result
        
        # Save to database
        video_db_id = save_video_to_db(user_id, video_data)
        
        if not video_db_id:
            raise HTTPException(status_code=500, detail="Failed to save video to database")
        
        # Save to conversations
        video_text = f"Downloaded YouTube video: {video_data.get('title', video_id)}"
        conversation_id = execute_query(
            "INSERT INTO conversations (user_id, conversation, model, api_provider) VALUES (%s, %s, %s, %s)",
            (user_id, video_text, "YouTube", "youtube"),
            return_last_id=True
        )
        
        # Link video to conversation
        link_video_to_conversation(conversation_id, video_db_id)
        
        return {
            "success": True,
            "video_id": video_id,
            "video_db_id": video_db_id,
            "filepath": video_data.get("filepath"),
            "title": video_data.get("title"),
            "thumbnail": video_data.get("thumbnail"),
            "conversation_added": True
        }
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e)
        }

@router.get("/videos/{video_db_id}")
async def get_video(
    video_db_id: int,
    token: str = Depends(oauth2_scheme)
):
    """Get a saved video by database ID"""
    user_id = int(token)
    
    try:
        video_data = get_video_by_id(video_db_id, user_id)
        
        if not video_data:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # If it's a downloaded video, check if the file still exists
        if video_data["type"] == "downloaded" and video_data["filepath"]:
            file_path = Path(video_data["filepath"])
            if not file_path.exists():
                return {
                    "success": False,
                    "error": "Video file not found on server",
                    "video_data": video_data
                }
        
        return {
            "success": True,
            "video_data": video_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/videos/{video_db_id}/stream")
async def stream_video(
    video_db_id: int,
    token: str = Depends(oauth2_scheme)
):
    """Stream a downloaded video"""
    user_id = int(token)
    
    try:
        video_data = get_video_by_id(video_db_id, user_id)
        
        if not video_data:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if video_data["type"] != "downloaded" or not video_data["filepath"]:
            raise HTTPException(status_code=400, detail="This is not a downloaded video")
        
        file_path = Path(video_data["filepath"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on server")
        
        # Return a streaming response
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=f"{video_data['title']}.mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
