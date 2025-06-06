"""LangGraph implementation with Google Gemini for search-powered conversations"""
import logging
import requests
import json
import webbrowser
import os
import sys
import time
import platform
import subprocess
import tempfile
import base64
import uuid
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager
from langgraph.prebuilt import create_react_agent

# Add cache management to suppress file_cache warnings
from googleapiclient.discovery_cache.base import Cache
class MemoryCache(Cache):
    _CACHE = {}
    def get(self, url): return MemoryCache._CACHE.get(url)
    def set(self, url, content): MemoryCache._CACHE[url] = content

# Import YouTube-related libraries
from googleapiclient.discovery import build
from pytube import YouTube
from pytube.exceptions import RegexMatchError, VideoUnavailable

from config import (
    GOOGLE_API_KEY, 
    GOOGLE_CSE_ID, 
    VALID_GOOGLE_MODELS,
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    YOUTUBE_PLAYER_WIDTH,
    YOUTUBE_PLAYER_HEIGHT,
    CHAIN_OF_THOUGHT_VISIBLE,
    HTML_PLAYER_TEMP_DIR
)

logger = logging.getLogger(__name__)

# Suppress googleapiclient.discovery_cache INFO messages
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.WARNING)

# Global variables for LangGraph components
gemini_llm = None
langgraph_agent = None
debug_mode = True  # Set to True to enable colored debug output

# Video storage configuration
VIDEO_STORAGE_DIR = Path("./storage/videos")
VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_SIZE_MB = 100  # Maximum size of videos to download (to avoid huge files)
ALLOWED_VIDEO_FORMATS = ['mp4', 'webm']  # Allowed formats to download

# YouTube API client initialization
def get_youtube_client():
    """Initialize and return a YouTube API client"""
    try:
        youtube = build(
            YOUTUBE_API_SERVICE_NAME, 
            YOUTUBE_API_VERSION, 
            developerKey=GOOGLE_API_KEY,
            cache=MemoryCache()  # Use memory cache to avoid file_cache warnings
        )
        return youtube
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Error] Failed to initialize YouTube API client: {str(e)}\033[0m")
        else:
            logger.error(f"Failed to initialize YouTube API client: {str(e)}")
        return None

# Create search tool using the @tool decorator
@tool
def google_search(query: str) -> str:
    """
    Search for information on the web using Google Search API.
    
    Args:
        query: The search query string
        
    Returns:
        Search results formatted as text
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Executing Google Search for: {query}\033[0m")
    else:
        logger.info(f"Executing Google Search for: {query}")
        
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"

    try:
        response = requests.get(url).json()
        if "items" in response:
            # Enhanced: Include titles and links for better context
            results = []
            for item in response["items"][:5]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
            
            result_text = "\n\n---\n\n".join(results)
            if debug_mode:
                logger.info(f"\033[0;32m[Tool] Search returned {len(results)} results\033[0m")
            else:
                logger.info(f"Search returned {len(results)} results")
            return result_text
        else:
            if debug_mode:
                logger.warning(f"\033[0;31m[Tool] Search returned no results\033[0m")
            else:
                logger.warning(f"Search returned no results")
            return "No search results found."
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Search error: {str(e)}\033[0m")
        else:
            logger.error(f"Search error: {str(e)}")
        return f"Error performing search: {str(e)}"

@tool
def youtube_search(query: str, max_results: int = 5) -> str:
    """
    Search for videos on YouTube.
    
    Args:
        query: The search query for YouTube videos
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        YouTube search results formatted as text
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Executing YouTube Search for: {query}\033[0m")
    else:
        logger.info(f"Executing YouTube Search for: {query}")
    
    try:
        youtube = get_youtube_client()
        if not youtube:
            return "YouTube API client could not be initialized."
        
        # Execute search request
        search_response = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=max_results,
            type="video"  # Only search for videos
        ).execute()
        
        # Process search results
        if "items" in search_response and search_response["items"]:
            results = []
            for item in search_response["items"]:
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                channel = item["snippet"]["channelTitle"]
                description = item["snippet"]["description"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                thumbnail = item["snippet"]["thumbnails"]["high"]["url"]
                
                results.append(
                    f"Title: {title}\n"
                    f"Channel: {channel}\n"
                    f"URL: {video_url}\n"
                    f"Thumbnail: {thumbnail}\n"
                    f"Description: {description}\n"
                    f"Commands:\n"
                    f"- Watch in app: Use 'youtube_embed({video_id})'\n"
                    f"- Download video: Use 'youtube_download({video_id})'\n"
                    f"- Get video details: Use 'youtube_video_info({video_id})'"
                )
            
            result_text = "\n\n---\n\n".join(results)
            if debug_mode:
                logger.info(f"\033[0;32m[Tool] YouTube search returned {len(results)} videos\033[0m")
            else:
                logger.info(f"YouTube search returned {len(results)} videos")
            return result_text
        else:
            if debug_mode:
                logger.warning(f"\033[0;31m[Tool] YouTube search returned no results\033[0m")
            else:
                logger.warning(f"YouTube search returned no results")
            return "No YouTube videos found for that query."
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] YouTube search error: {str(e)}\033[0m")
        else:
            logger.error(f"YouTube search error: {str(e)}")
        return f"Error searching YouTube: {str(e)}"

@tool
def youtube_video_info(video_url_or_id: str) -> str:
    """
    Get detailed information about a YouTube video.
    
    Args:
        video_url_or_id: Full URL of the YouTube video or just the video ID
        
    Returns:
        Detailed information about the video including available formats
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Getting YouTube video info for: {video_url_or_id}\033[0m")
    else:
        logger.info(f"Getting YouTube video info for: {video_url_or_id}")
    
    # Extract video ID if a full URL was provided
    video_id = video_url_or_id
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        if "v=" in video_url_or_id:
            video_id = video_url_or_id.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url_or_id:
            video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        # Use YouTube API to get video info
        youtube = get_youtube_client()
        if not youtube:
            return "YouTube API client could not be initialized."
            
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()
        
        if not video_response.get("items"):
            return f"No video found with ID: {video_id}"
            
        video_data = video_response["items"][0]
        snippet = video_data["snippet"]
        statistics = video_data["statistics"]
        content_details = video_data["contentDetails"]
        
        # Get pytube info for stream information
        yt = YouTube(video_url)
        
        # Basic video info
        info = {
            "Title": snippet["title"],
            "Channel": snippet["channelTitle"],
            "Duration": content_details["duration"],
            "Views": statistics.get("viewCount", "N/A"),
            "Likes": statistics.get("likeCount", "N/A"),
            "Description": snippet["description"][:500] + "..." if len(snippet["description"]) > 500 else snippet["description"],
            "Published At": snippet["publishedAt"],
            "Video URL": video_url,
            "Embed URL": f"https://www.youtube.com/embed/{video_id}",
            "Thumbnail": snippet["thumbnails"]["high"]["url"],
            "Commands": [
                f"Watch in app: Use 'youtube_embed({video_id})'",
                f"Download video: Use 'youtube_download({video_id})'",
                f"Add to history: Use 'youtube_save_to_history({video_id})'"
            ],
            "Available Formats": []
        }
        
        # Get available formats from pytube
        for stream in yt.streams.filter(progressive=True):
            info["Available Formats"].append({
                "Resolution": stream.resolution,
                "FPS": stream.fps,
                "MIME Type": stream.mime_type,
                "File Size": f"{stream.filesize / (1024 * 1024):.2f} MB"
            })
        
        # Format the response
        result = [f"{key}: {value}" for key, value in info.items() if key != "Available Formats" and key != "Commands"]
        
        # Add commands
        result.append("\nCommands:")
        for cmd in info["Commands"]:
            result.append(f"  • {cmd}")
        
        # Add formats information
        if info["Available Formats"]:
            result.append("\nAvailable Formats:")
            for i, fmt in enumerate(info["Available Formats"], 1):
                result.append(f"  {i}. Resolution: {fmt['Resolution']}, FPS: {fmt['FPS']}, Size: {fmt['File Size']}")
        
        if debug_mode:
            logger.info(f"\033[0;32m[Tool] YouTube video info retrieved successfully\033[0m")
        else:
            logger.info(f"YouTube video info retrieved successfully")
        return "\n".join(result)
    
    except (RegexMatchError, VideoUnavailable) as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Invalid YouTube URL or video unavailable: {str(e)}\033[0m")
        else:
            logger.error(f"Invalid YouTube URL or video unavailable: {str(e)}")
        return f"Error: The provided URL is not a valid YouTube video or the video is unavailable."
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Error getting YouTube video info: {str(e)}\033[0m")
        else:
            logger.error(f"Error getting YouTube video info: {str(e)}")
        return f"Error getting YouTube video info: {str(e)}"

@tool
def youtube_oembed(video_url_or_id: str, width: int = YOUTUBE_PLAYER_WIDTH, height: int = YOUTUBE_PLAYER_HEIGHT) -> str:
    """
    Get embedded HTML player for a YouTube video to watch in the application.
    
    Args:
        video_url_or_id: Full URL of the YouTube video or just the video ID
        width: Width of the embedded player
        height: Height of the embedded player
        
    Returns:
        HTML embed code and metadata for display within the application
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Creating embedded player for: {video_url_or_id}\033[0m")
    else:
        logger.info(f"Creating embedded player for: {video_url_or_id}")
    
    # Extract video ID if a full URL was provided
    video_id = video_url_or_id
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        if "v=" in video_url_or_id:
            video_id = video_url_or_id.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url_or_id:
            video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    
    try:
        # Get basic video info
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
        
        # Generate embed HTML
        embed_html = f"""
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
        
        # Store the embedded video in history
        # This will be handled in the create_embedded_player_response function in routes.py
        
        result = {
            "video_id": video_id,
            "title": video_info['title'] if video_info else "YouTube Video",
            "channel": video_info['channelTitle'] if video_info else "Unknown channel",
            "thumbnail": video_info['thumbnails']['high']['url'] if video_info else f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "embed_url": embed_url,
            "embed_html": embed_html,
            "width": width,
            "height": height,
            "type": "youtube_embed"
        }
        
        # Format the response for the agent
        response_text = f"""
I've prepared a YouTube video player that will be embedded in the chat.

Title: {result['title']}
Channel: {result['channel']}
Video ID: {result['video_id']}

The video player has been embedded and should appear in your chat window.
This video will also be saved in your conversation history.

[EMBEDDED_YOUTUBE_PLAYER_{video_id}]
"""
        
        if debug_mode:
            logger.info(f"\033[0;32m[Tool] YouTube embedded player created successfully\033[0m")
        else:
            logger.info(f"YouTube embedded player created successfully")
        
        return response_text
    
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Error creating embedded player: {str(e)}\033[0m")
        else:
            logger.error(f"Error creating embedded player: {str(e)}")
        return f"Error creating embedded player: {str(e)}"

@tool
async def youtube_create_html_player(video_url_or_id: str, width: int = YOUTUBE_PLAYER_WIDTH, height: int = YOUTUBE_PLAYER_HEIGHT) -> str:
    """
    Create an HTML file with an embedded YouTube player and return its path.
    
    Args:
        video_url_or_id: Full URL of the YouTube video or just the video ID
        width: Width of the embedded player
        height: Height of the embedded player
        
    Returns:
        Path to the created HTML file
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Creating HTML player for: {video_url_or_id}\033[0m")
    else:
        logger.info(f"Creating HTML player for: {video_url_or_id}")
    
    # Extract video ID if a full URL was provided
    video_id = video_url_or_id
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        if "v=" in video_url_or_id:
            video_id = video_url_or_id.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url_or_id:
            video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
    
    try:
        # Get basic video info for the title
        youtube = get_youtube_client()
        video_title = f"YouTube Video - {video_id}"
        
        if youtube:
            try:
                video_response = youtube.videos().list(
                    part="snippet",
                    id=video_id
                ).execute()
                
                if video_response.get("items"):
                    video_title = video_response["items"][0]["snippet"]["title"]
            except Exception as e:
                logger.error(f"Error fetching video title: {str(e)}")
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{video_title}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background-color: #000;
                }}
                .video-container {{
                    width: {width}px;
                    height: {height}px;
                    max-width: 100%;
                }}
                iframe {{
                    width: 100%;
                    height: 100%;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <div class="video-container">
                <iframe 
                    src="https://www.youtube.com/embed/{video_id}?autoplay=1" 
                    title="{video_title}"
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen>
                </iframe>
            </div>
        </body>
        </html>
        """
        
        # Create a temp directory if it doesn't exist
        temp_dir = HTML_PLAYER_TEMP_DIR if HTML_PLAYER_TEMP_DIR else tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a unique filename
        filename = f"youtube_player_{video_id}_{uuid.uuid4().hex[:6]}.html"
        file_path = os.path.join(temp_dir, filename)
        
        # Write the HTML file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        if debug_mode:
            logger.info(f"\033[0;32m[Tool] YouTube HTML player created at: {file_path}\033[0m")
        else:
            logger.info(f"YouTube HTML player created at: {file_path}")
        
        return file_path
    
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Error creating HTML player: {str(e)}\033[0m")
        else:
            logger.error(f"Error creating HTML player: {str(e)}")
        return f"Error creating HTML player: {str(e)}"

@tool
def youtube_download(video_url_or_id: str, resolution: str = "720p") -> str:
    """
    Download a YouTube video and save it to the database.
    
    Args:
        video_url_or_id: Full URL of the YouTube video or just the video ID
        resolution: Desired resolution (default: 720p)
        
    Returns:
        Information about the downloaded video
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Downloading YouTube video: {video_url_or_id}\033[0m")
    else:
        logger.info(f"Downloading YouTube video: {video_url_or_id}")
    
    # Extract video ID if a full URL was provided
    video_id = video_url_or_id
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        if "v=" in video_url_or_id:
            video_id = video_url_or_id.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url_or_id:
            video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        # Get video info
        yt = YouTube(video_url)
        
        # Check video size first
        stream = yt.streams.filter(progressive=True, file_extension='mp4', resolution=resolution).first()
        
        # If requested resolution not available, get the best available
        if not stream:
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
        if not stream:
            return f"No suitable download stream found for video: {yt.title}"
            
        # Check file size
        file_size_mb = stream.filesize / (1024 * 1024)
        if file_size_mb > MAX_VIDEO_SIZE_MB:
            return f"Video is too large to download ({file_size_mb:.2f} MB). Maximum size is {MAX_VIDEO_SIZE_MB} MB."
        
        # Create a unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{video_id}_{unique_id}.mp4"
        filepath = VIDEO_STORAGE_DIR / filename
        
        # Download the video
        stream.download(output_path=str(VIDEO_STORAGE_DIR), filename=filename)
        
        # Check if the file was downloaded
        if not filepath.exists():
            return f"Error: Video download failed. File not created."
        
        # Get file size
        file_size = filepath.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Encode a small preview for immediate display
        # For a real preview, this would need to process the video to create a thumbnail
        # Here we'll use the YouTube thumbnail instead
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        
        # This information will be available to handle in the API route
        result = {
            "video_id": video_id,
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "filepath": str(filepath),
            "file_size_mb": f"{file_size_mb:.2f}",
            "resolution": stream.resolution,
            "thumbnail": thumbnail_url,
            "type": "downloaded_video"
        }
        
        # Format the response for the agent
        response_text = f"""
I've downloaded the YouTube video for you.

Title: {yt.title}
Channel: {yt.author}
Duration: {yt.length} seconds
Resolution: {stream.resolution}
File Size: {file_size_mb:.2f} MB

The video has been downloaded and will appear in your chat window.
It has also been saved to your conversation history for future viewing.

[DOWNLOADED_VIDEO_{video_id}]
"""
        
        if debug_mode:
            logger.info(f"\033[0;32m[Tool] YouTube video downloaded successfully: {filepath}\033[0m")
        else:
            logger.info(f"YouTube video downloaded successfully: {filepath}")
        
        return response_text
    
    except (RegexMatchError, VideoUnavailable) as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Invalid YouTube URL or video unavailable: {str(e)}\033[0m")
        else:
            logger.error(f"Invalid YouTube URL or video unavailable: {str(e)}")
        return f"Error: The provided URL is not a valid YouTube video or the video is unavailable."
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Error downloading YouTube video: {str(e)}\033[0m")
        else:
            logger.error(f"Error downloading YouTube video: {str(e)}")
        return f"Error downloading YouTube video: {str(e)}"
    

@tool
def youtube_save_to_history(video_url_or_id: str) -> str:
    """
    Save a YouTube video to the chat history without downloading or embedding.
    
    Args:
        video_url_or_id: Full URL of the YouTube video or just the video ID
        
    Returns:
        Confirmation message
    """
    if debug_mode:
        logger.info(f"\033[0;33m[Tool] Saving YouTube video to history: {video_url_or_id}\033[0m")
    else:
        logger.info(f"Saving YouTube video to history: {video_url_or_id}")
    
    # Extract video ID if a full URL was provided
    video_id = video_url_or_id
    if "youtube.com" in video_url_or_id or "youtu.be" in video_url_or_id:
        if "v=" in video_url_or_id:
            video_id = video_url_or_id.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url_or_id:
            video_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
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
        
        # Generate result for storage in the database
        # This will be handled in the API route
        result = {
            "video_id": video_id,
            "title": video_info['title'] if video_info else "YouTube Video",
            "channel": video_info['channelTitle'] if video_info else "Unknown channel",
            "thumbnail": video_info['thumbnails']['high']['url'] if video_info else f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "video_url": video_url,
            "type": "youtube_reference"
        }
        
        # Format the response for the agent
        response_text = f"""
I've saved this YouTube video to your history.

Title: {result['title']}
Channel: {result['channel']}
Video URL: {video_url}

You can access this video anytime from your conversation history.

[SAVED_VIDEO_REFERENCE_{video_id}]
"""
        
        if debug_mode:
            logger.info(f"\033[0;32m[Tool] YouTube video saved to history successfully\033[0m")
        else:
            logger.info(f"YouTube video saved to history successfully")
        
        return response_text
    
    except Exception as e:
        if debug_mode:
            logger.error(f"\033[0;31m[Tool] Error saving YouTube video to history: {str(e)}\033[0m")
        else:
            logger.error(f"Error saving YouTube video to history: {str(e)}")
        return f"Error saving YouTube video to history: {str(e)}"

# Setup LangGraph components
def setup_langgraph_components(model_name: str = "gemini-1.5-pro"):
    """Set up LangGraph components with Google Gemini model"""
    global gemini_llm, langgraph_agent
    
    try:
        logger.info(f"Setting up LangGraph with Google GenAI model '{model_name}'")
        
        # Check API key
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set in config.py")
            return None, None
            
        logger.info("GOOGLE_API_KEY is set (basic check)")

        # Set up callbacks for verbose output
        callbacks = [StreamingStdOutCallbackHandler()] if debug_mode else []
        
        # Initialize the LLM with Google Gemini
        gemini_llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            model=model_name,
            callbacks=callbacks
        )
        
        logger.info(f"ChatGoogleGenerativeAI initialized with model '{model_name}'")
        
        # Test LLM directly
        try:
            logger.info("Testing LLM directly")
            test_message = [HumanMessage(content="Hello, this is a test.")]
            llm_result = gemini_llm.invoke(test_message)
            logger.info("LLM test successful")
        except Exception as llm_error:
            logger.error(f"LLM test failed: {llm_error}")
            return gemini_llm, None
            
        # Test Google Search directly
        try:
            logger.info("Testing Google Search API directly")
            test_result = google_search("test query")
            if "Error" in test_result:
                logger.error("Google Search API test failed")
                logger.error(f"Error details: {test_result}")
                return gemini_llm, None
            logger.info("Google Search API test successful")
        except Exception as search_error:
            logger.error(f"Google Search API test failed: {search_error}")
            return gemini_llm, None
            
        # Create the tools list
        tools = [
            google_search,
            youtube_search,
            youtube_video_info,
            youtube_oembed,
            youtube_download,
            youtube_save_to_history,
            youtube_create_html_player  # Added the new tool
        ]
        
        # Create the LangGraph agent
        try:
            logger.info("Creating LangGraph agent")
            agent = create_react_agent(gemini_llm, tools)
            logger.info("LangGraph agent created successfully")
            langgraph_agent = agent
            return gemini_llm, langgraph_agent
        except Exception as agent_error:
            logger.error(f"Error creating LangGraph agent: {agent_error}")
            return gemini_llm, None
            
    except Exception as e:
        logger.error(f"Error in LangGraph setup: {str(e)}")
        return None, None

def debug_print(message, type="INFO"):
    """Print debug messages with color if debug mode is enabled"""
    if not debug_mode:
        return
        
    color = "\033[0;34m"  # Blue for info
    if type == "THINKING":
        color = "\033[0;35m"  # Purple for thinking
    elif type == "ACTION":
        color = "\033[0;33m"  # Yellow for actions
    elif type == "RESULT":
        color = "\033[0;32m"  # Green for results
    
    logger.info(f"{color}[Agent {type}] {message}\033[0m")

async def generate_langgraph_response(prompt: str, model: str = "gemini-1.5-pro") -> Tuple[str, str]:
    """
    Generate a response using LangGraph with Google Gemini
    
    Args:
        prompt: The user's input prompt
        model: The Google Gemini model to use
        
    Returns:
        Tuple[str, str]: (generated content, model used)
    """
    global langgraph_agent
    
    if langgraph_agent is None:
        logger.warning("LangGraph agent is not available, setting up now...")
        _, langgraph_agent = setup_langgraph_components(model)
        if langgraph_agent is None:
            return "Sorry, the LangGraph agent couldn't be initialized. Please try again later.", f"Error-{model}"
    
    try:
        debug_print(f"Received prompt: '{prompt}'", "INFO")
        debug_print("Starting to think about this query...", "THINKING")
        
        # LangGraph uses a different invocation pattern
        messages = [HumanMessage(content=prompt)]
        
        # Using callbacks to track the execution
        result = langgraph_agent.invoke(
            {"messages": messages},
            config={"callbacks": [StreamingStdOutCallbackHandler()]} if debug_mode else {}
        )
        
        # Extract the AI's response from the result
        if "messages" in result and len(result["messages"]) > 0:
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                final_response = ai_messages[-1].content
                debug_print(f"Final response generated", "RESULT")
                
                # Check for embedded video markers
                if "[EMBEDDED_YOUTUBE_PLAYER_" in final_response or "[DOWNLOADED_VIDEO_" in final_response or "[SAVED_VIDEO_REFERENCE_" in final_response:
                    # Extract video ID from the marker
                    video_id = None
                    video_type = None
                    
                    if "[EMBEDDED_YOUTUBE_PLAYER_" in final_response:
                        parts = final_response.split("[EMBEDDED_YOUTUBE_PLAYER_")
                        if len(parts) > 1:
                            video_id = parts[1].split("]")[0]
                            video_type = "embedded"
                    
                    elif "[DOWNLOADED_VIDEO_" in final_response:
                        parts = final_response.split("[DOWNLOADED_VIDEO_")
                        if len(parts) > 1:
                            video_id = parts[1].split("]")[0]
                            video_type = "downloaded"
                    
                    elif "[SAVED_VIDEO_REFERENCE_" in final_response:
                        parts = final_response.split("[SAVED_VIDEO_REFERENCE_")
                        if len(parts) > 1:
                            video_id = parts[1].split("]")[0]
                            video_type = "reference"
                    
                    # Mark response for special handling in routes.py
                    if video_id:
                        final_response = f"VIDEO_RESPONSE_TYPE={video_type}|VIDEO_ID={video_id}|" + final_response
                
                return final_response, model
        
        return "I couldn't generate a proper response.", model
    except Exception as e:
        logger.error(f"Error using LangGraph agent: {e}")
        return f"I encountered an error while processing your request: {str(e)}", f"Error-{model}"

async def test_langgraph_agent(query: str) -> Dict[str, Any]:
    """Test LangGraph functionality"""
    global langgraph_agent
    
    if langgraph_agent is None:
        return {
            "success": False,
            "method": "langgraph_test",
            "query": query,
            "error": "LangGraph agent is not available"
        }
    
    try:
        debug_print(f"Testing LangGraph with query: '{query}'", "INFO")
        messages = [HumanMessage(content=query)]
        
        result = langgraph_agent.invoke(
            {"messages": messages},
            config={"callbacks": [StreamingStdOutCallbackHandler()]} if debug_mode else {}
        )
        
        if "messages" in result and len(result["messages"]) > 0:
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                agent_result = ai_messages[-1].content
                return {
                    "success": True,
                    "method": "langgraph_agent",
                    "query": query,
                    "result": agent_result
                }
        
        return {
            "success": False,
            "method": "langgraph_agent",
            "query": query,
            "error": "Could not extract response from agent"
        }
    except Exception as agent_error:
        logger.error(f"LangGraph agent error: {agent_error}")
        return {
            "success": False,
            "method": "langgraph_agent",
            "query": query,
            "error": str(agent_error)
        }

def test_youtube_api():
    """Test the YouTube API connectivity"""
    try:
        youtube = get_youtube_client()
        if not youtube:
            return False
            
        # Test with a simple request
        request = youtube.videos().list(
            part="snippet",
            id="dQw4w9WgXcQ"  # Rick Astley's "Never Gonna Give You Up"
        )
        response = request.execute()
        
        if response and "items" in response:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"YouTube API test failed: {str(e)}")
        return False

def test_youtube_oembed():
    """Test the YouTube oEmbed API connectivity"""
    try:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
        response = requests.get(oembed_url)
        response.raise_for_status()
        
        oembed_data = response.json()
        if "html" in oembed_data:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"YouTube oEmbed API test failed: {str(e)}")
        return False
