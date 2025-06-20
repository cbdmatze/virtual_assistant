
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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
gemini_llm_comprehensive = None # Keep comprehensive LLM separate if needed for config
langgraph_agent = None
langgraph_comprehensive_agent = None  # Agent using the comprehensive LLM
debug_mode = True  # Set to True to enable colored debug output

# Input threshold for switching modes (characters)
LARGE_INPUT_THRESHOLD = 10000

# New system instructions for comprehensive responses
COMPREHENSIVE_SYSTEM_INSTRUCTION = """You are an AI assistant that provides extremely detailed, comprehensive answers.
Your responses should:
- Explain concepts thoroughly with multiple examples and analogies
- Explore different perspectives on each topic
- Include relevant technical details, research findings, or historical context
- For technical topics, provide code examples when appropriate
- Structure your response with clear sections
- Be educational and insightful with well-organized information
""" # Removed "Think step by step" as that's handled in the thinking phase

# Video storage configuration
VIDEO_STORAGE_DIR = Path("./storage/videos")
VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_SIZE_MB = 100  # Maximum size of videos to download (to avoid huge files)
ALLOWED_VIDEO_FORMATS = ['mp4', 'webm']  # Allowed formats to download


# Define the missing debug_print function
def debug_print(message: str, level: str = "DEBUG"):
    """Prints colored debug messages if debug_mode is True"""
    if debug_mode:
        colors = {
            "INFO": "\033[0;34m",      # Blue
            "THINKING": "\033[0;33m",  # Yellow
            "ACTION": "\033[0;35m",    # Magenta
            "RESULT": "\033[0;32m",    # Green
            "DEBUG": "\033[0;36m",     # Cyan
        }
        color_code = colors.get(level.upper(), colors["DEBUG"])
        reset_code = "\033[0m"
        print(f"{color_code}[{level.upper()}] {message}{reset_code}")


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
                f"Get video details: Use 'youtube_video_info({video_id})'"
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
            logger.error(f"\033[0;31m[Tool] Error creating embedded player: {str(e)}\003[0m")
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


async def generate_thinking_response(prompt: str, model_obj: ChatGoogleGenerativeAI) -> str:
    """
    Have the model think through a response using a concise prompt.

    Args:
        prompt: The user's input prompt (or a summary/excerpt for very large inputs)
        model_obj: The LLM model object to use for thinking

    Returns:
        The concise thinking process result
    """
    debug_print("Starting concise thinking process...", "THINKING")

    # Use a concise thinking prompt to save tokens
    thinking_prompt = f"""Analyze the user's request to plan a detailed, comprehensive response.
User request (excerpt): "{prompt.strip()[:500]}..."

Consider:
1. Core requirements of the request.
2. What information (facts, concepts, examples, technical details) is needed?
3. Different perspectives or approaches to discuss.
4. How to structure the final answer effectively.
5. Potential tools to use (though the agent will handle execution).

Provide a concise outline of your plan.
"""

    try:
        # Use LangChain's invoke method with SystemMessage
        # We use the standard LLM here as the comprehensive one might have different System Instructions
        # Also, we rely on the concise prompt to limit the thinking output size,
        # as dynamically changing max_output_tokens per invoke is tricky with LangChain.
        messages = [
            SystemMessage(content="Plan a comprehensive response concisely."),
            HumanMessage(content=thinking_prompt)
        ]

        thinking_result = model_obj.invoke(messages)
        debug_print(f"Completed thinking process (approx. {len(thinking_result.content)} chars)", "THINKING")

        return thinking_result.content
    except Exception as e:
        logger.error(f"Error in thinking process: {str(e)}")
        return ""


def enhance_prompt_with_thinking(original_prompt: str, thinking_result: str) -> str:
    """
    Enhance a prompt with the results of the thinking process for the agent.

    Args:
        original_prompt: The original user query
        thinking_result: The result of the thinking process

    Returns:
        Enhanced prompt including thinking analysis and truncated original prompt
    """
    if not thinking_result.strip():
        return original_prompt

    # Use truncated original prompt to save tokens
    enhanced_prompt = f"""Based on the following analysis, provide an extremely comprehensive,
detailed response to the user's request.

Analysis:
{thinking_result}

Original Request (excerpt): "{original_prompt.strip()[:500]}..."

Ensure your response is well-structured with clear sections, include multiple examples or case studies,
explore different perspectives, and provide deep insights. Make your response educational and thorough.
Utilize available tools if necessary based on the analysis.
"""

    return enhanced_prompt


# Setup LangGraph components with improved parameters
def setup_langgraph_components(model_name: str = "gemini-1.5-pro"):
    """Set up LangGraph components with Google Gemini model"""
    global gemini_llm, gemini_llm_comprehensive, langgraph_agent, langgraph_comprehensive_agent

    try:
        logger.info(f"Setting up LangGraph with Google GenAI model '{model_name}'")

        # Check API key
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set in config.py")
            return None, None

        logger.info("GOOGLE_API_KEY is set (basic check)")

        # Set up callbacks for verbose output
        callbacks = [StreamingStdOutCallbackHandler()] if debug_mode else []

        # Initialize the standard LLM (used for thinking phase and standard responses)
        gemini_llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            model=model_name,
            callbacks=callbacks,
            max_output_tokens=4096 # Default max tokens
        )

        # Initialize the comprehensive LLM (used for final comprehensive response)
        # It has a different system instruction and potentially higher token limits
        gemini_llm_comprehensive = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            temperature=0.8,  # Slightly higher temperature for more creative responses
            model=model_name,
            callbacks=callbacks,
            max_output_tokens=8192,  # Higher token limit for longer responses
            top_p=0.95,  # Added for more diverse text generation
            top_k=40,     # Added to increase response variety
            # System instruction is now passed via HumanMessage in enhance_prompt_with_thinking
            # Or rely on the model's default comprehensive abilities if no specific system instruction is set here
            # Let's keep the comprehensive system instruction concept but apply it via prompt/thinking
            # Or, we can use the system_instruction here for the *final* step of the comprehensive agent
            system_instruction=COMPREHENSIVE_SYSTEM_INSTRUCTION
        )

        logger.info(f"ChatGoogleGenerativeAI initialized with models '{model_name}'")

        # Test LLMs directly
        try:
            logger.info("Testing LLMs directly")
            test_message = [HumanMessage(content="Hello, this is a test.")]
            llm_result_std = gemini_llm.invoke(test_message)
            llm_result_comp = gemini_llm_comprehensive.invoke(test_message)
            logger.info("LLM tests successful")
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
                # Don't return None here, agents can still work without search if tools are not used
                # return gemini_llm, None # Keep agent setup attempt
            else:
                 logger.info("Google Search API test successful")
        except Exception as search_error:
            logger.error(f"Google Search API test failed: {search_error}")
            # Don't return None here, agents can still work without search if tools are not used
            # return gemini_llm, None # Keep agent setup attempt


        # Create the tools list
        tools = [
            google_search,
            youtube_search,
            youtube_video_info,
            youtube_oembed,
            youtube_download,
            youtube_save_to_history,
            youtube_create_html_player
        ]

        # Create the LangGraph standard agent
        try:
            logger.info("Creating standard LangGraph agent")
            # The standard agent uses the standard LLM
            agent = create_react_agent(gemini_llm, tools)
            logger.info("Standard LangGraph agent created successfully")
            langgraph_agent = agent

            # Create the comprehensive LangGraph agent
            # The comprehensive agent uses the comprehensive LLM
            logger.info("Creating comprehensive LangGraph agent")
            comprehensive_agent = create_react_agent(gemini_llm_comprehensive, tools)
            logger.info("Comprehensive LangGraph agent created successfully")
            langgraph_comprehensive_agent = comprehensive_agent

            return gemini_llm, langgraph_agent # Return standard LLM and agent as primary
        except Exception as agent_error:
            logger.error(f"Error creating LangGraph agent: {agent_error}")
            return gemini_llm, None # Return standard LLM even if agent creation fails

    except Exception as e:
        logger.error(f"Error in LangGraph setup: {str(e)}")
        return None, None


async def generate_langgraph_response(
    prompt: str,
    model: str = "gemini-1.5-pro",
    comprehensive: bool = True
) -> Tuple[str, str]:
    """
    Generate a response using LangGraph with Google Gemini.
    Supports comprehensive mode with a thinking step (skipped for large inputs).

    Args:
        prompt: The user's input prompt
        model: The Google Gemini model to use
        comprehensive: Whether to use comprehensive mode (includes thinking for non-large inputs)

    Returns:
        Tuple[str, str]: (generated content, model used)
    """
    global langgraph_agent, langgraph_comprehensive_agent, gemini_llm

    # Determine if input is large
    input_size = len(prompt)
    is_large_input = input_size > LARGE_INPUT_THRESHOLD

    # Determine if thinking process should be used
    use_thinking = comprehensive and not is_large_input

    # Select the appropriate agent based on comprehensive flag
    agent_to_use = langgraph_comprehensive_agent if comprehensive and langgraph_comprehensive_agent else langgraph_agent
    llm_for_thinking = gemini_llm # Use the standard LLM for the thinking step

    if agent_to_use is None or (use_thinking and llm_for_thinking is None):
        logger.warning("LangGraph agent or LLM for thinking is not available, setting up now...")
        # Re-setup components. Note: this might re-initialize everything.
        # In a real application, you might want a more robust initialization check.
        _, _ = setup_langgraph_components(model)
        agent_to_use = langgraph_comprehensive_agent if comprehensive and langgraph_comprehensive_agent else langgraph_agent
        llm_for_thinking = gemini_llm

        if agent_to_use is None:
            return "Sorry, the LangGraph agent couldn't be initialized. Please try again later.", f"Error-{model}"
        if use_thinking and llm_for_thinking is None:
             # This case is unlikely if agent_to_use is available, but check anyway
             logger.error("LLM for thinking could not be initialized.")
             use_thinking = False # Fallback to no thinking if LLM isn't there
             agent_to_use = langgraph_agent # Ensure standard agent is used if comprehensive LLM failed

    try:
        debug_print(f"Received prompt (size: {input_size}): '{prompt[:100]}...'", "INFO")

        thinking_result = ""
        if use_thinking:
            debug_print("Starting comprehensive thinking process...", "THINKING")
            # Step 1: Have the model think about the response
            thinking_result = await generate_thinking_response(prompt, llm_for_thinking)

            if thinking_result:
                debug_print(f"Thinking complete: {thinking_result[:100]}...", "THINKING")
                # Step 2: Enhance the prompt with the thinking results
                actual_prompt = enhance_prompt_with_thinking(prompt, thinking_result)
                debug_print(f"Enhanced prompt with thinking", "ACTION")
            else:
                # If thinking failed, fall back to the original prompt
                actual_prompt = prompt
                use_thinking = False # Mark thinking as not used
                debug_print("Thinking process failed, using original prompt.", "WARNING")
        else:
            actual_prompt = prompt
            if is_large_input:
                 debug_print("Large input detected, skipping comprehensive thinking.", "INFO")
            elif comprehensive:
                 debug_print("Comprehensive mode requested but thinking skipped (e.g., fallback or specific logic).", "INFO")
            else:
                 debug_print("Standard mode, skipping thinking.", "INFO")


        debug_print("Invoking LangGraph agent...", "ACTION")

        # LangGraph uses a different invocation pattern
        messages = [HumanMessage(content=actual_prompt)]

        # Using callbacks to track the execution
        # Note: StreamingStdOutCallbackHandler might interfere with API responses if not handled carefully
        # For simple console debugging, it's fine.
        result = agent_to_use.invoke(
            {"messages": messages},
            config={"callbacks": [StreamingStdOutCallbackHandler()]} if debug_mode else {}
        )

        # Extract the AI's response from the result
        final_response = "I couldn't generate a proper response."
        if "messages" in result and len(result["messages"]) > 0:
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                final_response = ai_messages[-1].content
                debug_print(f"Final response generated (approx. {len(final_response)} chars)", "RESULT")

        # Add mode suffix for tracking
        mode_suffix = ""
        if use_thinking:
            mode_suffix = "-Comprehensive"
        elif is_large_input:
            mode_suffix = "-LargeInput"
        # Else: no suffix for standard mode

        # Check for embedded video markers (keeping the existing functionality)
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

        # If CHAIN_OF_THOUGHT_VISIBLE is true AND thinking was used, append the thinking result
        # Note: This is for debugging/demonstration, not typically part of a final user response
        if CHAIN_OF_THOUGHT_VISIBLE and use_thinking and thinking_result:
             final_response += f"\n\n--- My Thinking Process ---\n\n{thinking_result}"


        return final_response, f"{model}{mode_suffix}"
    except Exception as e:
        logger.error(f"Error using LangGraph agent: {e}")
        # Fallback response
        return f"I encountered an error while processing your request: {str(e)}", f"Error-{model}"


async def test_langgraph_agent(query: str, comprehensive: bool = True) -> Dict[str, Any]:
    """Test LangGraph functionality with comprehensive mode option"""
    global langgraph_agent, langgraph_comprehensive_agent, gemini_llm

    # Determine if input is large
    input_size = len(query)
    is_large_input = input_size > LARGE_INPUT_THRESHOLD

    # Determine if thinking process should be used
    use_thinking = comprehensive and not is_large_input

    # Select the appropriate agent based on comprehensive flag
    agent_to_use = langgraph_comprehensive_agent if comprehensive and langgraph_comprehensive_agent else langgraph_agent
    llm_for_thinking = gemini_llm # Use the standard LLM for the thinking step


    if agent_to_use is None or (use_thinking and llm_for_thinking is None):
         # Attempt setup if not available
        _, _ = setup_langgraph_components()
        agent_to_use = langgraph_comprehensive_agent if comprehensive and langgraph_comprehensive_agent else langgraph_agent
        llm_for_thinking = gemini_llm
        if agent_to_use is None:
            return {
                "success": False,
                "method": "langgraph_test",
                "query": query,
                "error": "LangGraph agent is not available after setup attempt."
            }
        if use_thinking and llm_for_thinking is None:
             logger.error("LLM for thinking could not be initialized for test.")
             use_thinking = False


    try:
        mode_str = "comprehensive" if comprehensive else "standard"
        debug_print(f"Testing LangGraph with query '{query[:100]}...' (size: {input_size}) in {mode_str} mode", "INFO")

        thinking_result = ""
        if use_thinking:
            debug_print("Starting comprehensive thinking process for test...", "THINKING")
            thinking_result = await generate_thinking_response(query, llm_for_thinking)
            if thinking_result:
                debug_print(f"Thinking complete for test: {thinking_result[:100]}...", "THINKING")
                actual_query = enhance_prompt_with_thinking(query, thinking_result)
                debug_print(f"Enhanced query for test", "ACTION")
            else:
                 actual_query = query
                 use_thinking = False # Mark thinking as not used
                 debug_print("Thinking process failed for test, using original query.", "WARNING")
        else:
            actual_query = query
            if is_large_input:
                 debug_print("Large input detected for test, skipping comprehensive thinking.", "INFO")
            elif comprehensive:
                 debug_print("Comprehensive mode requested for test but thinking skipped.", "INFO")
            else:
                 debug_print("Standard mode for test, skipping thinking.", "INFO")


        messages = [HumanMessage(content=actual_query)]

        result = agent_to_use.invoke(
            {"messages": messages},
            config={"callbacks": [StreamingStdOutCallbackHandler()]} if debug_mode else {}
        )

        agent_result = "Could not extract response from agent."
        if "messages" in result and len(result["messages"]) > 0:
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                agent_result = ai_messages[-1].content

        # If comprehensive and thinking is visible, include the thinking
        if CHAIN_OF_THOUGHT_VISIBLE and use_thinking and thinking_result:
            agent_result += (
                f"\n\n--- My Thinking Process ---\n\n"
                f"{thinking_result}"
            )

        # Add mode suffix for tracking
        mode_suffix = ""
        if use_thinking:
            mode_suffix = "-Comprehensive"
        elif is_large_input:
            mode_suffix = "-LargeInput"


        return {
            "success": True,
            "method": f"langgraph_agent_{mode_str}{mode_suffix}",
            "query": query,
            "result": agent_result
        }

    except Exception as agent_error:
        logger.error(f"LangGraph agent test error: {agent_error}")
        return {
            "success": False,
            "method": f"langgraph_agent_{mode_str}",
            "query": query,
            "error": str(agent_error)
        }

# Removed expand_langgraph_response as it doesn't align with the comprehensive mode logic implemented.

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
