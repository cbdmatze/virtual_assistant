import requests
import logging
from typing import Tuple, Dict, Any

from config import RAPIDAPI_CONFIG


logger = logging.getLogger(__name__)


async def generate_image_from_prompt(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 1
) -> Tuple[str, dict[str, Any]]:
    """
    Generate an image from a text prompt using RapidAPI
    
    Args:
        prompt: Text description of the desired image
        width: Image width in pixels
        height: Image height in pixels
        steps: Number of diffusion steps (higher = potentially better quality)
       
    Returns:
        Tuple[str, dict[str, Any]]: (image data as base64 string, status info)"
    """
    url = RAPIDAPI_CONFIG["url"]
    
    payload = {
        "text": prompt,
        "width": width,
        "height": height,
        "steps": steps
    }

    headers = {
        "x-rapidapi-key": RAPIDAPI_CONFIG["key"],
        "x-rapidapi-host": RAPIDAPI_CONFIG["host"],
        "Content-Type": "application/json"
    }
    
    logger.info(f"Sending image generation request: {prompt[:50]}...")

    try:
        api_response = requests.post(url, json=payload, headers=headers)
        api_response.raise_for_status()

        response_data = api_response.json()
        logger.debug(f"RapidAPI reesponse keys: {response_data.keys()}")

        if "generated_image" in response_data:
            image_data = response_data["generated_image"]
            logger.info("Image generated successfully")
            return image_data, {"success": True}
        else:
            error_msg = response_data.get("error", "Unknown error")
            logger.error(f"Error generating image: {error_msg}")
            return "", {"success": False, "error": error_msg}
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during image generation: {e.response.status_code} - {e.response.text}")
        return "", {"success": False, "error": f"HTTP error: {e.response.status_code}"}
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during image generation: {str(e)}")
        return "", {"success": False, "error": "Connection error"}
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return "", {"success": False, "error": {str(e)}}
