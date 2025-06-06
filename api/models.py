from pydantic import BaseModel
from typing import Optional


class User(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    prompt: str
    model: str
    temperature: float = 0.7
    api_provider: str = "openai" # Match the field name used in the frontend


class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 1


class YouTubeRequest(BaseModel):
    """YouTube video player request model"""
    video_id: str  # YouTube video ID
    width: Optional[int] = None  # Optional width for the player
    height: Optional[int] = None  # Optional height for the player
