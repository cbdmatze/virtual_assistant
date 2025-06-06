from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import mysql.connector

from config import DB_CONFIG
from database.crud import execute_query, fetch_one
from api.models import User


router = APIRouter(tags=["authentication"], prefix="")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = logging.getLogger(__name__)


def authenticate_user(username: str, password: str):
    """Authenticate a user by username and password"""
    try:
        logger.info(f"Authenticating user: {username}")
        # Fixed: Removed extra comma after 'password'
        result = fetch_one("SELECT id, password FROM users WHERE username = %s", (username,))

        if not result:
            logger.warning(f"Authentication failed: User '{username}' not found")
            return None
        
        user_id, hashed_password = result

        if check_password_hash(hashed_password, password):
            logger.info(f"Authencation successful for user: {username}")
            return user_id
        else:
            logger.warning(f"Authentication failed: Incorrect password for user '{username}'")
            return None
    except Exception as e:
        logger.error(f"error during authentication: {e}")
        return None


@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to obtain an OAuth2 Bearer token"""
    logger.info(f"Login attempt with username: {form_data.username}")
    user_id = authenticate_user(form_data.username, form_data.password)
    if not user_id:
        logger.warning(f"Login failed for user: {form_data.username}")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    logger.info(f"Login successful for username: {form_data.username}")
    return {"access_token": user_id, "token_type": "bearer"}


@router.post("/register")
async def register(user: User):
    """Register a new user"""
    try:
        hashed_password = generate_password_hash(user.password)
        execute_query("INSERT INTO users (username, password) VALUES (%s, %s)",
                      (user.username, hashed_password))
        logger.info(f"Successfully registered user: {user.username}")
        return {"message": "Registration successful"}
    except mysql.connector.IntegrityError:
        logger.warning(f"Registration failed: Username '{user.username}' already exists")
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")
