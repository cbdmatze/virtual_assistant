import logging
from typing import List, Tuple, Any, Optional, Dict

from database.connection import get_connection, release_connection


logger = logging.getLogger(__name__)


def execute_query(query: str, params: Optional[Tuple] = None, return_last_id: bool = False) -> Any:
    """
    Execute a SQL query with optional parameters
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters in the query
        return_last_id: Whether to return the last inserted ID
        
    Returns:
        bool/int: True/last_id if successful, False/None otherwise
    """
    conn = get_connection()
    if not conn:
        logger.error("Failed to get database connection")
        return False if not return_last_id else None
   
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        conn.commit()
        
        # Return last inserted ID if requested
        if return_last_id:
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        else:
            cursor.close()
            return True
    except Exception as e:
        logger.error(f"Database error executing query: {e}")
        return False if not return_last_id else None
    finally:
        release_connection(conn)


def fetch_one(query: str, params: Optional[Tuple] = None) -> Optional[Tuple]:
    """
    Execute a SQL query and fetch one result
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters in the query
        
    Returns:
        Optional[Tuple]: The fetched result or None
    """
    conn = get_connection()
    if not conn:
        logger.error("Failed to get database connection")
        return None
    
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        result = cursor.fetchone()
        cursor.close()
        return result
    except Exception as e:
        logger.error(f"Database error fetching one result: {e}")
        return None
    finally:
        release_connection(conn)


def fetch_all(query: str, params: Optional[Tuple] = None) -> List[Tuple]:
    """
    Execute a SQL query and fetch all results
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters in the query
        
    Returns:
        List[Tuple]: The fetched results
    """
    conn = get_connection()
    if not conn:
        logger.error("Failed to get database connection")
        return []
    
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        logger.error(f"Database error fetching all results: {e}")
        return []
    finally:
        release_connection(conn)


def insert_and_get_id(query: str, params: Tuple) -> Optional[int]:
    """
    Execute an INSERT query and return the last inserted ID
    
    Args:
        query: SQL query string
        params: Tuple of parameters in the query
        
    Returns:
        Optional[int]: The last inserted ID or None on error
    """
    conn = get_connection()
    if not conn:
        logger.error("Failed to get database connection")
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

        last_id = cursor.lastrowid
        cursor.close()
        return last_id
    except Exception as e:
        logger.error(f"Database error inserting and getting ID: {e}")
        return None
    finally:
        release_connection(conn)


def save_video_to_db(user_id: int, video_data: Dict[str, Any]) -> Optional[int]:
    """
    Save video data to the database
    
    Args:
        user_id: The user ID
        video_data: Dictionary with video information
        
    Returns:
        Optional[int]: The video ID or None on error
    """
    try:
        video_type = video_data.get("type", "reference")
        
        if video_type == "downloaded":
            query = """
                INSERT INTO videos 
                (user_id, video_id, title, channel, type, filepath, thumbnail_url) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id, 
                video_data.get("video_id"), 
                video_data.get("title", "Unknown title"), 
                video_data.get("channel", "Unknown channel"),
                "downloaded", 
                video_data.get("filepath"), 
                video_data.get("thumbnail")
            )
        elif video_type == "embedded":
            query = """
                INSERT INTO videos 
                (user_id, video_id, title, channel, type, embed_html, thumbnail_url) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id, 
                video_data.get("video_id"), 
                video_data.get("title", "YouTube Video"), 
                video_data.get("channel", "Unknown channel"), 
                "embedded", 
                video_data.get("embed_html"), 
                video_data.get("thumbnail")
            )
        else:  # reference
            query = """
                INSERT INTO videos 
                (user_id, video_id, title, channel, type, thumbnail_url) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id, 
                video_data.get("video_id"), 
                video_data.get("title", "YouTube Video"), 
                video_data.get("channel", "Unknown channel"), 
                "reference", 
                video_data.get("thumbnail")
            )
            
        return execute_query(query, params, return_last_id=True)
    except Exception as e:
        logger.error(f"Error saving video to database: {e}")
        return None


def get_video_by_id(video_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get video data from the database by ID
    
    Args:
        video_id: The video record ID
        user_id: The user ID for permission check
        
    Returns:
        Optional[Dict]: Video data or None
    """
    query = """
        SELECT id, video_id, title, channel, type, filepath, thumbnail_url, embed_html, created_at
        FROM videos 
        WHERE id = %s AND user_id = %s
    """
    
    result = fetch_one(query, (video_id, user_id))
    if not result:
        return None
    
    return {
        "db_id": result[0],
        "video_id": result[1],
        "title": result[2],
        "channel": result[3],
        "type": result[4],
        "filepath": result[5],
        "thumbnail": result[6],
        "embed_html": result[7],
        "created_at": result[8]
    }


def get_videos_by_user(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Get all videos for a user
    
    Args:
        user_id: The user ID
        limit: Maximum number of videos to return
        offset: Number of videos to skip
        
    Returns:
        List[Dict]: List of video data
    """
    query = """
        SELECT id, video_id, title, channel, type, filepath, thumbnail_url, embed_html, created_at
        FROM videos 
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """
    
    results = fetch_all(query, (user_id, limit, offset))
    
    videos = []
    for result in results:
        videos.append({
            "db_id": result[0],
            "video_id": result[1],
            "title": result[2],
            "channel": result[3],
            "type": result[4],
            "filepath": result[5],
            "thumbnail": result[6],
            "embed_html": result[7],
            "created_at": result[8]
        })
        
    return videos


def link_video_to_conversation(conversation_id: int, video_db_id: int) -> bool:
    """
    Link a video to an existing conversation
    
    Args:
        conversation_id: The conversation ID
        video_db_id: The video database ID
        
    Returns:
        bool: True if successful
    """
    query = "UPDATE conversations SET video_id = %s WHERE id = %s"
    return execute_query(query, (video_db_id, conversation_id))


def get_conversations_with_videos(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Get conversations with associated videos
    
    Args:
        user_id: The user ID
        limit: Maximum number of conversations
        offset: Number of conversations to skip
        
    Returns:
        List[Dict]: List of conversations with video data
    """
    query = """
        SELECT c.id, c.conversation, c.model, c.temperature, c.timestamp, c.api_provider,
               v.id, v.video_id, v.title, v.type, v.thumbnail_url, v.filepath, v.embed_html
        FROM conversations c
        LEFT JOIN videos v ON c.video_id = v.id
        WHERE c.user_id = %s
        ORDER BY c.timestamp DESC
        LIMIT %s OFFSET %s
    """
    
    results = fetch_all(query, (user_id, limit, offset))
    
    conversations = []
    for result in results:
        conv = {
            "id": result[0],
            "conversation": result[1],
            "model": result[2],
            "temperature": result[3],
            "timestamp": result[4],
            "api_provider": result[5],
            "has_video": result[6] is not None
        }
        
        # Add video data if present
        if result[6]:  # If video ID exists
            conv["video_db_id"] = result[6]
            conv["video_id"] = result[7]
            conv["video_title"] = result[8]
            conv["video_type"] = result[9]
            conv["video_thumbnail"] = result[10]
            conv["video_filepath"] = result[11]
            conv["video_embed_html"] = result[12]
            
        conversations.append(conv)
        
    return conversations
