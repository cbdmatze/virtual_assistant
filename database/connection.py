import mysql.connector
from mysql.connector import pooling
import logging
from typing import Optional

from config import DB_CONFIG
from database.models import TABLES, TABLE_COLUMNS

logger = logging.getLogger(__name__)

# Global connection pool
connection_pool = None

def init_database():
    """Initialize the database connection and create tables if they don't exist"""
    global connection_pool

    try:
        # Create connection pool
        connection_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="bullsai_pool",
            pool_size=5,
            **DB_CONFIG
        )
        
        logger.info("Database connection pool established")
        
        # Get a connection from the pool
        conn = get_connection()
        if conn:
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            for table_query in TABLES:
                cursor.execute(table_query)
            
            # Check for and add any missing columns
            for table, columns in TABLE_COLUMNS.items():
                for column in columns:
                    try:
                        cursor.execute(f"SHOW COLUMNS FROM {table} LIKE '{column['name']}'")
                        if not cursor.fetchone():
                            cursor.execute(column["query"])
                            logger.info(f"Added missing column '{column['name']}' to table '{table}'")
                    except Exception as e:
                        logger.error(f"Error checking or adding column '{column['name']}' to table '{table}': {e}")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            
            # Check if users table is empty and create default user if needed
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count == 0:
                from werkzeug.security import generate_password_hash
                hashed_password = generate_password_hash("admin")
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    ("admin", hashed_password)
                )
                conn.commit()
                logger.info("Created default user: username='admin', password='admin'")
            
            cursor.close()
            release_connection(conn)
            
    except mysql.connector.Error as e:
        logger.error(f"Error initializing database: {e}")
        
        # Create dummy connection pool for safer operation if DB connection fails
        class DummyPool:
            def get_connection(self):
                class DummyConnection:
                    def cursor(self):
                        class DummyCursor:
                            def execute(self, *args, **kwargs): pass
                            def fetchone(self): return None
                            def fetchall(self): return []
                            def close(self): pass
                        return DummyCursor()
                    def commit(self): pass
                    def close(self): pass
                    def is_connected(self): return False
                return DummyConnection()
        
        connection_pool = DummyPool()
        logger.warning("Using dummy database connection due to initialization error")

def get_connection():
    """Get a connection from the pool"""
    global connection_pool
    
    try:
        if connection_pool:
            return connection_pool.get_connection()
        else:
            logger.error("Connection pool not initialized")
            return None
    except Exception as e:
        logger.error(f"Error getting database connection: {e}")
        return None

def release_connection(conn):
    """Release a connection back to the pool"""
    if conn:
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error releasing database connection: {e}")

def check_database_connection() -> bool:
    """Check if the database is connected and working"""
    conn = get_connection()
    if conn:
        is_connected = conn.is_connected()
        release_connection(conn)
        return is_connected
    return False
