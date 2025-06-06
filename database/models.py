"""
This module defines the database schema for the application.

Since we're using raw SQL queries rather than an ORM, these are not actual model classes,
but rather definitions that can be used to create the database tables.
"""

# Users table definition
USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
)
"""

# Conversations table definition
CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    conversation TEXT,
    model VARCHAR(255),
    temperature FLOAT,
    api_provider VARCHAR(50) DEFAULT 'openai',
    image_data LONGTEXT,
    video_id INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

# Videos table definition
VIDEOS_TABLE = """
CREATE TABLE IF NOT EXISTS videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    video_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    channel VARCHAR(255),
    type ENUM('embedded', 'downloaded', 'reference') NOT NULL,
    filepath VARCHAR(255),
    thumbnail_url VARCHAR(512),
    embed_html TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

# List of all table creation statements
TABLES = [USERS_TABLE, CONVERSATIONS_TABLE, VIDEOS_TABLE]

# Table column checks
TABLE_COLUMNS = {
    "conversations": [
        {"name": "temperature", "query": "ALTER TABLE conversations ADD COLUMN temperature FLOAT"},
        {"name": "model", "query": "ALTER TABLE conversations ADD COLUMN model VARCHAR(255)"},
        {"name": "image_data", "query": "ALTER TABLE conversations ADD COLUMN image_data LONGTEXT"},
        {"name": "api_provider", "query": "ALTER TABLE conversations ADD COLUMN api_provider VARCHAR(50) DEFAULT 'openai'"},
        {"name": "video_id", "query": "ALTER TABLE conversations ADD COLUMN video_id INT"}
    ],
    "videos": [
        # No column checks needed as we're creating the table from scratch
    ]
}
