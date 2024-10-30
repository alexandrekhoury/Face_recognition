import sqlite3
import os

os.makedirs("client_data", exist_ok=True)

# Paths for directories
embedding_dir = os.path.join("client_data", "embeddings")
image_dir = os.path.join("client_data", "images")

# Create directories if they don't exist
os.makedirs(embedding_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Connect to the database (or create it if it doesn’t exist)
conn = sqlite3.connect('clients.db')
cursor = conn.cursor()

# Create the clients table
cursor.execute('''
CREATE TABLE IF NOT EXISTS clients (
    client_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    image_path TEXT NOT NULL,
    embedding_path TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print("Initialization complete: Database and directories have been created.")
