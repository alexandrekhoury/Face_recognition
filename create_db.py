import sqlite3

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

