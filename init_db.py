import sqlite3

conn = sqlite3.connect("events.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    tickets_sold INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL
)
""")

sample_events = [
    ("evt_101", "Conference", 520, 1),
    ("evt_102", "Concert", 850, 1),
    ("evt_103", "Workshop", 120, 0),
    ("Race 5657", "Concert", 1200, 1)
]

cursor.executemany("INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?)", sample_events)
conn.commit()
conn.close()
print("Database 'events.db' initialized with sample records.")
