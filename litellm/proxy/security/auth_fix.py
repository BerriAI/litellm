# Security fix for CVE-2026-42208 — parameterized queries
import sqlite3
def safe_auth(db, token):
    cursor = db.cursor()
    cursor.execute(
        "SELECT user_id FROM api_keys WHERE token = ?",
        (token,)  # Parameter, not string interpolation
    )
    result = cursor.fetchone()
    return result[0] if result else None
