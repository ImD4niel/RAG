"""
reset_database.py 

A learning tool and helper script to completely wipe the PostgreSQL database 
schema if it ever gets corrupted or stuck on old code.

USE WITH CAUTION: This deletes all tables and data!
"""

from rag.db import _get_engine
from sqlalchemy import text

def reset_db():
    print("\n⚠️ WARNING: This will permanently drop all tables and data in your PostgreSQL 'ragdb'.")
    confirm = input("Are you absolutely sure you want to proceed? (y/n): ")
    if confirm.lower() == 'y':
        print("Dropping public schema...")
        engine, _ = _get_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text('DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO public;'))
                conn.commit()
            print("✅ Database wiped completely!")
            print("\nNext step: Run `alembic upgrade head` in your terminal to rebuild the empty tables.")
        except Exception as e:
            print(f"❌ Failed to reset database: {e}")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    reset_db()
