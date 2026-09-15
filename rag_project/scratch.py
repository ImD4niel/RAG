from rag.db import get_session, Document
import sys

sys.stdout.reconfigure(encoding='utf-8')

with get_session() as session:
    docs = session.query(Document).limit(5).all()
    for d in docs:
        print(f"--- DOC ID: {d.id} ---")
        print(d.content[:500])
