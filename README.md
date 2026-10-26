# DataFlow

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn main:app --reload

Open http://localhost:8000

Upload CSV / JSON / Excel / Word / PDF / TXT. Tables go to SQLite; text is chunked for RAG.
Build charts, save or delete them, and ask questions (text-to-SQL + RAG).

Upgrade path: swap TF-IDF in `retrieve()` for embeddings + pgvector, SQLite for PostgreSQL, and add a user_id column for multi-user.
