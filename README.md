# DRAGON RAG

DRAGON RAG is a production-grade Agentic Retrieval-Augmented Generation system. This project is built using Python, and uses PostgreSQL with `pgvector` for scalable, persistent vector storage.

## Prerequisites
- **Python 3.10+**
- **PostgreSQL Database** running with the **`pgvector`** extension installed.

## Setup Guide

### 1. Database Setup
Ensure PostgreSQL is running on your machine. You need to create an empty database (the default expected database name is `ragdb`, with user `postgres`).

```sql
CREATE DATABASE ragdb;
```

*Note: The `pgvector` extension must be installed in your PostgreSQL environment.*

### 2. Environment Configuration
Navigate to the `rag_project` directory and copy the environment template:
```bash
cd rag_project
cp .env.example .env
```
Fill out the required configuration variables in the `.env` file, especially your database credentials and API keys (e.g., OpenAI, Gemini, or Github).

### 3. Install Dependencies
It is highly recommended to use a virtual environment. From the project root (`LEARNING_RAG`), run:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Initialise Database Schema
As this project uses Alembic to manage database migrations, you must apply the initial schema before running the application:

```bash
cd rag_project
alembic upgrade head
```

### 5. Add Documents
Create a `data/` directory at the root of the project and place any PDFs, PowerPoint files, or text data you want the RAG system to index inside it. The system's document loaders are configured to automatically scan the `<repo_root>/data/` path by default.

## Running the Application

To start the interactive DRAGON chat interface, run:
```bash
cd rag_project
python main.py
```

### In-App Commands
- `/refresh` : Scans for new or changed files incrementally and updates the vector database.
- `/reset` : Completely wipes the database and forces a full re-index from scratch.
- `exit` : Quits the application.

## Version Control
This repository has its database schema migrations consolidated to a single baseline. If you make modifications to the `rag/db.py` models, you can generate a new migration by running:
```bash
alembic revision --autogenerate -m "describe changes"
alembic upgrade head
```
