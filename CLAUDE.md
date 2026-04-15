# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Requires Python 3.13+ and `uv`. Always use `uv` — never `pip`. On Windows, use Git Bash (not PowerShell/CMD).

```bash
uv sync                  # install dependencies
./run.sh                 # start the server (from project root)
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

The app serves at `http://localhost:8000`. FastAPI serves both the REST API and the static frontend — no separate frontend build step.

## Architecture

**Single-process**: FastAPI (`backend/app.py`) handles API routes and serves `frontend/` as static files from the same port.

**Query flow:**
1. Frontend POSTs `{query, session_id}` to `POST /api/query`
2. `RAGSystem.query()` fetches session history, then calls `AIGenerator.generate_response()` with the query, history, and tool definitions
3. **First Claude API call**: Claude either answers directly (general knowledge) or invokes the `search_course_content` tool
4. If tool use: `CourseSearchTool` → `VectorStore.search()` → ChromaDB returns ranked `CourseChunk`s
5. **Second Claude API call** (only when tool was used): Claude synthesizes retrieved chunks into a final answer
6. Sources and answer returned to frontend; exchange saved to session history

**Document ingestion** runs at startup: `.txt/.pdf/.docx` files in `/docs` are parsed by `DocumentProcessor` into sentence-based overlapping chunks and stored in ChromaDB. Courses are deduplicated by title.

**ChromaDB has two collections:**
- `course_catalog` — one entry per course (used for fuzzy course name resolution)
- `course_content` — one entry per `CourseChunk` (the semantically searchable text)

When Claude passes a `course_name` to the search tool, it is first resolved via semantic search against `course_catalog`, then used as a metadata filter on `course_content`. This makes partial/fuzzy course name matching work.

## Configuration (`backend/config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Generation model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `CHUNK_SIZE` | 800 | Max characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | ChromaDB hits per search |
| `MAX_HISTORY` | 2 | Conversation exchanges kept per session |
| `CHROMA_PATH` | `./chroma_db` | Vector DB path (relative to `backend/`) |

## Course Document Format

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<transcript text>

Lesson 1: <title>
...
```

Course title is the unique ID in ChromaDB — re-ingesting a file with the same title is a no-op.

## Adding New Tools

1. Subclass `Tool` in `backend/search_tools.py`, implementing `get_tool_definition()` (Anthropic tool schema) and `execute()`
2. Register it in `RAGSystem.__init__()` via `self.tool_manager.register_tool(your_tool)`

`ToolManager` handles dispatch automatically. If your tool tracks sources for the UI, add a `last_sources` list attribute — `ToolManager.get_last_sources()` and `reset_sources()` will pick it up automatically.