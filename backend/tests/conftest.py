import sys
import os

# Add the backend directory to sys.path so test files can import backend modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.testclient import TestClient


@pytest.fixture
def mock_rag_system():
    """RAGSystem stand-in with sensible defaults for all methods used by the API layer."""
    mock = MagicMock()
    mock.query.return_value = ("Test answer.", [])
    mock.session_manager.create_session.return_value = "auto-session-id"
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return mock


@pytest.fixture
def test_client(mock_rag_system):
    """
    TestClient for a minimal FastAPI app that mirrors app.py's routes without
    mounting static files (which don't exist in the test environment).
    """

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag_system.session_manager.delete_session(session_id)
        return {"status": "ok"}

    return TestClient(app)
