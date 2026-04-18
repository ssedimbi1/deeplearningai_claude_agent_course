"""
Tests for the FastAPI HTTP layer (routes defined in app.py).

A minimal test app mirrors app.py's route logic without mounting static files,
so these tests run offline and never touch ChromaDB or the Anthropic API.
All RAGSystem behaviour is controlled via the mock_rag_system fixture in conftest.py.
"""
import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_returns_200_with_valid_request(self, test_client):
        response = test_client.post("/api/query", json={"query": "What is RAG?"})
        assert response.status_code == 200

    def test_response_contains_required_fields(self, test_client):
        response = test_client.post("/api/query", json={"query": "What is RAG?"})
        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert "session_id" in body

    def test_answer_matches_rag_system_output(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("Detailed explanation of RAG.", [])
        response = test_client.post("/api/query", json={"query": "Explain RAG"})
        assert response.json()["answer"] == "Detailed explanation of RAG."

    def test_sources_propagated_from_rag_system(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Answer.",
            ["Course A - Lesson 1::https://example.com/l1"],
        )
        response = test_client.post("/api/query", json={"query": "course question"})
        assert response.json()["sources"] == ["Course A - Lesson 1::https://example.com/l1"]

    def test_auto_creates_session_when_none_provided(self, test_client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "new-session-99"
        response = test_client.post("/api/query", json={"query": "Hello"})
        assert response.json()["session_id"] == "new-session-99"

    def test_uses_provided_session_id_without_creating_new_one(self, test_client, mock_rag_system):
        response = test_client.post(
            "/api/query", json={"query": "Follow-up", "session_id": "existing-session"}
        )
        mock_rag_system.session_manager.create_session.assert_not_called()
        assert response.json()["session_id"] == "existing-session"

    def test_passes_session_id_to_rag_query(self, test_client, mock_rag_system):
        test_client.post("/api/query", json={"query": "Hello", "session_id": "sess-42"})
        mock_rag_system.query.assert_called_once_with("Hello", "sess-42")

    def test_returns_500_when_rag_system_raises(self, test_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        response = test_client.post("/api/query", json={"query": "Anything"})
        assert response.status_code == 500

    def test_500_detail_contains_exception_message(self, test_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB unavailable")
        response = test_client.post("/api/query", json={"query": "Anything"})
        assert "DB unavailable" in response.json()["detail"]

    def test_missing_query_field_returns_422(self, test_client):
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422

    def test_empty_sources_list_is_valid(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("General answer.", [])
        response = test_client.post("/api/query", json={"query": "General question"})
        assert response.json()["sources"] == []


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_returns_200(self, test_client):
        response = test_client.get("/api/courses")
        assert response.status_code == 200

    def test_response_contains_required_fields(self, test_client):
        body = test_client.get("/api/courses").json()
        assert "total_courses" in body
        assert "course_titles" in body

    def test_total_courses_matches_analytics(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        body = test_client.get("/api/courses").json()
        assert body["total_courses"] == 5

    def test_course_titles_matches_analytics(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Intro to Python", "Advanced ML"],
        }
        body = test_client.get("/api/courses").json()
        assert body["course_titles"] == ["Intro to Python", "Advanced ML"]

    def test_returns_500_when_analytics_raises(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = Exception("chroma error")
        response = test_client.get("/api/courses")
        assert response.status_code == 500

    def test_500_detail_contains_exception_message(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = Exception("chroma error")
        response = test_client.get("/api/courses")
        assert "chroma error" in response.json()["detail"]

    def test_empty_course_list_is_valid(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        body = test_client.get("/api/courses").json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:

    def test_returns_200(self, test_client):
        response = test_client.delete("/api/session/sess-123")
        assert response.status_code == 200

    def test_response_body_is_ok_status(self, test_client):
        response = test_client.delete("/api/session/sess-123")
        assert response.json() == {"status": "ok"}

    def test_delegates_to_session_manager(self, test_client, mock_rag_system):
        test_client.delete("/api/session/sess-abc")
        mock_rag_system.session_manager.delete_session.assert_called_once_with("sess-abc")

    def test_session_id_passed_verbatim(self, test_client, mock_rag_system):
        test_client.delete("/api/session/my-special-session")
        args = mock_rag_system.session_manager.delete_session.call_args[0]
        assert args[0] == "my-special-session"
