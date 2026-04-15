"""
Tests for RAGSystem.query() in rag_system.py.

Focus: handling of content-related queries — how the system routes questions
through the AI generator + tool pipeline, manages session state, and surfaces
sources back to the caller.

All heavy dependencies (ChromaDB, sentence-transformers, Anthropic) are mocked
so the tests are fast and offline.
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def rag():
    """
    RAGSystem instance with all external dependencies replaced by MagicMocks.
    The fixture patches at the rag_system module level so that __init__
    receives mock constructors and the resulting instance attributes are
    accessible mocks.
    """
    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore"), \
         patch("rag_system.AIGenerator"), \
         patch("rag_system.SessionManager"), \
         patch("rag_system.ToolManager"), \
         patch("rag_system.CourseSearchTool"), \
         patch("rag_system.CourseOutlineTool"):

        from rag_system import RAGSystem

        cfg = MagicMock()
        cfg.CHUNK_SIZE = 800
        cfg.CHUNK_OVERLAP = 100
        cfg.CHROMA_PATH = "/tmp/test_chroma"
        cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        cfg.MAX_RESULTS = 5
        cfg.ANTHROPIC_API_KEY = "test-key"
        cfg.ANTHROPIC_MODEL = "claude-test"
        cfg.MAX_HISTORY = 2

        system = RAGSystem(cfg)
        yield system


# ---------------------------------------------------------------------------
# Return value structure
# ---------------------------------------------------------------------------

class TestQueryReturnValue:

    def test_query_returns_two_tuple(self, rag):
        """query() returns a (response, sources) tuple."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        result = rag.query("Any question")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_response_matches_ai_generator_output(self, rag):
        """The first element of the tuple is exactly what generate_response returned."""
        rag.ai_generator.generate_response.return_value = "Detailed explanation."
        rag.tool_manager.get_last_sources.return_value = []

        response, _ = rag.query("Explain neural networks")

        assert response == "Detailed explanation."

    def test_query_sources_matches_tool_manager_sources(self, rag):
        """The second element of the tuple is exactly what get_last_sources returned."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = [
            "Course A - Lesson 1::http://example.com/l1"
        ]

        _, sources = rag.query("Content question")

        assert sources == ["Course A - Lesson 1::http://example.com/l1"]

    def test_empty_sources_returned_when_no_tool_used(self, rag):
        """Sources list is empty when the AI answered from general knowledge."""
        rag.ai_generator.generate_response.return_value = "General knowledge answer."
        rag.tool_manager.get_last_sources.return_value = []

        _, sources = rag.query("What is the speed of light?")

        assert sources == []

    def test_multiple_sources_propagated(self, rag):
        """Multiple sources from tool searches are all returned to the caller."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = [
            "Course A - Lesson 1::http://example.com/l1",
            "Course A - Lesson 2::http://example.com/l2",
        ]

        _, sources = rag.query("Content question")

        assert len(sources) == 2


# ---------------------------------------------------------------------------
# Tool pipeline wiring
# ---------------------------------------------------------------------------

class TestToolPipelineWiring:

    def test_query_passes_tool_definitions_to_ai_generator(self, rag):
        """generate_response receives the tool definitions from tool_manager."""
        tool_defs = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        rag.tool_manager.get_tool_definitions.return_value = tool_defs
        rag.ai_generator.generate_response.return_value = "Response."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Course question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["tools"] == tool_defs

    def test_query_passes_tool_manager_to_ai_generator(self, rag):
        """generate_response receives the tool_manager so it can dispatch tool calls."""
        rag.ai_generator.generate_response.return_value = "Response."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["tool_manager"] is rag.tool_manager

    def test_user_question_is_embedded_in_query_prompt(self, rag):
        """The user's raw question appears inside the prompt sent to generate_response."""
        rag.ai_generator.generate_response.return_value = "Response."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("What are transformers?")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "What are transformers?" in kwargs["query"]

    def test_sources_reset_after_retrieval(self, rag):
        """reset_sources() is called after get_last_sources() to avoid stale data."""
        rag.ai_generator.generate_response.return_value = "Response."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("question")

        rag.tool_manager.reset_sources.assert_called_once()

    def test_reset_called_after_get_last_sources(self, rag):
        """reset_sources() is called after — not before — get_last_sources()."""
        call_order = []
        rag.tool_manager.get_last_sources.side_effect = lambda: call_order.append("get") or []
        rag.tool_manager.reset_sources.side_effect = lambda: call_order.append("reset")
        rag.ai_generator.generate_response.return_value = "Response."

        rag.query("question")

        assert call_order == ["get", "reset"]


# ---------------------------------------------------------------------------
# Session / conversation history handling
# ---------------------------------------------------------------------------

class TestSessionHandling:

    def test_session_history_fetched_when_session_id_provided(self, rag):
        """get_conversation_history is called with the provided session_id."""
        rag.session_manager.get_conversation_history.return_value = "Prior context"
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Follow-up", session_id="sess-001")

        rag.session_manager.get_conversation_history.assert_called_once_with("sess-001")

    def test_history_forwarded_to_ai_generator(self, rag):
        """The conversation history retrieved from session_manager is passed to generate_response."""
        rag.session_manager.get_conversation_history.return_value = "User: prev\nAssistant: ans"
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Follow-up", session_id="sess-002")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] == "User: prev\nAssistant: ans"

    def test_no_history_fetched_without_session_id(self, rag):
        """get_conversation_history is NOT called when no session_id is given."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Stateless question")

        rag.session_manager.get_conversation_history.assert_not_called()

    def test_none_history_passed_when_no_session(self, rag):
        """generate_response receives conversation_history=None when there is no session."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Stateless question")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["conversation_history"] is None

    def test_exchange_saved_to_session_after_response(self, rag):
        """add_exchange is called with the original question and the AI's response."""
        rag.ai_generator.generate_response.return_value = "The final answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Important question?", session_id="sess-003")

        rag.session_manager.add_exchange.assert_called_once_with(
            "sess-003", "Important question?", "The final answer."
        )

    def test_exchange_not_saved_without_session_id(self, rag):
        """add_exchange is NOT called when no session_id is provided."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("No session question")

        rag.session_manager.add_exchange.assert_not_called()

    def test_saved_question_is_the_raw_user_input(self, rag):
        """add_exchange receives the user's original question, not the wrapped prompt."""
        rag.ai_generator.generate_response.return_value = "Answer."
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("User's raw question", session_id="sess-004")

        saved_question = rag.session_manager.add_exchange.call_args[0][1]
        assert saved_question == "User's raw question"


# ---------------------------------------------------------------------------
# Content-query routing (integration-style unit tests)
# ---------------------------------------------------------------------------

class TestContentQueryRouting:

    def test_content_question_reaches_ai_with_tools_available(self, rag):
        """A content-specific query is sent to the AI with tools enabled."""
        rag.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"}
        ]
        rag.ai_generator.generate_response.return_value = "Course content answer."
        rag.tool_manager.get_last_sources.return_value = ["Course X - Lesson 1"]

        response, sources = rag.query("What does Lesson 1 of Course X cover?")

        # Tools must have been passed
        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert kwargs["tools"] is not None
        assert len(kwargs["tools"]) > 0

        assert response == "Course content answer."
        assert "Course X - Lesson 1" in sources

    def test_general_knowledge_question_also_reaches_ai_with_tools(self, rag):
        """Even general knowledge questions go through generate_response with tools."""
        rag.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"}
        ]
        rag.ai_generator.generate_response.return_value = "The sun is a star."
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("What is the sun?")

        kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in kwargs  # tools are always offered; Claude decides whether to use them
        assert response == "The sun is a star."
        assert sources == []
