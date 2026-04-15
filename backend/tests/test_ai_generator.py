"""
Tests for AIGenerator in ai_generator.py.

Focus: verify that the generator correctly dispatches tool calls to the
CourseSearchTool (via ToolManager) and properly constructs both the first
and follow-up API calls.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def text_block(text: str):
    """Create a mock text content block."""
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def tool_use_block(name: str, input_kwargs: dict, tool_id: str = "tool_abc"):
    """Create a mock tool_use content block."""
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.input = input_kwargs
    b.id = tool_id
    return b


def make_response(stop_reason: str, content_blocks: list):
    """Create a mock Anthropic Message response."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    """AIGenerator with a fully mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        instance = AIGenerator(api_key="test-key", model="claude-test")
        # Expose the mock client for assertion
        instance._mock_client = MockAnthropic.return_value
        instance.client = instance._mock_client
    return instance


# ---------------------------------------------------------------------------
# Direct (no-tool) response tests
# ---------------------------------------------------------------------------

class TestDirectResponse:

    def test_returns_text_when_stop_reason_is_end_turn(self, gen):
        """generate_response() returns the text content directly on end_turn."""
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("The answer is 42.")]
        )

        result = gen.generate_response(query="What is 6*7?")

        assert result == "The answer is 42."

    def test_only_one_api_call_when_no_tool_use(self, gen):
        """Only a single API call is made when Claude does not request a tool."""
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Direct answer.")]
        )

        gen.generate_response(query="General question")

        assert gen.client.messages.create.call_count == 1

    def test_tools_absent_from_api_call_when_not_provided(self, gen):
        """'tools' and 'tool_choice' are not included when caller passes no tools."""
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Answer.")]
        )

        gen.generate_response(query="question")

        kwargs = gen.client.messages.create.call_args[1]
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_tools_included_in_api_call_when_provided(self, gen):
        """When tools are provided, they are passed to the API with tool_choice=auto."""
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Answer.")]
        )

        gen.generate_response(query="question", tools=tools)

        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == {"type": "auto"}


# ---------------------------------------------------------------------------
# Tool call dispatching tests
# ---------------------------------------------------------------------------

class TestToolCallDispatching:

    def test_tool_use_triggers_execute_tool_on_tool_manager(self, gen):
        """When Claude returns tool_use, tool_manager.execute_tool() is called."""
        first = make_response(
            "tool_use",
            [tool_use_block("search_course_content", {"query": "Python decorators"}, "id1")],
        )
        second = make_response("end_turn", [text_block("Decorators wrap functions.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Decorators are function wrappers."

        result = gen.generate_response(
            query="What are decorators?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        mock_tm.execute_tool.assert_called_once_with(
            "search_course_content", query="Python decorators"
        )
        assert result == "Decorators wrap functions."

    def test_correct_tool_name_dispatched(self, gen):
        """The tool name from the content block is passed verbatim to execute_tool."""
        first = make_response(
            "tool_use",
            [tool_use_block("get_course_outline", {"course_title": "MCP"}, "id2")],
        )
        second = make_response("end_turn", [text_block("Outline result.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "Course: MCP\nLesson 1: Intro"

        gen.generate_response(
            query="Outline for MCP?",
            tools=[{"name": "get_course_outline"}],
            tool_manager=mock_tm,
        )

        assert mock_tm.execute_tool.call_args[0][0] == "get_course_outline"

    def test_tool_inputs_forwarded_as_kwargs(self, gen):
        """All keys from tool_use.input are forwarded to execute_tool as keyword args."""
        first = make_response(
            "tool_use",
            [tool_use_block(
                "search_course_content",
                {"query": "RAG overview", "course_name": "AI Course", "lesson_number": 3},
                "id3",
            )],
        )
        second = make_response("end_turn", [text_block("RAG answer.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "RAG stands for ..."

        gen.generate_response(
            query="What is RAG?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        mock_tm.execute_tool.assert_called_once_with(
            "search_course_content",
            query="RAG overview",
            course_name="AI Course",
            lesson_number=3,
        )

    def test_two_api_calls_are_made_when_tool_is_used(self, gen):
        """Exactly two API calls are made: one to get tool use, one to synthesise."""
        first = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "test"})]
        )
        second = make_response("end_turn", [text_block("Final.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        assert gen.client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# Second API call construction tests
# ---------------------------------------------------------------------------

class TestSecondApiCallConstruction:

    def test_tool_result_in_second_call_messages(self, gen):
        """The tool result is included as a tool_result message in the follow-up call."""
        first = make_response(
            "tool_use",
            [tool_use_block("search_course_content", {"query": "RAG"}, "tool_xyz")],
        )
        second = make_response("end_turn", [text_block("Answer.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "RAG retrieves documents before generating."

        gen.generate_response(
            query="What is RAG?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        second_call_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        # Last message should be the user-role tool_result
        tool_result_msg = second_call_msgs[-1]
        assert tool_result_msg["role"] == "user"

        result_block = tool_result_msg["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "tool_xyz"
        assert "RAG retrieves documents before generating." in result_block["content"]

    def test_assistant_tool_use_message_is_in_second_call(self, gen):
        """The assistant's tool use response is included in the follow-up call's messages."""
        tool_block = tool_use_block("search_course_content", {"query": "test"}, "id99")
        first = make_response("tool_use", [tool_block])
        second = make_response("end_turn", [text_block("Answer.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "search result"

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        second_call_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        # Expect: [user_query, assistant_tool_use, user_tool_result]
        assistant_msg = second_call_msgs[1]
        assert assistant_msg["role"] == "assistant"

    def test_synthesis_call_has_no_tools(self, gen):
        """After 2 tool rounds, the final synthesis call has no 'tools' key."""
        first = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "step1"}, "id1")]
        )
        second = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "step2"}, "id2")]
        )
        third = make_response("end_turn", [text_block("Final synthesis.")])
        gen.client.messages.create.side_effect = [first, second, third]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        synthesis_kwargs = gen.client.messages.create.call_args_list[2][1]
        assert "tools" not in synthesis_kwargs

    def test_system_prompt_preserved_in_second_call(self, gen):
        """The system prompt used in the first call is also used in the follow-up call."""
        first = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "test"})]
        )
        second = make_response("end_turn", [text_block("Answer.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "result"

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        first_system = gen.client.messages.create.call_args_list[0][1]["system"]
        second_system = gen.client.messages.create.call_args_list[1][1]["system"]
        assert first_system == second_system


# ---------------------------------------------------------------------------
# Conversation history tests
# ---------------------------------------------------------------------------

class TestConversationHistory:

    def test_history_appended_to_system_prompt(self, gen):
        """Conversation history is appended to the system prompt, not the messages list."""
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Answer.")]
        )
        history = "User: earlier question\nAssistant: earlier answer"

        gen.generate_response(query="Follow-up", conversation_history=history)

        kwargs = gen.client.messages.create.call_args[1]
        assert "earlier question" in kwargs["system"]
        assert "earlier answer" in kwargs["system"]

    def test_no_history_uses_base_system_prompt_unchanged(self, gen):
        """Without history, the system prompt equals the static SYSTEM_PROMPT exactly."""
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Answer.")]
        )

        gen.generate_response(query="Simple question")

        kwargs = gen.client.messages.create.call_args[1]
        assert kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    def test_history_not_sent_as_separate_message(self, gen):
        """Conversation history must NOT appear as an extra message; it goes in system only."""
        gen.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Answer.")]
        )
        history = "User: old question\nAssistant: old answer"

        gen.generate_response(query="New question", conversation_history=history)

        kwargs = gen.client.messages.create.call_args[1]
        messages = kwargs["messages"]
        assert len(messages) == 1, "Only the current query should be in messages"
        assert messages[0] == {"role": "user", "content": "New question"}


# ---------------------------------------------------------------------------
# Sequential tool calling tests
# ---------------------------------------------------------------------------

class TestSequentialToolCalling:

    def _make_tm(self, return_value="result"):
        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = return_value
        return mock_tm

    def test_two_rounds_executes_tools_in_each_round(self, gen):
        """Both tool rounds execute their respective tools."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("tool_use", [tool_use_block("get_course_outline", {"course_title": "X"}, "id2")])
        third = make_response("end_turn", [text_block("Done.")])
        gen.client.messages.create.side_effect = [first, second, third]

        mock_tm = self._make_tm()
        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        assert mock_tm.execute_tool.call_count == 2
        assert gen.client.messages.create.call_count == 3

    def test_two_rounds_returns_text_from_third_response(self, gen):
        """Return value is the text from the synthesis (3rd) response."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q2"}, "id2")])
        third = make_response("end_turn", [text_block("Synthesised answer.")])
        gen.client.messages.create.side_effect = [first, second, third]

        result = gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=self._make_tm(),
        )

        assert result == "Synthesised answer."

    def test_second_round_api_call_includes_tools(self, gen):
        """The 2nd API call (round 2) still includes tools so Claude can call again."""
        tools = [{"name": "search_course_content"}]
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q2"}, "id2")])
        third = make_response("end_turn", [text_block("Done.")])
        gen.client.messages.create.side_effect = [first, second, third]

        gen.generate_response(
            query="question",
            tools=tools,
            tool_manager=self._make_tm(),
        )

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tools"] == tools

    def test_messages_grow_correctly_across_two_rounds(self, gen):
        """Synthesis call's messages list has 5 entries in alternating user/assistant roles."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q2"}, "id2")])
        third = make_response("end_turn", [text_block("Done.")])
        gen.client.messages.create.side_effect = [first, second, third]

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=self._make_tm(),
        )

        synthesis_msgs = gen.client.messages.create.call_args_list[2][1]["messages"]
        assert len(synthesis_msgs) == 5
        roles = [m["role"] for m in synthesis_msgs]
        assert roles == ["user", "assistant", "user", "assistant", "user"]

    def test_stops_after_one_round_if_second_returns_end_turn(self, gen):
        """If round 2 returns end_turn, its text is returned directly (no synthesis call)."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("end_turn", [text_block("Answer after one tool round.")])
        gen.client.messages.create.side_effect = [first, second]

        result = gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=self._make_tm(),
        )

        assert gen.client.messages.create.call_count == 2
        assert result == "Answer after one tool round."


# ---------------------------------------------------------------------------
# Tool execution error tests
# ---------------------------------------------------------------------------

class TestToolExecutionErrors:

    def test_tool_error_stops_further_rounds(self, gen):
        """A tool execution exception stops the loop after 1 round (no round 2 tool call)."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("end_turn", [text_block("Graceful response.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = Exception("db error")

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        assert mock_tm.execute_tool.call_count == 1
        assert gen.client.messages.create.call_count == 2

    def test_tool_error_passed_as_tool_result_to_synthesis(self, gen):
        """The error message is included as tool_result content in the synthesis call."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("end_turn", [text_block("Graceful response.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = Exception("db error")

        gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        synthesis_msgs = gen.client.messages.create.call_args_list[1][1]["messages"]
        tool_result_msg = synthesis_msgs[-1]
        assert tool_result_msg["role"] == "user"
        result_block = tool_result_msg["content"][0]
        assert result_block["type"] == "tool_result"
        assert "db error" in result_block["content"]

    def test_tool_error_returns_synthesised_response(self, gen):
        """generate_response returns the synthesis text and does not raise on tool error."""
        first = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q1"}, "id1")])
        second = make_response("end_turn", [text_block("I was unable to complete the search.")])
        gen.client.messages.create.side_effect = [first, second]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = Exception("db error")

        result = gen.generate_response(
            query="question",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        assert result == "I was unable to complete the search."
