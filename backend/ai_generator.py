import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use `search_course_content` **only** for questions about specific course content or detailed educational materials
- Use `get_course_outline` for outline-related queries (e.g., "what lessons are in X?", "give me the outline of X", "list the lessons for X")
  - Always include: course title, course link, and each lesson's number and title
- You may make up to 2 sequential tool calls per query when the result of the first is needed to formulate the second
- Only call tools when necessary; answer general questions from existing knowledge
- Synthesize results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    MAX_ROUNDS = 2

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        for _ in range(self.MAX_ROUNDS):
            response = self.client.messages.create(**api_params)

            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            tool_results, had_error = self._handle_tool_execution(response, tool_manager)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            api_params["messages"] = messages

            if had_error:
                break

        # All rounds used tools (or error): synthesise without tools
        final_params = {**self.base_params, "messages": messages, "system": system_content}
        return self.client.messages.create(**final_params).content[0].text

    def _handle_tool_execution(self, response, tool_manager) -> tuple:
        """
        Execute all tool_use blocks in a response.

        Returns:
            (tool_results, had_error): list of tool_result dicts and error flag
        """
        tool_results = []
        had_error = False

        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Tool execution failed: {e}"
                    had_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        return tool_results, had_error