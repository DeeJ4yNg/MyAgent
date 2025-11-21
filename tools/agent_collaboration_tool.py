"""
Agent Collaboration Tool

This tool allows sub-agents to call other sub-agents for collaboration.
"""

from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown


_AGENT_CACHE = {}

@tool
async def call_other_agent(
    agent_name: str,
    request: str,
    agent_registry: Optional[dict] = None
) -> str:
    """
    Call another agent to help with a task. This enables inter-agent collaboration.
    Parameters:
        agent_name (str): The name of the agent to call. Must be one of "coder", "file_manager", or "log_analyzer".
        request (str): The request or task description for the agent.
        agent_registry (Optional[dict]): A dictionary mapping agent names to agent instances. If not provided, the global agent cache will be used.
    Returns:
        str: The response or result from the called agent.
    """
    # Validate agent name
    valid_agents = ["coder", "file_manager", "log_analyzer"]
    if agent_name not in valid_agents:
        return f"Error: Invalid agent name '{agent_name}'. Available agents: {', '.join(valid_agents)}"
    # Get agent instance from registry or cache
    registry = agent_registry or _AGENT_CACHE
    agent = registry.get(agent_name)
    if not agent:
        try:
            if agent_name == "coder":
                from coder_agent import CoderAgent
                agent = _AGENT_CACHE.get("coder") or CoderAgent()
                await agent.initialize()
                _AGENT_CACHE["coder"] = agent
            elif agent_name == "file_manager":
                from file_management_agent import Agent as FileAgent
                agent = _AGENT_CACHE.get("file_manager") or FileAgent()
                await agent.initialize()
                _AGENT_CACHE["file_manager"] = agent
            elif agent_name == "log_analyzer":
                from log_analyzer_agent import LogAnalyzerAgent
                agent = _AGENT_CACHE.get("log_analyzer") or LogAnalyzerAgent()
                await agent.initialize()
                _AGENT_CACHE["log_analyzer"] = agent
        except Exception as e:
            return f"Error initializing '{agent_name}' agent: {str(e)}"
    try:
        temp_state = type('State', (), {'messages': [HumanMessage(content=request)]})()
        response = agent.model_response(temp_state)
        if "messages" in response and len(response["messages"]) > 0:
            response_message = response["messages"][0]
            if (hasattr(response_message, 'tool_calls') and response_message.tool_calls):
                temp_state.messages = [HumanMessage(content=request), response_message]
                tool_response = await agent.tool_use(temp_state)
                if "messages" in tool_response:
                    temp_state.messages.extend(tool_response["messages"])
                final_response = agent.model_response(temp_state)
                final_message = final_response["messages"][0]
                if isinstance(final_message.content, list):
                    response_text = ""
                    for item in final_message.content:
                        if item["type"] == "text":
                            response_text += item.get("text", "")
                else:
                    response_text = final_message.content
                return f"Response from {agent_name} agent:\n\n{response_text}"
            else:
                if isinstance(response_message.content, list):
                    response_text = ""
                    for item in response_message.content:
                        if item["type"] == "text":
                            response_text += item.get("text", "")
                else:
                    response_text = response_message.content
                return f"Response from {agent_name} agent:\n\n{response_text}"
        else:
            return f"Error: No response from {agent_name} agent."
    except Exception as e:
        return f"Error calling {agent_name} agent: {str(e)}"

