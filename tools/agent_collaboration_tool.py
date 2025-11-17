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


@tool
def call_other_agent(
    agent_name: str,
    request: str,
    agent_registry: Optional[dict] = None
) -> str:
    """
    Call another agent to help with a task. This enables inter-agent collaboration.
    
    Args:
        agent_name: Name of the agent to call ("coder", "file_manager", "log_analyzer")
        request: The request or question to send to the other agent
        agent_registry: Registry of available agents (set automatically by orchestrator)
    
    Returns:
        Response from the called agent
    """
    console = Console()
    
    if not agent_registry:
        return "Error: Agent registry not available. Cannot call other agents."
    
    # Validate agent name
    valid_agents = ["coder", "file_manager", "log_analyzer"]
    if agent_name not in valid_agents:
        return f"Error: Invalid agent name '{agent_name}'. Available agents: {', '.join(valid_agents)}"
    
    # Get the agent from registry
    agent = agent_registry.get(agent_name)
    if not agent:
        return f"Error: Agent '{agent_name}' not found in registry."
    
    try:
        # Create a temporary state with the request
        temp_state = type('State', (), {'messages': [HumanMessage(content=request)]})()
        
        # Call the agent's model_response method (synchronous)
        try:
            response = agent.model_response(temp_state)
        except Exception as e:
            return f"Error calling {agent_name} agent's model_response: {str(e)}"
        
        # Extract response content
        if "messages" in response and len(response["messages"]) > 0:
            response_message = response["messages"][0]
            
            # Check if tool calls are needed
            if (hasattr(response_message, 'tool_calls') and 
                response_message.tool_calls):
                # Execute tools if needed
                import asyncio
                try:
                    tool_response = asyncio.run(agent.tool_use(temp_state))
                except Exception as e:
                    return f"Error executing tools for {agent_name} agent: {str(e)}"
                
                # Get final response after tool execution
                temp_state.messages = [HumanMessage(content=request), response_message]
                if "messages" in tool_response:
                    temp_state.messages.extend(tool_response["messages"])
                
                final_response = agent.model_response(temp_state)
                final_message = final_response["messages"][0]
                
                # Extract content
                if isinstance(final_message.content, list):
                    response_text = ""
                    for item in final_message.content:
                        if item["type"] == "text":
                            response_text += item.get("text", "")
                else:
                    response_text = final_message.content
                
                return f"Response from {agent_name} agent:\n\n{response_text}"
            else:
                # No tool calls, return direct response
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

