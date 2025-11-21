from typing import Annotated, Sequence
from dotenv import load_dotenv
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from tools.token_tracker import tracker


class GeneralAssistantState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class GeneralAssistantAgent:
    def __init__(self, config_path: str = "config.json"):
        self._initialized = False
        load_dotenv()
        self.console = Console()
        self.config = self._load_config(config_path)
        self.agent_registry = None
        self.model_type = self.config.get("model", {}).get("type", "ollama")
        if self.model_type == "ollama":
            ollama_config = self.config.get("model", {}).get("ollama", {})
            general_config = self.config.get("model", {}).get("general", {})
            base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
            model = os.getenv("OLLAMA_MODEL", ollama_config.get("model", "llama3.1:8b"))
            temperature = general_config.get("temperature", 0.3)
            self.model = ChatOllama(base_url=base_url, model=model, temperature=temperature, max_tokens=ollama_config.get("max_tokens", 4096))
        else:
            cloud_config = self.config.get("model", {}).get("cloud", {})
            general_config = self.config.get("model", {}).get("general", {})
            api_base = os.getenv("OPENAI_API_BASE", cloud_config.get("api_base", "https://api.openai.com/v1"))
            api_key = os.getenv(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
            model_name = os.getenv(general_config.get("model_env", "OPENAI_MODEL_GENERAL"), cloud_config.get("model", "gpt-4"))
            temperature = general_config.get("temperature", 0.3)
            self.model = ChatOpenAI(base_url=api_base, api_key=api_key, model=model_name, temperature=temperature, max_tokens=cloud_config.get("max_tokens", 4096))
        self.workflow = StateGraph(GeneralAssistantState)
        self.workflow.add_node("user_input", self.user_input)
        self.workflow.add_node("model_response", self.model_response)
        self.workflow.add_node("tool_use", self.tool_use)
        self.workflow.set_entry_point("user_input")
        self.workflow.add_edge("tool_use", "model_response")
        self.workflow.add_conditional_edges("user_input", self.check_exit, {"exit": END, "user_input": "model_response"})
        self.workflow.add_conditional_edges("model_response", self.check_tool_use, {"tool_use": "tool_use", "user_input": "user_input"})

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def set_agent_registry(self, agent_registry: dict):
        self.agent_registry = agent_registry

    async def initialize(self):
        if self._initialized:
            return self
        local_tools = []
        if self.agent_registry:
            from tools.agent_collaboration_tool import call_other_agent
            from langchain_core.tools import tool
            def call_agent_with_registry(agent_name: str, request: str) -> str:
                """Call another agent (coder, file_manager, log_analyzer) to help with a task."""
                return call_other_agent.invoke({"agent_name": agent_name, "request": request, "agent_registry": self.agent_registry})
            collaboration_tool = tool(call_agent_with_registry)
            collaboration_tool.name = "call_other_agent"
            local_tools.append(collaboration_tool)
        self.tools = local_tools
        self.model_with_tools = self.model.bind_tools(self.tools)
        agent_config = self.config.get("agents", {}).get("general", {})
        db_path = os.path.join(os.getcwd(), agent_config.get("checkpoint_db", "general_checkpoints.db"))
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)
        self._initialized = True
        self.console.print(
            Panel.fit(
                Markdown("**General Assistant**"),
                title="[bold green]Ready[/bold green]",
                border_style="green",
            )
        )
        return self

    async def close_checkpointer(self):
        if hasattr(self, "_checkpointer_ctx"):
            await self._checkpointer_ctx.__aexit__(None, None, None)

    def user_input(self, state: GeneralAssistantState) -> GeneralAssistantState:
        return {"messages": state.messages}

    def model_response(self, state: GeneralAssistantState) -> GeneralAssistantState:
        system_text = """
You are a helpful General Assistant. Answer user requests clearly and efficiently.
Use tools to consult specialized agents when their expertise is needed.
If a task requires coding, file operations, or log analysis, call the appropriate agent.
"""
        messages = [SystemMessage(content=system_text)] + list(state.messages)
        response = self.model_with_tools.invoke(messages)
        try:
            prompt_text = "".join([m.content if isinstance(m.content, str) else str(m.content) for m in messages])
            completion_text = response.content if isinstance(response.content, str) else str(response.content)
            tracker.record("general", tracker.estimate_tokens(prompt_text), tracker.estimate_tokens(completion_text))
        except Exception:
            pass
        return {"messages": [response]}

    def check_tool_use(self, state: GeneralAssistantState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_use"
        for msg in reversed(state.messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return "tool_use"
        return "user_input"

    def check_exit(self, state: GeneralAssistantState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "content") and last_message.content == "__EXIT__":
            return "exit"
        return "user_input"

    async def tool_use(self, state: GeneralAssistantState) -> GeneralAssistantState:
        from langgraph.prebuilt import ToolNode
        from langchain_core.messages import AIMessage
        response = []
        tools_by_name = {t.name: t for t in self.tools}
        last_message = state.messages[-1]
        ai_msg = None
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            ai_msg = last_message
        else:
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    ai_msg = msg
                    break
        if not ai_msg:
            return {"messages": []}
        for tc in ai_msg.tool_calls:
            tool = tools_by_name.get(tc["name"])
            if not tool:
                continue
            tool_node = ToolNode([tool])
            tmp_state = {"messages": [ai_msg]}
            tool_result = await tool_node.ainvoke(tmp_state)
            if "messages" in tool_result and tool_result["messages"]:
                response.append(tool_result["messages"][0])
        return {"messages": response}