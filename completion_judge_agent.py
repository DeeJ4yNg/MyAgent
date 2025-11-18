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
from tools.end_conversation_tool import end_conversation
from tools.token_tracker import tracker

class JudgeState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class CompletionJudgeAgent:
    def __init__(self, config_path: str = "config.json"):
        self._initialized = False
        load_dotenv()
        self.console = Console()
        self.config = self._load_config(config_path)
        self.model_type = self.config.get("model", {}).get("type", "ollama")
        if self.model_type == "ollama":
            ollama_config = self.config.get("model", {}).get("ollama", {})
            judge_config = self.config.get("model", {}).get("orchestrator", {})
            base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
            model = os.getenv("OLLAMA_MODEL", ollama_config.get("model", "llama3.1:8b"))
            temperature = judge_config.get("temperature", 0.2)
            self.model = ChatOllama(base_url=base_url, model=model, temperature=temperature, max_tokens=ollama_config.get("max_tokens", 4096))
        else:
            cloud_config = self.config.get("model", {}).get("cloud", {})
            judge_config = self.config.get("model", {}).get("orchestrator", {})
            api_base = os.getenv("OPENAI_API_BASE", cloud_config.get("api_base", "https://api.openai.com/v1"))
            api_key = os.getenv(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
            model_name = os.getenv(judge_config.get("model_env", "OPENAI_MODEL_ORC"), cloud_config.get("model", "gpt-4"))
            temperature = judge_config.get("temperature", 0.2)
            self.model = ChatOpenAI(base_url=api_base, api_key=api_key, model=model_name, temperature=temperature, max_tokens=cloud_config.get("max_tokens", 4096))
        self.workflow = StateGraph(JudgeState)
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

    async def initialize(self):
        if self._initialized:
            return self
        tools = [end_conversation]
        self.model_with_tools = self.model.bind_tools(tools)
        agent_config = self.config.get("agents", {}).get("orchestrator", {})
        db_path = os.path.join(os.getcwd(), agent_config.get("checkpoint_db", "orchestrator_checkpoints.db"))
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)
        self._initialized = True
        return self

    def user_input(self, state: JudgeState) -> JudgeState:
        return {"messages": state.messages}

    def model_response(self, state: JudgeState) -> JudgeState:
        system_text = """
You decide if the task response is Completed or NeedsUserInteraction.
If the task is Completed, call the end_conversation tool with a short summary.
If the task NeedsUserInteraction, respond with: NEEDS_USER_INTERACTION and specify what is needed.
"""
        messages = [
            SystemMessage(content=system_text),
        ] + state.messages
        response = self.model_with_tools.invoke(messages)
        try:
            prompt_text = ""
            for m in messages:
                c = m.content
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_text += item.get("text", "")
                elif isinstance(c, str):
                    prompt_text += c
                else:
                    prompt_text += str(c)
            if isinstance(response.content, list):
                completion_text = ""
                for item in response.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        completion_text += item.get("text", "")
            else:
                completion_text = response.content if isinstance(response.content, str) else str(response.content)
            tracker.record("completion_judge", tracker.estimate_tokens(prompt_text), tracker.estimate_tokens(completion_text))
        except Exception:
            pass
        return {"messages": [response]}

    def check_tool_use(self, state: JudgeState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_use"
        return "user_input"

    async def tool_use(self, state: JudgeState) -> JudgeState:
        from langgraph.prebuilt import ToolNode
        from langchain_core.messages import AIMessage
        response = []
        tools_by_name = {"end_conversation": end_conversation}
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}
        for tc in last_message.tool_calls:
            tool = tools_by_name.get(tc["name"]) 
            tool_node = ToolNode([tool])
            tool_result = await tool_node.ainvoke(state)
            response.append(tool_result["messages"][0])
        return {"messages": response}

    def check_exit(self, state: JudgeState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "content") and last_message.content == "__EXIT__":
            return "exit"
        return "user_input"