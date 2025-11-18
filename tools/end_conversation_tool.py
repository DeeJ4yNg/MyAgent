from typing import Optional
from langchain_core.tools import tool
from tools.token_tracker import tracker
import json

@tool
def end_conversation(session_id: Optional[str] = None, summary: Optional[str] = None) -> str:
    """
    End the conversation and return a JSON payload containing an optional session summary
    and aggregated token usage across agents.
    """
    usage = tracker.summary()
    return json.dumps({"session_id": session_id, "summary": summary or "", "token_usage": usage})