import threading
from typing import Dict, Any

class TokenTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.totals: Dict[str, Dict[str, Any]] = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        with self.lock:
            entry = self.totals.get(model, {"prompt": 0, "completion": 0, "calls": 0})
            entry["prompt"] += int(prompt_tokens or 0)
            entry["completion"] += int(completion_tokens or 0)
            entry["calls"] += 1
            self.totals[model] = entry

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def summary(self) -> Dict[str, Any]:
        total_prompt = sum(v["prompt"] for v in self.totals.values())
        total_completion = sum(v["completion"] for v in self.totals.values())
        return {
            "total_prompt": total_prompt,
            "total_completion": total_completion,
            "total": total_prompt + total_completion,
            "per_model": self.totals,
        }

tracker = TokenTracker()