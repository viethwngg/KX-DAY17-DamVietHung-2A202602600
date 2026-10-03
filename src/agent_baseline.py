from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import extract_profile_updates, merge_facts, estimate_tokens
from model_provider import build_chat_model
from responses import SYSTEM_PROMPT, model_answer, offline_response, prompt_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Short-term memory only; each thread starts without any profile facts."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id in self.thread_users and self.thread_users[thread_id] != user_id:
            raise ValueError('A thread cannot be shared between users')
        self.thread_users[thread_id] = user_id
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        state = self.sessions.get(thread_id)
        return state.token_usage if state else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        state = self.sessions.get(thread_id)
        return state.prompt_tokens_processed if state else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({'role': 'user', 'content': message})
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + state.messages
        if self.langchain_agent is not None:
            answer, output, processed = model_answer(self.langchain_agent, messages)
        else:
            facts: dict[str, str] = {}
            for item in state.messages:
                if item['role'] == 'user':
                    facts = merge_facts(facts, extract_profile_updates(item['content']))
            answer = offline_response(message, facts)
            output, processed = estimate_tokens(answer), prompt_tokens(messages)
        state.messages.append({'role': 'assistant', 'content': answer})
        state.token_usage += output
        state.prompt_tokens_processed += processed
        return {'response': answer, 'tokens': output, 'prompt_tokens': processed,
                'token_usage': state.token_usage,
                'prompt_tokens_processed': state.prompt_tokens_processed,
                'mode': 'live' if self.langchain_agent is not None else 'offline'}

    def _maybe_build_langchain_agent(self):
        if self.force_offline or not self.config.live_mode:
            return None
        return build_chat_model(self.config.model)
