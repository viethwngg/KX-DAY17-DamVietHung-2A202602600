from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model
from responses import SYSTEM_PROMPT, model_answer, offline_response, prompt_tokens


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Recent messages + durable User.md + bounded per-thread summary."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / 'profiles')
        self.compact_memory = CompactMemoryManager(
            self.config.compact_threshold_tokens, self.config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id in self.thread_users and self.thread_users[thread_id] != user_id:
            raise ValueError('A thread cannot be shared between users')
        self.thread_users[thread_id] = user_id
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        if updates:
            self.profile_store.upsert_facts(user_id, updates)
        self.compact_memory.append(thread_id, 'user', message)
        messages = self._prompt_messages(user_id, thread_id)
        if self.langchain_agent is not None:
            answer, output, processed = model_answer(self.langchain_agent, messages)
        else:
            answer = self._offline_response(user_id, thread_id, message)
            output = estimate_tokens(answer)
            processed = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.compact_memory.append(thread_id, 'assistant', answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + processed
        return {'response': answer, 'tokens': output, 'prompt_tokens': processed,
                'token_usage': self.token_usage(thread_id),
                'prompt_tokens_processed': self.prompt_token_usage(thread_id),
                'mode': 'live' if self.langchain_agent is not None else 'offline'}

    def _prompt_messages(self, user_id: str, thread_id: str) -> list[dict[str, str]]:
        context = self.compact_memory.context(thread_id)
        # Profile is authoritative when an older thread summary contains stale facts.
        system = SYSTEM_PROMPT + '\n\nUser.md (fact hiện tại, ưu tiên hơn summary):\n' + self.profile_store.read_text(user_id)
        if context['summary']:
            system += '\n\nSummary (ngữ cảnh cũ):\n' + context['summary']
        return [{'role': 'system', 'content': system}] + context['messages']

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        return prompt_tokens(self._prompt_messages(user_id, thread_id))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        return offline_response(message, self.profile_store.facts(user_id),
                                self.compact_memory.context(thread_id)['summary'])

    def _maybe_build_langchain_agent(self):
        if self.force_offline or not self.config.live_mode:
            return None
        # The same memory layer supplies context to every supported chat model.
        return build_chat_model(self.config.model)
