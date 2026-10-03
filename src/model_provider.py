from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower().replace('_', '-')
    provider = {'anthorpic': 'anthropic', 'claude': 'anthropic',
                'google': 'gemini', 'google-genai': 'gemini',
                'openai-compatible': 'custom', 'open-router': 'openrouter'}.get(provider, provider)
    if provider not in {'openai', 'custom', 'gemini', 'anthropic', 'ollama', 'openrouter'}:
        raise ValueError(f'Unsupported provider: {value!r}')
    return provider


def build_chat_model(config: ProviderConfig) -> Any:
    """Load only the requested integration; construction makes no model call."""
    provider = normalize_provider(config.provider)
    integrations = {
        'openai': ('langchain_openai', 'ChatOpenAI'),
        'custom': ('langchain_openai', 'ChatOpenAI'),
        'gemini': ('langchain_google_genai', 'ChatGoogleGenerativeAI'),
        'anthropic': ('langchain_anthropic', 'ChatAnthropic'),
        'ollama': ('langchain_ollama', 'ChatOllama'),
        'openrouter': ('langchain_openrouter', 'ChatOpenRouter'),
    }
    if provider != 'ollama' and not config.api_key:
        raise ValueError(f'API key required for {provider} in live mode')
    if provider == 'custom' and not config.base_url:
        raise ValueError('CUSTOM_BASE_URL is required for custom provider')
    module, class_name = integrations[provider]
    try:
        model_class = getattr(import_module(module), class_name)
    except ImportError as exc:
        raise RuntimeError(f'Install {module.replace("_", "-")} to use {provider}') from exc
    kwargs: dict[str, Any] = {'model': config.model_name, 'temperature': config.temperature}
    if config.api_key:
        kwargs['api_key'] = config.api_key
    if config.base_url:
        if provider == 'gemini':
            kwargs['client_options'] = {'api_endpoint': config.base_url}
        else:
            kwargs['base_url'] = config.base_url
    return model_class(**kwargs)
