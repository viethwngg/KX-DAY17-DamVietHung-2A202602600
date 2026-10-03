from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    live_mode: bool = False

    def __post_init__(self) -> None:
        if self.compact_threshold_tokens < 1 or self.compact_keep_messages < 1:
            raise ValueError('Compact threshold and keep_messages must be positive')


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f'{prefix}_PROVIDER', fallback.provider if fallback else 'openai'))
    defaults = {'openai': 'gpt-4o-mini', 'custom': 'local-model', 'gemini': 'gemini-2.5-flash',
                'anthropic': 'claude-sonnet-4-5', 'ollama': 'llama3.2', 'openrouter': 'openai/gpt-4o-mini'}
    same_provider = fallback is not None and provider == fallback.provider
    key = os.getenv(f'{prefix}_API_KEY') or os.getenv(f'{provider.upper()}_API_KEY')
    if provider == 'gemini':
        key = key or os.getenv('GOOGLE_API_KEY')
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f'{prefix}_MODEL', fallback.model_name if same_provider else defaults[provider]),
        temperature=float(os.getenv(f'{prefix}_TEMPERATURE', str(fallback.temperature if same_provider else 0))),
        api_key=key or (fallback.api_key if same_provider else None),
        base_url=os.getenv(f'{prefix}_BASE_URL') or os.getenv(f'{provider.upper()}_BASE_URL')
        or (fallback.base_url if same_provider else None),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        if (root / '.env').exists():
            raise RuntimeError('Install python-dotenv to load .env configuration')
    else:
        load_dotenv(root / '.env', override=False)
    state_dir = Path(os.getenv('LAB_STATE_DIR', str(root / 'state')))
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    model = _provider_config('LLM')
    config = LabConfig(
        base_dir=root, data_dir=root / 'data', state_dir=state_dir.resolve(),
        compact_threshold_tokens=int(os.getenv('COMPACT_THRESHOLD_TOKENS', '1400')),
        compact_keep_messages=int(os.getenv('COMPACT_KEEP_MESSAGES', '6')),
        model=model, judge_model=_provider_config('JUDGE', model),
        live_mode=os.getenv('LLM_LIVE', 'false').lower() in {'true', '1', 'yes'},
    )
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
