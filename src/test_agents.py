from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, summarize_messages
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def make_config(tmp_path: Path):
    return replace(load_config(), state_dir=tmp_path / 'state',
                   compact_threshold_tokens=300, compact_keep_messages=4, live_mode=False)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / 'profiles')
    assert store.read_text('alice') == ''
    assert store.file_size('alice') == 0
    path = store.write_text('alice', '# User\nTên: Hùng\nTên: Hùng\n')
    assert path.name == 'User.md'
    assert store.file_size('alice') == len(path.read_bytes())
    assert store.edit_text('alice', 'Hùng', 'Lan')
    assert store.read_text('alice').count('Hùng') == 1
    assert 'Lan' in store.read_text('alice')
    assert not store.edit_text('alice', 'missing', 'new')
    assert not store.edit_text('alice', '', 'new')
    store.upsert_fact('alice', 'location', 'Huế')
    store.upsert_fact('alice', 'location', 'Đà Nẵng')
    assert store.facts('alice')['location'] == 'Đà Nẵng'
    assert 'Huế' not in store.read_text('alice')
    assert 'Tên: Lan' in store.read_text('alice')
    store.write_text('alice', store.read_text('alice') + '\n- note: Ghi chú riêng\n')
    store.upsert_fact('alice', 'profession', 'MLOps engineer')
    assert '- note: Ghi chú riêng' in store.read_text('alice')


def test_profile_path_is_safe_and_users_do_not_collide(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    users = ['../../escape', 'a/b', 'a\\b', 'Alice', 'alice', 'CON', '.', 'người dùng']
    paths = [store.write_text(user, user).resolve() for user in users]
    assert len(set(str(path).casefold() for path in paths)) == len(users)
    assert all(path.is_relative_to(tmp_path.resolve()) for path in paths)
    assert [store.read_text(user) for user in users] == users
    with pytest.raises(ValueError):
        store.path_for('')


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=120, keep_messages=2)
    manager.append('long', 'user', 'Mình tên là Lan. Mình ở Huế.')
    manager.append('long', 'assistant', 'Đã ghi nhận.')
    manager.append('long', 'user', 'Tin NASA Artemis III là ví dụ về roadmap có các mốc kiểm chứng. ' * 30)
    context = manager.context('long')
    assert manager.compaction_count('long') == 1
    assert len(context['messages']) == 2
    assert 'Lan' in context['summary']
    assert context['messages'][-1]['content'].startswith('Tin NASA')
    assert manager.compaction_count('other') == 0
    # Returned context cannot be used to mutate the manager accidentally.
    context['messages'].clear()
    assert len(manager.context('long')['messages']) == 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply('lan', 'first', 'Mình tên là Lan. Mình ở Huế và đang làm backend engineer.')
        assert 'Lan' in agent.reply('lan', 'first', 'Mình tên gì?')['response']
    assert 'Lan' not in baseline.reply('lan', 'fresh', 'Mình tên gì?')['response']
    # Reconstruct the agent to prove the profile survived process/session state.
    restarted = AdvancedAgent(config, force_offline=True)
    assert 'Lan' in restarted.reply('lan', 'fresh', 'Mình tên gì?')['response']
    assert 'Lan' not in restarted.reply('other-user', 'other', 'Mình tên gì?')['response']


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    compact = AdvancedAgent(config, True)
    uncompressed = AdvancedAgent(replace(config, state_dir=tmp_path / 'plain', compact_threshold_tokens=1000000), True)
    baseline = BaselineAgent(config, True)
    for agent in (compact, uncompressed, baseline):
        agent.reply('u', 'long', 'Mình tên là Lan. Mình muốn trả lời ngắn gọn thành 3 bullet.')
        for index in range(20):
            agent.reply('u', 'long', f'Đoạn {index}: ' + 'Dữ liệu ngữ cảnh tạm thời của hệ thống. ' * 80)
    assert compact.compaction_count('long') > 1
    assert uncompressed.compaction_count('long') == 0
    assert compact.prompt_token_usage('long') < uncompressed.prompt_token_usage('long') * 0.5
    assert compact.prompt_token_usage('long') < baseline.prompt_token_usage('long') * 0.5
    assert 'Lan' in compact.reply('u', 'new', 'Mình tên gì?')['response']


@pytest.mark.parametrize('message', [
    'Mình đang ở Hà Nội?',
    'Mình tên là Lan phải không?',
    'Nếu mình chuyển sang product manager thì sao?',
    'Bạn có biết DũngCT không?',
    'Nếu sau này mình nhắc Đà Nẵng như ví dụ cũ thì đừng lấy làm nơi ở hiện tại.',
    'Mình đùa là chuyển sang product manager, nhưng đó chỉ là câu đùa.',
    'Hà Nội chỉ là nơi mình bay ra họp hai ngày chứ không phải nơi ở hiện tại.',
])
def test_questions_and_noise_are_not_profile_facts(message: str) -> None:
    assert extract_profile_updates(message) == {}


def test_corrections_replace_obsolete_facts(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), True)
    agent.reply('u', 'a', 'Mình ở Đà Nẵng và đang làm backend engineer.')
    agent.reply('u', 'a', 'Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.')
    agent.reply('u', 'a', 'Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.')
    agent.reply('u', 'a', 'Mình đùa rằng chuyển sang product manager, nhưng đó chỉ là câu đùa.')
    agent.reply('u', 'a', 'Hà Nội chỉ là nơi đi họp, không phải nơi ở hiện tại.')
    facts = agent.profile_store.facts('u')
    assert facts['location'] == 'Huế'
    assert facts['profession'] == 'MLOps engineer'
    answer = agent.reply('u', 'fresh', 'Nhắc lại nơi ở hiện tại và nghề nghiệp hiện tại.')['response']
    assert 'Huế' in answer and 'MLOps engineer' in answer
    assert 'Đà Nẵng' not in answer and 'backend engineer' not in answer
    assert 'product manager' not in answer and 'Hà Nội' not in answer


def test_questions_cannot_modify_existing_profile(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), True)
    agent.reply('u', 'a', 'Mình tên là Lan. Mình đang ở Huế.')
    before = agent.profile_store.read_text('u')
    agent.reply('u', 'new', 'Nếu mình tên là Mai và mình đang ở Hà Nội thì sao?')
    assert agent.profile_store.read_text('u') == before


def test_repeated_updates_do_not_grow_profile(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), True)
    message = 'Mình tên là Lan. Mình thích Python, AI. Mình muốn trả lời ngắn gọn, có bullet và ví dụ thực chiến.'
    agent.reply('u', 'a', message)
    before = agent.profile_store.read_text('u')
    for _ in range(15):
        agent.reply('u', 'a', message)
    assert agent.profile_store.read_text('u') == before
    assert 'Python' in agent.profile_store.facts('u')['interests']


def test_summary_keeps_topics_and_latest_correction() -> None:
    first = summarize_messages([
        {'role': 'user', 'content': 'Mình ở Huế. NASA Artemis III cho thấy roadmap cần có cột mốc kiểm chứng.'},
        {'role': 'assistant', 'content': 'Không dùng câu trả lời assistant để cập nhật facts.'},
    ])
    updated = summarize_messages([
        {'role': 'summary', 'content': first},
        {'role': 'user', 'content': 'Giờ mình đang ở Đà Nẵng. X-59 cân bằng hiệu năng với tác động lên cộng đồng.'},
    ])
    assert 'Artemis III' in updated and 'X-59' in updated
    assert '- location: Đà Nẵng' in updated
    assert '- location: Huế' not in updated


@pytest.mark.parametrize('agent_class', [BaselineAgent, AdvancedAgent])
def test_thread_cannot_leak_between_users(tmp_path: Path, agent_class) -> None:
    agent = agent_class(make_config(tmp_path), True)
    agent.reply('alice', 'shared', 'Mình tên là Lan.')
    with pytest.raises(ValueError, match='shared between users'):
        agent.reply('bob', 'shared', 'Mình tên gì?')


@pytest.mark.parametrize('filename', ['conversations.json', 'advanced_long_context.json'])
def test_full_dataset_cross_session_recall(tmp_path: Path, filename: str) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1400, compact_keep_messages=6)
    conversations = load_conversations(config.data_dir / filename)
    baseline = run_agent_benchmark('Baseline', BaselineAgent(config, True), conversations, config)
    advanced = run_agent_benchmark('Advanced', AdvancedAgent(config, True), conversations, config)
    assert baseline.recall_score == 0
    assert advanced.recall_score == 1
    assert baseline.memory_growth_bytes == 0
    assert advanced.memory_growth_bytes > 0
    if filename == 'advanced_long_context.json':
        assert advanced.compactions > 1
        assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed
    else:
        assert advanced.prompt_tokens_processed > baseline.prompt_tokens_processed


def test_benchmark_evaluates_before_future_corrections(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    conversations = [
        {'id': 'a', 'user_id': 'u', 'turns': ['Mình ở Huế.'],
         'recall_questions': [{'question': 'Hiện tại mình đang ở đâu?', 'expected_contains': ['Huế']}]},
        {'id': 'b', 'user_id': 'u', 'turns': ['Giờ mình đang ở Đà Nẵng.'],
         'recall_questions': [{'question': 'Hiện tại mình đang ở đâu?', 'expected_contains': ['Đà Nẵng']}]},
    ]
    assert run_agent_benchmark('Advanced', AdvancedAgent(config, True), conversations, config).recall_score == 1


def test_token_accounting_and_partial_recall(tmp_path: Path) -> None:
    agent = BaselineAgent(make_config(tmp_path), True)
    first = agent.reply('u', 't', 'Mình tên là Lan.')
    second = agent.reply('u', 't', 'Mình tên gì?')
    assert agent.token_usage('t') == first['tokens'] + second['tokens']
    assert agent.prompt_token_usage('t') == first['prompt_tokens'] + second['prompt_tokens']
    assert agent.token_usage('missing') == agent.prompt_token_usage('missing') == 0
    assert estimate_tokens('   ') == 0
    assert estimate_tokens('abcd') == 1
    assert recall_points('Huế', ['Huế', 'Python']) == 0.5
    assert recall_points('HUẾ', ['Huế']) == 1


@pytest.mark.parametrize('provider,module,class_name', [
    ('openai', 'langchain_openai', 'ChatOpenAI'),
    ('custom', 'langchain_openai', 'ChatOpenAI'),
    ('gemini', 'langchain_google_genai', 'ChatGoogleGenerativeAI'),
    ('anthropic', 'langchain_anthropic', 'ChatAnthropic'),
    ('ollama', 'langchain_ollama', 'ChatOllama'),
    ('openrouter', 'langchain_openrouter', 'ChatOpenRouter'),
])
def test_provider_routing_without_network(monkeypatch, provider, module, class_name) -> None:
    import model_provider
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(invoke=lambda _: None)
    def fake_import(name):
        assert name == module
        return SimpleNamespace(**{class_name: factory})
    monkeypatch.setattr(model_provider, 'import_module', fake_import)
    build_chat_model(ProviderConfig(provider, 'demo-model', api_key='test-key',
                                    base_url='https://example.test' if provider == 'custom' else None))
    assert captured['model'] == 'demo-model'
    if provider == 'custom':
        assert captured['base_url'] == 'https://example.test'


def test_invalid_configuration_and_provider(tmp_path: Path, monkeypatch) -> None:
    assert normalize_provider(' Anthorpic ') == 'anthropic'
    with pytest.raises(ValueError):
        normalize_provider('unknown')
    with pytest.raises(ValueError):
        build_chat_model(ProviderConfig('custom', 'model', api_key='test'))
    with pytest.raises(ValueError):
        build_chat_model(ProviderConfig('openai', 'model'))
    with pytest.raises(ValueError):
        CompactMemoryManager(0, 2)
    monkeypatch.setenv('COMPACT_THRESHOLD_TOKENS', '0')
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_env_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('LLM_PROVIDER', 'custom')
    monkeypatch.setenv('LLM_MODEL', 'demo')
    monkeypatch.setenv('CUSTOM_BASE_URL', 'https://example.test/v1')
    monkeypatch.setenv('CUSTOM_API_KEY', 'test-key')
    monkeypatch.setenv('LAB_STATE_DIR', 'runtime')
    monkeypatch.setenv('JUDGE_PROVIDER', 'ollama')
    monkeypatch.setenv('JUDGE_MODEL', 'local-judge')
    config = load_config(tmp_path)
    assert config.state_dir == tmp_path / 'runtime'
    assert config.model.provider == 'custom'
    assert config.model.api_key == 'test-key'
    assert config.judge_model.provider == 'ollama'
    assert config.judge_model.model_name == 'local-judge'


@pytest.mark.parametrize('agent_class', [BaselineAgent, AdvancedAgent])
def test_live_model_receives_memory_and_returns_usage(tmp_path: Path, monkeypatch, agent_class) -> None:
    module = __import__(agent_class.__module__)
    prompts = []
    def invoke(messages):
        prompts.append(messages)
        return SimpleNamespace(content='Câu trả lời thử.', usage_metadata={'input_tokens': 100, 'output_tokens': 7})
    monkeypatch.setattr(module, 'build_chat_model', lambda _: SimpleNamespace(invoke=invoke))
    agent = agent_class(replace(make_config(tmp_path), live_mode=True))
    agent.reply('u', 'a', 'Mình tên là Lan.')
    response = agent.reply('u', 'b', 'Mình tên gì?')
    assert response['mode'] == 'live'
    assert response['tokens'] == 7 and response['prompt_tokens'] == 100
    second_prompt = json.dumps(prompts[-1], ensure_ascii=False)
    assert ('Lan' in second_prompt) == (agent_class is AdvancedAgent)


def test_force_offline_never_constructs_live_model(tmp_path: Path, monkeypatch) -> None:
    for module_name, agent_class in [('agent_baseline', BaselineAgent), ('agent_advanced', AdvancedAgent)]:
        module = __import__(module_name)
        def unexpected_call(_):
            raise AssertionError('Offline must not construct a live model')
        monkeypatch.setattr(module, 'build_chat_model', unexpected_call)
        agent = agent_class(replace(make_config(tmp_path), live_mode=True), force_offline=True)
        assert agent.reply('u', 't', 'Xin chào')['mode'] == 'offline'
