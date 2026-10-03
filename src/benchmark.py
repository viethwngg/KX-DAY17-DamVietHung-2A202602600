from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(conversations, list):
        raise ValueError('Dataset must contain a list of conversations')
    for conversation in conversations:
        if not all(key in conversation for key in ('id', 'user_id', 'turns', 'recall_questions')):
            raise ValueError('Incomplete conversation')
        if not all(isinstance(turn, str) for turn in conversation['turns']):
            raise ValueError('Each turn must be text')
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected substrings found, including partial recall."""
    if not expected:
        return 1.0
    normalized = unicodedata.normalize('NFC', answer).casefold()
    return sum(unicodedata.normalize('NFC', fact).casefold() in normalized for fact in expected) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline proxy: factual coverage + brevity + a nonempty answer, not an LLM judge."""
    if not answer.strip():
        return 0.0
    return 0.8 * recall_points(answer, expected) + 0.1 * (len(answer) <= 800) + 0.1


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    users = {item['user_id'] for item in conversations}
    size = lambda user: agent.memory_file_size(user) if hasattr(agent, 'memory_file_size') else 0
    initial_size = sum(size(user) for user in users)
    threads: list[str] = []
    recalls: list[float] = []
    qualities: list[float] = []
    # Train and evaluate sequentially: corrections from future conversations must
    # never influence an earlier recall question (e.g. Huế vs Đà Nẵng).
    for index, conversation in enumerate(conversations):
        user = conversation['user_id']
        thread = f'{agent_name}:train:{index}:{conversation["id"]}'
        threads.append(thread)
        for turn in conversation['turns']:
            agent.reply(user, thread, turn)
        for question_index, question in enumerate(conversation['recall_questions']):
            recall_thread = f'{agent_name}:recall:{index}:{question_index}'
            threads.append(recall_thread)
            answer = agent.reply(user, recall_thread, question['question'])['response']
            recalls.append(recall_points(answer, question['expected_contains']))
            qualities.append(heuristic_quality(answer, question['expected_contains']))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread) for thread in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread) for thread in threads),
        recall_score=sum(recalls) / len(recalls) if recalls else 0.0,
        response_quality=sum(qualities) / len(qualities) if qualities else 0.0,
        memory_growth_bytes=sum(size(user) for user in users) - initial_size,
        compactions=sum(agent.compaction_count(thread) for thread in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ['Agent', 'Agent tokens only', 'Prompt tokens processed',
               'Cross-session recall', 'Response quality', 'Memory growth (bytes)', 'Compactions']
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        cells = [row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
                 f'{row.recall_score:.1%}', f'{row.response_quality:.1%}',
                 str(row.memory_growth_bytes), str(row.compactions)]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare short-term and persistent/compact memory')
    parser.add_argument('--live', action='store_true', help='Explicitly call the configured real model')
    parser.add_argument('--json-output', type=Path, help='Save measured metrics as JSON')
    args = parser.parse_args()
    config = load_config()
    config = replace(config, live_mode=args.live)
    suites = [('Standard Benchmark', 'conversations.json'),
              ('Long-Context Stress Benchmark', 'advanced_long_context.json')]
    results = {}
    print('Mode: ' + ('live' if args.live else 'offline (deterministic)'))
    print('Offline tokens are estimates; quality is a heuristic, not an LLM judge.\n')
    for title, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        # Isolate each run without deleting or reading the user's real profiles.
        with TemporaryDirectory(prefix='benchmark-', dir=config.state_dir) as directory:
            isolated = replace(config, state_dir=Path(directory))
            rows = [run_agent_benchmark('Baseline', BaselineAgent(isolated, force_offline=not args.live), conversations, isolated),
                    run_agent_benchmark('Advanced', AdvancedAgent(isolated, force_offline=not args.live), conversations, isolated)]
        results[title] = [asdict(row) for row in rows]
        print(title)
        print(format_rows(rows))
        print()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
