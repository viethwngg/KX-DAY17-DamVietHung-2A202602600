"""Shared deterministic response policy, so the agents differ in memory only."""
from __future__ import annotations

import re
from typing import Any

from memory_store import FACT_LABELS, estimate_tokens


SYSTEM_PROMPT = (
    'Bạn là trợ lý tiếng Việt. Chỉ nhắc lại thông tin cá nhân có trong ngữ cảnh; '
    'nếu chưa biết thì nói rõ. Ưu tiên bản đính chính mới nhất, bỏ qua câu đùa '
    'và địa điểm đi họp. Tuân theo style trả lời của người dùng.'
)


def offline_response(message: str, facts: dict[str, str], summary: str = '') -> str:
    lower = message.lower()
    recall = bool(re.search(r'nhắc lại|nhớ lại|tên (?:mình|gì)|mình tên gì|nghề (?:gì|hiện)|'
                            r'hiện tại.*(?:ở đâu|đang ở|làm nghề)|yêu thích.*(?:gì|của mình)|'
                            r'mình.*nuôi con gì|bạn.*biết|tóm tắt.*mình|style.*mình thích|'
                            r'hiện đang ở đâu|nhắc.*nghề nghiệp', lower))
    if recall:
        wanted = []
        triggers = {
            'name': ('tên', 'là ai', 'biết'),
            'location': ('ở đâu', 'nơi ở', 'đang ở', 'còn ở', 'huế', 'hà nội'),
            'profession': ('nghề', 'là ai'),
            'response_style': ('style', 'kiểu trả lời', 'trả lời như thế nào'),
            'favorite_drink': ('đồ uống',), 'favorite_food': ('món ăn',),
            'pet': ('nuôi', 'con gì', 'thú cưng'),
            'interests': ('quan tâm', 'kỹ thuật chính', 'là ai'),
        }
        for key, words in triggers.items():
            if any(word in lower for word in words):
                wanted.append(key)
        if not wanted:
            wanted = list(facts)
        lines = [f'{FACT_LABELS[key]}: {facts[key]}.' if key in facts
                 else f'{FACT_LABELS[key]}: mình chưa có thông tin này.' for key in wanted]
        return _format(lines or ['Mình chưa có thông tin về bạn trong ngữ cảnh hiện tại.'], facts)
    if re.search(r'tóm tắt|nhớ.*chủ đề|bốn tin|roadmap|pattern', lower) and summary:
        topics = [line.removeprefix('- context: ') for line in summary.splitlines()
                  if line.startswith('- context: ')]
        if topics:
            return _format(topics[-3:], facts)
    if re.search(r'benchmark|memory|compact|token|recall', lower):
        lines = ['Profile giữ fact ổn định; lịch sử gần nhất hỗ trợ câu hỏi tiếp nối.',
                 'Compact giảm ngữ cảnh lặp lại, nhưng có thể mất chi tiết của hội thoại cũ.',
                 'Ví dụ thực chiến: so recall và prompt tokens trên cùng một bộ dữ liệu.']
    else:
        lines = ['Mình đã ghi nhận thông tin bạn vừa chia sẻ.',
                 'Mình sẽ ưu tiên thông tin được đính chính gần nhất.',
                 'Bạn có thể hỏi lại các ý đã chia sẻ trong ngữ cảnh.']
    return _format(lines, facts)


def _format(lines: list[str], facts: dict[str, str]) -> str:
    style = facts.get('response_style', '')
    if '3 bullet' in style:
        groups = [' '.join(lines[i::3]) for i in range(3)]
        fillers = ['Mình chỉ nhắc lại các fact đã biết.',
                   'Ví dụ thực chiến: dùng profile để giữ fact qua phiên mới.',
                   'Trade-off: recall tốt hơn đi kèm chi phí lưu và đọc memory.']
        return '\n'.join(f'- {group or fillers[i]}' for i, group in enumerate(groups))
    if 'bullet' in style:
        return '\n'.join(f'- {line}' for line in lines)
    return ' '.join(lines)


def prompt_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(item['content']) + 4 for item in messages)


def model_answer(model: Any, messages: list[dict[str, str]]) -> tuple[str, int, int]:
    result = model.invoke(messages)
    content = result.content
    if isinstance(content, list):
        content = '\n'.join(block if isinstance(block, str) else block.get('text', '') for block in content)
    answer = str(content)
    usage = getattr(result, 'usage_metadata', None) or {}
    return answer, int(usage.get('output_tokens', estimate_tokens(answer))), int(usage.get('input_tokens', prompt_tokens(messages)))
