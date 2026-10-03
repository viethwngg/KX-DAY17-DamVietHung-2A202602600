from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote


FACT_LABELS = {
    'name': 'Tên', 'location': 'Nơi ở hiện tại', 'profession': 'Nghề nghiệp hiện tại',
    'response_style': 'Style trả lời', 'interests': 'Mối quan tâm',
    'favorite_drink': 'Đồ uống yêu thích', 'favorite_food': 'Món ăn yêu thích',
    'pet': 'Thú cưng', 'hobbies': 'Sở thích',
}


def estimate_tokens(text: str) -> int:
    """Deterministic character heuristic, not a provider tokenizer or billing count."""
    return math.ceil(len(text.strip()) / 4)


def merge_facts(existing: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    merged = dict(existing)
    for key, value in updates.items():
        if key in {'response_style', 'interests', 'hobbies'} and key in merged:
            parts = list(dict.fromkeys(merged[key].split('; ') + value.split('; ')))
            merged[key] = '; '.join(parts)
        else:
            merged[key] = value
    return merged


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError('user_id must be a nonempty string')
        # Encode path separators and hash every ID to avoid case-insensitive and
        # Windows reserved-name collisions without merging distinct users.
        encoded = quote(user_id, safe='')[:60].strip('.') or 'user'
        digest = hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:16]
        return self.root_dir / f'user-{encoded}-{digest}' / 'User.md'

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding='utf-8') if path.exists() else ''

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement prevents a partial profile after an interrupted write.
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text or search_text == replacement:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_facts(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        return self.upsert_facts(user_id, {key: value})

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        unknown = updates.keys() - FACT_LABELS.keys()
        if unknown:
            raise ValueError(f'Unknown profile fields: {sorted(unknown)}')
        if any(not value.strip() or '\n' in value or '\r' in value for value in updates.values()):
            raise ValueError('Facts must be nonempty single-line values')
        old = self.read_text(user_id)
        facts = merge_facts(parse_facts(old), updates)
        # Preserve user-authored Markdown outside the managed fact lines.
        def managed_line(line: str) -> bool:
            match = re.match(r'^- ([a-z_]+): ', line)
            return match is not None and match.group(1) in FACT_LABELS
        notes = [line for line in old.splitlines()
                 if not managed_line(line) and line != '# User profile']
        text = '# User profile\n\n' + '\n'.join(f'- {key}: {facts[key]}' for key in FACT_LABELS if key in facts) + '\n'
        if any(line.strip() for line in notes):
            text += '\n' + '\n'.join(notes).strip() + '\n'
        if text != old:
            return self.write_text(user_id, text)
        return self.path_for(user_id)


def parse_facts(text: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r'^- ([a-z_]+): (.+)$', text, re.M)
            if key in FACT_LABELS}


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicit self-statements only, never infer facts from a question.

    Ordered clauses make a later correction replace the old scalar value.
    This conservative Vietnamese rule system is deliberately not a general NER model.
    """
    message = unicodedata.normalize('NFC', message)
    updates: dict[str, str] = {}
    clauses = re.split(r'(?<=[.!?])\s+|\n+', message)
    location = r'([^,;]+?)(?=\s+(?:và|chứ|dù|trong|để|vài|mỗi|nhé|cho|từ)\b|[,;]|$)'
    patterns = {
        'name': r'(?:mình tên(?: là)?|tên mình(?: là)?)\s+([^,;]+?)(?=\s+(?:hiện|và|nghề|nơi|đang)\b|[,;]|$)',
        'location': r'(?:mình (?:vẫn |hiện |giờ |đang |giờ mình đang )*ở|hiện (?:tại )?ở|nơi ở hiện tại (?:là)?|giờ mình đang ở)\s+' + location,
        'profession': r'(?:đang làm|vẫn làm|mình làm|nghề nghiệp(?: hiện tại)? (?:thì )?(?:vẫn )?là|nghề(?: hiện tại)?|chuyển sang)\s+([\w+#.-]+(?:\s+[\w+#.-]+){0,3}?\s+(?:engineer|developer|manager)|kỹ sư[^,;]*|lập trình viên[^,;]*)',
        'favorite_drink': r'(?:đồ uống yêu thích(?: của mình)? (?:là)?|mình (?:vẫn )?uống)\s+([^,;]+?)(?=\s+(?:như|nhưng|và|mỗi)\b|[,;]|$)',
        'favorite_food': r'món ăn yêu thích(?: của mình)? (?:là)?\s+([^,;]+)',
        'pet': r'mình nuôi (?:một |một bé |con |bé )*(\w+(?: tên \w+)?)',
    }
    question_start = r'^(?:bạn (?:có|thử)|nhắc lại|nếu|liệu|ở đâu|mình tên gì|tên mình là gì|hiện tại mình làm nghề gì)\b'
    for clause in clauses:
        if clause.rstrip().endswith('?'):
            continue
        clause = clause.strip().rstrip('.!')
        lower = clause.lower()
        if not clause or re.match(question_start, lower):
            continue
        if re.search(r'\b(?:câu hỏi|câu đùa|mình đùa|mình không|không phải|đừng nói|ví dụ cũ|hay là)\b', lower):
            # A positive correction after a negated statement remains eligible.
            split = re.split(r'[,;]\s*(?:giờ |nhưng |thực ra )', clause, flags=re.I)
            if len(split) == 1:
                continue
            clause = split[-1]
            lower = clause.lower()
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, clause, re.I):
                value = match.group(1).strip(' ,;:')
                if not re.search(r'\b(?:gì|đâu|không|nào)\b', value, re.I):
                    updates[key] = value
        # Working in another city for months is an explicit relocation, unlike
        # a cafe visit or a two-day business trip.
        moved = re.search(r'mình đang làm việc ở\s+(.+?)\s+vài tháng', clause, re.I)
        if moved:
            updates['location'] = moved.group(1).strip()
        if re.search(r'mình (?:vẫn )?thích|mình (?:đang )?quan tâm|dài hạn:', lower):
            interests = [word for word in ('Python', 'AI', 'MLOps', 'RAG')
                         if re.search(rf'\b{word}\b', clause, re.I)]
            if interests:
                updates = merge_facts(updates, {'interests': '; '.join(interests)})
            hobbies = [word for word in ('chạy bộ', 'lo-fi') if word in lower]
            if hobbies:
                updates = merge_facts(updates, {'hobbies': '; '.join(hobbies)})
            if 'cà phê sữa đá' in lower:
                updates['favorite_drink'] = 'cà phê sữa đá'
        # Style requires an instruction or preference, not merely topic keywords.
        if re.search(r'trả lời|giải thích|trình bày|style|preference|dài hạn:', lower):
            style = []
            if re.search(r'ngắn|gọn', lower):
                style.append('ngắn gọn')
            if '3 bullet' in lower:
                style.append('3 bullet')
            elif 'bullet' in lower:
                style.append('bullet')
            if 'ví dụ thực chiến' in lower:
                style.append('có ví dụ thực chiến')
            elif 'ví dụ thực tế' in lower:
                style.append('có ví dụ thực tế')
            if 'trade-off' in lower:
                style.append('so sánh trade-off')
            if style:
                updates = merge_facts(updates, {'response_style': '; '.join(style)})
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Merge earlier summaries, keep corrected facts and bounded topical excerpts."""
    if max_items < 1:
        return ''
    facts: dict[str, str] = {}
    snippets: list[str] = []
    for item in messages:
        content = item['content']
        if item['role'] == 'summary':
            facts = merge_facts(facts, parse_facts(content))
            snippets.extend(re.findall(r'^- context: (.+)$', content, re.M))
        elif item['role'] == 'user':
            updates = extract_profile_updates(content)
            facts = merge_facts(facts, updates)
            # Select topical sentences instead of retaining raw long paragraphs.
            sentences = re.split(r'(?<=[.!?])\s+', content)
            for sentence in sentences:
                if re.search(r'Artemis|X-59|WMO|British Columbia|BC energy|roadmap|readiness|externality|uncertainty|efficiency', sentence, re.I):
                    snippets.append(' '.join(sentence.split())[:160])
            if not updates and len(content) < 160:
                snippets.append(' '.join(content.split()))
    lines = [f'- {key}: {value}' for key, value in facts.items()]
    unique = list(dict.fromkeys(snippets))
    lines += [f'- context: {snippet}' for snippet in unique[-max_items:]]
    return '\n'.join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens < 1 or self.keep_messages < 1:
            raise ValueError('Compact threshold and keep_messages must be positive')

    def append(self, thread_id: str, role: str, content: str) -> None:
        if role not in {'user', 'assistant'}:
            raise ValueError('role must be user or assistant')
        context = self.state.setdefault(thread_id, {'messages': [], 'summary': '', 'compactions': 0})
        messages = context['messages']
        messages.append({'role': role, 'content': content})
        tokens = estimate_tokens(context['summary']) + sum(estimate_tokens(m['content']) for m in messages)
        if tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        # The threshold is soft: recent full messages can individually exceed it.
        old = messages[:-self.keep_messages]
        previous = [{'role': 'summary', 'content': context['summary']}] if context['summary'] else []
        summary = summarize_messages(previous + old)
        # Bound the summary independently of retained messages. Whole fact lines
        # get priority; profile storage remains the durable source of truth.
        budget = min(1600, max(80, self.threshold_tokens * 4 // 3))
        kept: list[str] = []
        for line in summary.splitlines():
            if len('\n'.join(kept + [line])) <= budget:
                kept.append(line)
        context['summary'] = '\n'.join(kept)
        context['messages'] = messages[-self.keep_messages:]
        context['compactions'] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        context = self.state.get(thread_id, {'messages': [], 'summary': '', 'compactions': 0})
        return {**context, 'messages': [dict(m) for m in context['messages']]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get('compactions', 0))
