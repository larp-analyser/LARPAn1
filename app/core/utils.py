import re

def sanitize_think_tags(text: str) -> str:
    """Comprehensive think-tag sanitization matching the old 5-regex chain."""
    if not text:
        return ""
    clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\|?think\|?>.*?</\|?think\|?>\s*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\|?think\|?>.*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\|channel>thought.*?</channel\|>\s*", "", clean, flags=re.DOTALL | re.IGNORECASE)
    return clean.strip()

def trim_history_by_tokens(history_docs: list, max_tokens: int) -> list:
    """Trim history to fit within a token budget using word-count estimation."""
    total = 0
    trimmed = []
    for m in reversed(history_docs):
        sender = m.get("username") or m.get("sender") or m.get("role") or "User"
        formatted = f"[{sender}]: {m.get('content', '')}"
        estimated_tokens = int(len(formatted.split()) * 1.5)
        if total + estimated_tokens > max_tokens:
            break
        trimmed.insert(0, m)
        total += estimated_tokens
    return trimmed
