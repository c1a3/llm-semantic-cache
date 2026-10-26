"""Prompt normalization and canonicalization utilities.
Ensures consistent hash keys and clean input text for embedding inference.
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple


def normalize_text(text: str) -> str:
    """Strip unnecessary whitespace, normalize punctuation and lowercases.
    Preserves meaningful semantic tokens while discarding formatting artifacts.
    """
    if not text:
        return ""
    # Collapse multiple whitespace/newlines to single space
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_prompt_representations(
    messages: List[Dict[str, Any]], 
    temperature: float = 0.7
) -> Tuple[str, str, str]:
    """Extracts:
    1. exact_key: SHA-256 hash including system instructions + all message history + temperature
    2. query_text: The latest user message text (normalized for semantic embedding)
    3. full_context: Concatenated conversation context
    """
    system_prompt = ""
    latest_user_query = ""
    history_segments = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multimodal/structured content list
            content = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        content_str = str(content)

        if role == "system":
            system_prompt = content_str
        elif role == "user":
            latest_user_query = content_str
        
        history_segments.append(f"{role}:{content_str}")

    full_context = "\n".join(history_segments)

    # Exact key: Deterministic hash of system prompt, conversation history, and temperature
    exact_payload = f"temp={temperature:.2f}|sys={system_prompt}|history={full_context}"
    exact_key = hashlib.sha256(exact_payload.encode("utf-8")).hexdigest()

    # Query text for semantic embedding
    query_text = normalize_text(latest_user_query)

    return exact_key, query_text, full_context
