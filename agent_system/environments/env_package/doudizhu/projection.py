import ast
import json
import re
from typing import Any, Dict, List, Tuple


TAG_NAMES = ("think", "action", "chat", "memory")


def extract_tag(text: str, tag: str):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _empty_action(raw_text: Any = "") -> Dict[str, Any]:
    return {
        "clicks": [],
        "think": "",
        "chat": "",
        "memory": "",
        "raw_action_text": "",
        "raw_response": raw_text if isinstance(raw_text, str) else "",
        "projection_valid": 0,
    }


def _parse_clicks(action_text: str, max_clicks: int) -> Tuple[List[List[float]], bool]:
    try:
        parsed = json.loads(action_text)
    except Exception:
        try:
            parsed = ast.literal_eval(action_text)
        except Exception:
            return [], False

    if not isinstance(parsed, (list, tuple)):
        return [], False
    if len(parsed) == 0 or len(parsed) > max_clicks:
        return [], False

    clicks = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return [], False
        x, y = item
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return [], False
        if not (1 <= float(x) <= 1000 and 1 <= float(y) <= 1000):
            return [], False
        clicks.append([float(x), float(y)])
    return clicks, True


def doudizhu_projection(text_actions: List[str], max_clicks: int = 8):
    structured_actions = []
    valids = []

    for response in text_actions:
        if not isinstance(response, str):
            structured_actions.append(_empty_action(response))
            valids.append(0)
            continue

        extracted = {tag: extract_tag(response, tag) for tag in TAG_NAMES}
        valid = all(extracted[tag] is not None and extracted[tag] != "" for tag in TAG_NAMES)
        clicks, action_valid = _parse_clicks(extracted["action"] or "", max_clicks=max_clicks)
        valid = bool(valid and action_valid)

        action = {
            "clicks": clicks if action_valid else [],
            "think": extracted["think"] or "",
            "chat": extracted["chat"] or "",
            "memory": extracted["memory"] or "",
            "raw_action_text": extracted["action"] or "",
            "raw_response": response,
            "projection_valid": int(valid),
        }
        structured_actions.append(action)
        valids.append(int(valid))

    return structured_actions, valids
