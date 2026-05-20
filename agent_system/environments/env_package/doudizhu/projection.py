import re
from typing import Any, Dict, List, Tuple


TOOL_CALL_TAG_NAMES = ("plan", "action", "tool_call", "chat", "memory")
TOOL_CALL_PATTERN = re.compile(r"^\s*left_click\((?P<args>.*)\)\s*$", flags=re.DOTALL)
COORDINATE_PATTERN = re.compile(r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]")


def extract_tag(text: str, tag: str):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _empty_action(raw_text: Any = "") -> Dict[str, Any]:
    return {
        "clicks": [],
        "plan": "",
        "semantic_action": "",
        "chat": "",
        "memory": "",
        "raw_action_text": "",
        "raw_tool_call_text": "",
        "tool_calls": [],
        "tool_calling": 0,
        "raw_response": raw_text if isinstance(raw_text, str) else "",
        "projection_valid": 0,
    }


def _parse_coordinate_pair(coordinate) -> Tuple[List[float], bool]:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        return [], False
    x, y = coordinate
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return [], False
    if not (0 <= float(x) <= 1000 and 0 <= float(y) <= 1000):
        return [], False
    return [float(x), float(y)], True


def _parse_left_click_args(args_text: str, max_clicks: int) -> Tuple[List[List[float]], bool]:
    args_text = args_text.strip()
    if not args_text:
        return [], False

    clicks = []
    pos = 0
    while pos < len(args_text):
        match = COORDINATE_PATTERN.match(args_text, pos)
        if match is None:
            return [], False
        click, coordinate_valid = _parse_coordinate_pair([float(match.group(1)), float(match.group(2))])
        if not coordinate_valid:
            return [], False
        clicks.append(click)
        if len(clicks) > max_clicks:
            return [], False

        pos = match.end()
        while pos < len(args_text) and args_text[pos].isspace():
            pos += 1
        if pos == len(args_text):
            break
        if args_text[pos] != ",":
            return [], False
        pos += 1
        while pos < len(args_text) and args_text[pos].isspace():
            pos += 1
        if pos == len(args_text):
            return [], False

    return clicks, bool(clicks)


def parse_left_click_tool_call(tool_call_text: str, max_clicks: int) -> Tuple[List[List[float]], List[Dict[str, Any]], bool]:
    match = TOOL_CALL_PATTERN.match(tool_call_text)
    if match is None:
        return [], [], False

    clicks, valid = _parse_left_click_args(match.group("args"), max_clicks=max_clicks)
    if not valid:
        return [], [], False

    normalized_calls = [
        {
            "name": "computer_use",
            "arguments": {
                "action": "left_click",
                "coordinate": click,
            },
        }
        for click in clicks
    ]

    return clicks, normalized_calls, True


def doudizhu_projection(text_actions: List[str], max_clicks: int = 12):
    structured_actions = []
    valids = []

    for response in text_actions:
        if not isinstance(response, str):
            structured_actions.append(_empty_action(response))
            valids.append(0)
            continue

        extracted = {tag: extract_tag(response, tag) for tag in TOOL_CALL_TAG_NAMES}
        has_tool_call_format = all(extracted[tag] is not None and extracted[tag] != "" for tag in TOOL_CALL_TAG_NAMES)

        if has_tool_call_format:
            clicks, tool_calls, action_valid = parse_left_click_tool_call(extracted["tool_call"] or "", max_clicks=max_clicks)
        else:
            clicks, tool_calls, action_valid = [], [], False
        valid = bool(action_valid)

        action = {
            "clicks": clicks if action_valid else [],
            "plan": extracted["plan"] or "",
            "semantic_action": extracted["action"] or "",
            "chat": extracted["chat"] or "",
            "memory": extracted["memory"] or "",
            "raw_action_text": extracted["action"] or "",
            "raw_tool_call_text": extracted.get("tool_call") or "",
            "tool_calls": tool_calls if action_valid else [],
            "tool_calling": len(tool_calls) if action_valid else 0,
            "raw_response": response,
            "projection_valid": int(valid),
        }
        structured_actions.append(action)
        valids.append(int(valid))

    return structured_actions, valids
