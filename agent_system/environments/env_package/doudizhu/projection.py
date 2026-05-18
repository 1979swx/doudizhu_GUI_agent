import ast
import json
import re
from typing import Any, Dict, List, Tuple


LEGACY_TAG_NAMES = ("plan", "action", "chat", "memory")
TOOL_CALL_TAG_NAMES = ("plan", "action", "tool_call", "chat", "memory")
CLICK_ACTIONS = {"left_click", "click"}


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


def _load_literal(text: str):
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _parse_coordinate_pair(coordinate) -> Tuple[List[float], bool]:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        return [], False
    x, y = coordinate
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return [], False
    if not (0 <= float(x) <= 1000 and 0 <= float(y) <= 1000):
        return [], False
    return [float(x), float(y)], True


def _parse_clicks(action_text: str, max_clicks: int) -> Tuple[List[List[float]], bool]:
    parsed = _load_literal(action_text)
    if parsed is None:
        return [], False

    if not isinstance(parsed, (list, tuple)):
        return [], False
    if len(parsed) == 0 or len(parsed) > max_clicks:
        return [], False

    clicks = []
    for item in parsed:
        click, valid = _parse_coordinate_pair(item)
        if not valid:
            return [], False
        clicks.append(click)
    return clicks, True


def _normalize_tool_calls(parsed):
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        return parsed["tool_calls"]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def _tool_call_payload(call):
    if not isinstance(call, dict):
        return None, None

    if isinstance(call.get("function"), dict):
        function = call["function"]
        return function.get("name"), function.get("arguments", {})

    return call.get("name"), call.get("arguments", {})


def _parse_tool_calls(tool_call_text: str, max_clicks: int) -> Tuple[List[List[float]], List[Dict[str, Any]], bool]:
    parsed = _load_literal(tool_call_text)
    calls = _normalize_tool_calls(parsed)
    if len(calls) == 0 or len(calls) > max_clicks:
        return [], [], False

    clicks = []
    normalized_calls = []
    for call in calls:
        name, arguments = _tool_call_payload(call)
        if isinstance(arguments, str):
            arguments = _load_literal(arguments)
        if name != "computer_use" or not isinstance(arguments, dict):
            return [], [], False
        if arguments.get("action") not in CLICK_ACTIONS:
            return [], [], False
        click, coordinate_valid = _parse_coordinate_pair(arguments.get("coordinate"))
        if not coordinate_valid:
            return [], [], False
        normalized_call = {
            "name": "computer_use",
            "arguments": {
                "action": "left_click",
                "coordinate": click,
            },
        }
        clicks.append(click)
        normalized_calls.append(normalized_call)

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
            clicks, tool_calls, action_valid = _parse_tool_calls(extracted["tool_call"] or "", max_clicks=max_clicks)
            valid = bool(action_valid)
        else:
            extracted = {tag: extract_tag(response, tag) for tag in LEGACY_TAG_NAMES}
            has_legacy_format = all(extracted[tag] is not None and extracted[tag] != "" for tag in LEGACY_TAG_NAMES)
            clicks, action_valid = _parse_clicks(extracted["action"] or "", max_clicks=max_clicks)
            tool_calls = [
                {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": click}}
                for click in clicks
            ] if action_valid else []
            valid = bool(has_legacy_format and action_valid)

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
