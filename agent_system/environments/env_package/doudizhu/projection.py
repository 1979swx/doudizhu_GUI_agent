import re
from typing import Any, Dict, List, Tuple


TOOL_CALL_TAG_NAMES = ("plan", "action", "tool_call", "chat", "memory")
TOOL_CALL_PATTERN = re.compile(r"^\s*left_click\((?P<args>.*)\)\s*$", flags=re.DOTALL)
COORDINATE_PATTERN = re.compile(r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]")
ACTION_CARD_TO_INTERNAL = {
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "T": "T",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
    "2": "2",
    "BJ": "B",
    "B": "B",
    "RJ": "R",
    "R": "R",
}
ACTION_INTERNAL_TO_CARD = {
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "10",
    "J": "J",
    "Q": "Q",
    "K": "K",
    "A": "A",
    "2": "2",
    "B": "BJ",
    "R": "RJ",
}
ACTION_PASS_ALIASES = {"pass", "不要", "不出", "过", "过牌"}


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
        "normalized_action_text": "",
        "action_tag_parse_ok": False,
        "action_tag_parse_error": "",
        "action_tag_raw": "",
        "action_tag_cards": [],
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


def _strip_optional_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _format_action_list(cards: List[str]) -> str:
    if cards == ["pass"]:
        return "[pass]"
    return "[" + ", ".join(cards) + "]"


def parse_doudizhu_action_tag(action_text: str) -> Dict[str, Any]:
    """Parse the <action> tag list-of-card contract.

    The model-facing contract uses display ranks such as 10/BJ/RJ, while the
    game environment uses T/B/R internally.
    """
    result: Dict[str, Any] = {
        "parse_ok": False,
        "raw": "",
        "cards": [],
        "normalized_text": "",
        "error": "",
    }
    if not isinstance(action_text, str):
        result["error"] = "action_not_string"
        return result

    raw = action_text.strip()
    if not raw:
        result["error"] = "action_empty"
        return result
    if not (raw.startswith("[") and raw.endswith("]")):
        result["error"] = "action_not_list"
        return result

    inner = raw[1:-1].strip().replace("，", ",")
    if not inner:
        result["error"] = "action_list_empty"
        return result

    tokens = [_strip_optional_quotes(item) for item in inner.split(",")]
    if any(not token for token in tokens):
        result["error"] = "action_empty_item"
        return result

    lower_tokens = [token.lower() for token in tokens]
    pass_flags = [token in ACTION_PASS_ALIASES for token in lower_tokens]
    if any(pass_flags):
        if len(tokens) != 1 or not pass_flags[0]:
            result["error"] = "pass_mixed_with_cards"
            return result
        result.update(
            {
                "parse_ok": True,
                "raw": "pass",
                "cards": ["pass"],
                "normalized_text": "[pass]",
                "error": "",
            }
        )
        return result

    internal_cards = []
    display_cards = []
    for token in tokens:
        normalized = token.upper()
        if normalized not in ACTION_CARD_TO_INTERNAL:
            result["error"] = f"invalid_card:{token}"
            return result
        internal = ACTION_CARD_TO_INTERNAL[normalized]
        internal_cards.append(internal)
        display_cards.append(ACTION_INTERNAL_TO_CARD[internal])

    result.update(
        {
            "parse_ok": True,
            "raw": "".join(internal_cards),
            "cards": display_cards,
            "normalized_text": _format_action_list(display_cards),
            "error": "",
        }
    )
    return result


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
        action_parse = parse_doudizhu_action_tag(extracted["action"] or "") if has_tool_call_format else {
            "parse_ok": False,
            "raw": "",
            "cards": [],
            "normalized_text": "",
            "error": "missing_required_tags",
        }

        if has_tool_call_format:
            clicks, tool_calls, action_valid = parse_left_click_tool_call(extracted["tool_call"] or "", max_clicks=max_clicks)
        else:
            clicks, tool_calls, action_valid = [], [], False
        valid = bool(action_valid and action_parse["parse_ok"])

        action = {
            "clicks": clicks if action_valid else [],
            "plan": extracted["plan"] or "",
            "semantic_action": action_parse["raw"] if action_parse["parse_ok"] else "",
            "chat": extracted["chat"] or "",
            "memory": extracted["memory"] or "",
            "raw_action_text": extracted["action"] or "",
            "normalized_action_text": action_parse["normalized_text"],
            "action_tag_parse_ok": bool(action_parse["parse_ok"]),
            "action_tag_parse_error": action_parse["error"],
            "action_tag_raw": action_parse["raw"],
            "action_tag_cards": action_parse["cards"],
            "raw_tool_call_text": extracted.get("tool_call") or "",
            "tool_calls": tool_calls if action_valid else [],
            "tool_calling": len(tool_calls) if action_valid else 0,
            "raw_response": response,
            "projection_valid": int(valid),
        }
        structured_actions.append(action)
        valids.append(int(valid))

    return structured_actions, valids
