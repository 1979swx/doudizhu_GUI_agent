from typing import Any, Dict, List

from agent_system.environments.env_package.doudizhu.projection import (
    extract_tag,
    parse_left_click_tool_call,
)


def _empty_action(raw_text: Any = "") -> Dict[str, Any]:
    return {
        "clicks": [],
        "plan": "",
        "raw_tool_call_text": "",
        "tool_calls": [],
        "tool_calling": 0,
        "raw_response": raw_text if isinstance(raw_text, str) else "",
        "projection_valid": 0,
    }


def doudizhu_grounding_projection(text_actions: List[str], max_clicks: int = 12):
    structured_actions = []
    valids = []

    for response in text_actions:
        if not isinstance(response, str):
            structured_actions.append(_empty_action(response))
            valids.append(0)
            continue

        plan = extract_tag(response, "plan")
        tool_call_text = extract_tag(response, "tool_call")
        has_required_tags = bool(plan) and bool(tool_call_text)
        if has_required_tags:
            clicks, tool_calls, action_valid = parse_left_click_tool_call(tool_call_text or "", max_clicks=max_clicks)
        else:
            clicks, tool_calls, action_valid = [], [], False

        valid = bool(action_valid)
        structured_actions.append(
            {
                "clicks": clicks if action_valid else [],
                "plan": plan or "",
                "raw_tool_call_text": tool_call_text or "",
                "tool_calls": tool_calls if action_valid else [],
                "tool_calling": len(tool_calls) if action_valid else 0,
                "raw_response": response,
                "projection_valid": int(valid),
            }
        )
        valids.append(int(valid))

    return structured_actions, valids
