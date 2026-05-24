from agent_system.environments.env_package.doudizhu.projection import parse_doudizhu_action_tag
from data_synthesis.doudizhu_end_to_end_sft import (
    DOUDIZHU_VISUAL_TEMPLATE_ZH,
    FilterConfig,
    build_prompt,
    filter_reasons_for_step,
    first_episode_index,
    normalized_answer,
)


def _accepted_step() -> dict:
    return {
        "api": {"ok": True},
        "image_path": "raw/images/train/example.png",
        "projection": {"projection_valid": 1},
        "env_info": {
            "click_valid_ratio": 1.0,
            "rule_action_valid": 1.0,
            "fallback_used": False,
            "game_action": "3TBR",
            "submit_kind": "play",
        },
        "action_parse": {"parse_ok": True, "raw": "3TBR"},
        "tokens": {
            "prompt_tokens_target_model": 100,
            "response_tokens_target_model": 100,
            "full_sequence_tokens_target_model": 400,
        },
    }


def test_action_tag_parser_accepts_display_contract():
    assert parse_doudizhu_action_tag("[pass]")["raw"] == "pass"
    assert parse_doudizhu_action_tag('["3", "3"]')["raw"] == "33"
    assert parse_doudizhu_action_tag("[9, 10, J, Q, K]")["raw"] == "9TJQK"
    assert parse_doudizhu_action_tag("[BJ, RJ]")["raw"] == "BR"
    assert parse_doudizhu_action_tag("[不要]")["normalized_text"] == "[pass]"


def test_normalized_answer_rewrites_only_action_tag():
    raw = "<plan>keep [10] here</plan><action>[T]</action><tool_call>left_click([1,2])</tool_call><chat>ok</chat><memory>played T</memory>"
    assert normalized_answer(raw, "[10]") == ("<plan>keep [10] here</plan><action>[10]</action><tool_call>left_click([1,2])</tool_call><chat>ok</chat><memory>played T</memory>")


def test_build_prompt_uses_original_environment_template_without_collection_suffix():
    memory = "上一轮选择了[pass]"
    prompt = build_prompt("zh", memory)
    assert prompt == DOUDIZHU_VISUAL_TEMPLATE_ZH.format(previous_memory=memory)
    assert "数据采集额外约束" not in prompt


def test_first_episode_index_skips_existing_resume_seeds():
    existing = [{"episode_seed": 20260524}, {"episode_seed": 20260527}]
    assert first_episode_index(20260524, existing) == 4


def test_filter_accepts_strict_valid_winning_step():
    step = _accepted_step()
    episode = {"won": True, "normal_end": True, "final_player0_num_cards_left": 0}
    assert filter_reasons_for_step(step, episode, FilterConfig()) == []


def test_filter_accepts_near_terminal_remaining_episode():
    step = _accepted_step()
    episode = {"won": False, "normal_end": True, "final_player0_num_cards_left": 2}
    assert filter_reasons_for_step(step, episode, FilterConfig(terminal_max_player0_hand=2)) == []


def test_filter_with_zero_terminal_remaining_matches_strict_win_mode():
    step = _accepted_step()
    episode = {"won": False, "normal_end": True, "final_player0_num_cards_left": 2}
    assert "terminal_player0_hand_gt_threshold" in filter_reasons_for_step(step, episode, FilterConfig(terminal_max_player0_hand=0))


def test_filter_reports_multiple_rejection_reasons():
    step = _accepted_step()
    step["projection"]["projection_valid"] = 0
    step["env_info"]["fallback_used"] = True
    step["action_parse"]["raw"] = "pass"
    step["tokens"]["response_tokens_target_model"] = 2048
    episode = {"won": False, "normal_end": False, "final_player0_num_cards_left": 6}
    reasons = set(filter_reasons_for_step(step, episode, FilterConfig(max_response_tokens=1024)))
    assert {
        "terminal_player0_hand_gt_threshold",
        "episode_truncated",
        "projection_invalid",
        "fallback_used",
        "action_mismatch",
        "response_too_long",
    }.issubset(reasons)
