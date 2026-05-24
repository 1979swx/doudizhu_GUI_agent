import random

from agent_system.environments.env_package.doudizhu.core.judger import DoudizhuJudger

from data_synthesis.doudizhu_qa.canonical import (
    display_action_to_raw,
    hand_to_display_list,
    longest_intervals,
    raw_action_to_display_list,
)
from data_synthesis.doudizhu_qa.schemas import GenerationConfig
from data_synthesis.doudizhu_qa.task_specs import build_task_specs
from data_synthesis.doudizhu_qa.verifier import verify_response


def test_card_and_action_canonicalization():
    assert hand_to_display_list("3TBR") == ["3", "10", "BJ", "RJ"]
    assert raw_action_to_display_list("334455") == ["3", "3", "4", "4", "5", "5"]
    assert raw_action_to_display_list("pass") == ["不要"]
    assert display_action_to_raw(["BJ", "RJ"]) == "BR"
    assert display_action_to_raw(["4", "3", "3"]) == "334"


def test_longest_interval_helpers_exclude_two_and_jokers():
    intervals, longest = longest_intervals("3456789TJQKA2BR", min_count=1, min_length=5)
    assert intervals == [["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]]
    assert longest == 12

    pair_intervals, pair_longest = longest_intervals("3344556622BR", min_count=2, min_length=3)
    assert pair_intervals == [["3", "4", "5", "6"]]
    assert pair_longest == 4


def test_all_task_specs_generate_verifiable_responses():
    hand = "33334445556667778899TJQKA2BR"
    playable = sorted(DoudizhuJudger.playable_cards_from_hand(hand))
    rich_state = {
        "self": 0,
        "landlord": 0,
        "current_hand": hand,
        "actions": playable[:12],
        "trace": [],
        "played_cards": ["", "", ""],
        "num_cards_left": [len(hand), 10, 10],
    }
    small_action_state = {
        **rich_state,
        "actions": ["pass", "9", "T"],
        "trace": [(1, "8")],
    }
    config = GenerationConfig(n_all=3, list_k=4)
    rng = random.Random(7)
    state_by_task = {"K_all_legal_actions": small_action_state}

    for spec in build_task_specs():
        state = state_by_task.get(spec.task_id, rich_state)
        if not spec.applies_to(state, config):
            continue
        gold = spec.build_gold(state, rng, config)
        assert gold is not None, spec.task_id
        response = spec.build_response(gold)
        result = verify_response(spec, response, gold)
        assert result.ok, (spec.task_id, result)


def test_targeted_label_buckets_and_forced_candidate_generation():
    specs = {spec.task_id: spec for spec in build_task_specs()}
    config = GenerationConfig(n_all=3, list_k=4)
    rng = random.Random(11)

    can_pass_state = {"current_hand": "34567", "actions": ["pass", "8"], "trace": [(1, "7")]}
    lead_state = {"current_hand": "34567", "actions": ["3", "4"], "trace": []}
    assert specs["I_can_pass"].label_bucket_for_state(can_pass_state, config) == "can_pass:true"
    assert specs["I_can_pass"].label_bucket_for_state(lead_state, config) == "can_pass:false"

    straight_state = {"current_hand": "3456789", "actions": ["34567"], "trace": []}
    no_straight_state = {"current_hand": "3337788992BR", "actions": ["3"], "trace": []}
    assert specs["F_straight"].label_bucket_for_state(straight_state, config) == "has_straight:true"
    assert specs["F_straight"].label_bucket_for_state(no_straight_state, config) == "has_straight:false"

    plane_attach_state = {"current_hand": "33344456789", "actions": ["33344456"], "trace": []}
    no_plane_attach_state = {"current_hand": "333444", "actions": ["333444"], "trace": []}
    assert specs["H2_plane_attachments"].label_bucket_for_state(plane_attach_state, config) == "has_plane_attachment:true"
    assert specs["H2_plane_attachments"].label_bucket_for_state(no_plane_attach_state, config) == "has_plane_attachment:false"

    small_action_state = {"current_hand": "34567", "actions": ["pass", "8"], "trace": [(1, "7")]}
    assert specs["K_all_legal_actions"].label_bucket_for_state(small_action_state, config) == "legal_action_count:2"

    gold = specs["M_candidate_legality"].build_gold_for_label(lead_state, rng, config, "candidate:illegal_pass")
    assert gold is not None
    assert gold.answer == {"候选动作": ["不要"], "是否合法": False}
    assert gold.metadata["label_bucket"] == "candidate:illegal_pass"
    assert gold.plan_aux["失败原因"] == "首发时不能选择不要"
    result = verify_response(specs["M_candidate_legality"], specs["M_candidate_legality"].build_response(gold), gold)
    assert result.ok, result
