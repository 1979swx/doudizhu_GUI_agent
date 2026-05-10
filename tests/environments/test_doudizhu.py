from functools import partial

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments.env_manager import DoudizhuEnvironmentManager
from agent_system.environments.env_package.doudizhu import build_doudizhu_envs, doudizhu_projection
from agent_system.environments.env_package.doudizhu.core.base import Card
from agent_system.environments.env_package.doudizhu.envs import DoudizhuSingleEnv
from agent_system.environments.env_package.doudizhu.renderer import DoudizhuRenderer


VALID_RESPONSE = "<plan>plan</plan><action>[[55, 870], [424, 758]]</action><chat>Let's press them.</chat><memory>I led with a low card.</memory>"


def _env_config(use_ray=False, language=None, chinese_mode=None):
    doudizhu_cfg = {
        "use_ray": use_ray,
        "image_width": 640,
        "image_height": 480,
        "max_clicks": 8,
        "max_memory_chars": 128,
        "max_bot_turns": 256,
        "reward": {
            "projection_valid": 0.05,
            "click_valid": 0.05,
            "rule_action_valid": 0.10,
            "hand_depletion": 0.01,
            "win": 1.0,
            "loss": -1.0,
        },
    }
    if language is not None:
        doudizhu_cfg["language"] = language
    if chinese_mode is not None:
        doudizhu_cfg["chinese_mode"] = chinese_mode
    return {
        "doudizhu": doudizhu_cfg,
    }


def test_doudizhu_projection_valid_and_invalid_cases():
    actions, valids = doudizhu_projection(
        [
            VALID_RESPONSE,
            "<plan>plan</plan><action>[[0, 100]]</action><chat>hi</chat><memory>m</memory>",
            "<plan>plan</plan><action>not-json</action><chat>hi</chat><memory>m</memory>",
            "<plan>plan</plan><action>[[100, 100]]</action><memory>m</memory>",
        ],
        max_clicks=8,
    )

    assert valids == [1, 0, 0, 0]
    assert actions[0]["clicks"] == [[55.0, 870.0], [424.0, 758.0]]
    assert actions[0]["chat"] == "Let's press them."
    assert actions[1]["clicks"] == []


def test_single_env_executes_gui_clicks_and_fallback():
    env = DoudizhuSingleEnv(seed=1, env_config=_env_config())
    obs, info = env.reset()
    assert obs.shape == (480, 640, 3)
    assert obs.dtype == np.uint8
    assert info["landlord"] == 0
    assert len(info["legal_actions"]) > 0

    _obs, reward, done, info = env.step({"clicks": [[55, 870], [424, 758]], "projection_valid": 1})
    assert np.isclose(reward, 0.21)
    assert done is False
    assert info["click_valid_ratio"] == 1.0
    assert info["rule_action_valid"] == 1.0
    assert info["fallback_used"] is False
    assert info["hand_cards_reduced"] == 1
    assert np.isclose(info["hand_depletion_reward"], 0.01)
    assert np.isclose(info["projection_valid_reward"], 0.05)
    assert np.isclose(info["click_valid_reward"], 0.05)
    assert np.isclose(info["rule_action_valid_reward"], 0.10)
    assert np.isclose(info["hand_depletion_reward_total"], 0.01)
    assert np.isclose(info["win_reward"], 0.0)

    env.reset(seed=1)
    _obs, reward, _done, info = env.step({"clicks": [[577, 758]], "projection_valid": 1})
    assert np.isclose(reward, 0.1)
    assert info["click_valid_ratio"] == 1.0
    assert info["rule_action_valid"] == 0.0
    assert info["fallback_used"] is True
    assert info["hand_cards_reduced"] == 1
    assert info["hand_depletion_reward"] == 0.0
    assert np.isclose(info["projection_valid_reward"], 0.05)
    assert np.isclose(info["click_valid_reward"], 0.05)
    assert np.isclose(info["rule_action_valid_reward"], 0.0)
    assert np.isclose(info["hand_depletion_reward_total"], 0.0)


def test_validity_rewards_accumulate_as_trajectory_average():
    env = DoudizhuSingleEnv(seed=1, env_config=_env_config())
    env.reset()

    _obs, first_reward, _done, first_info = env.step({"clicks": [[55, 870], [424, 758]], "projection_valid": 1})
    _obs, second_reward, _done, second_info = env.step({"clicks": [], "projection_valid": 0})

    assert np.isclose(first_info["validity_reward_average"], 0.20)
    assert np.isclose(first_reward, 0.21)
    assert np.isclose(second_info["validity_reward_average"], 0.10)
    assert np.isclose(second_info["validity_reward_delta"], -0.10)
    assert np.isclose(first_reward + second_reward, 0.11)
    assert np.isclose(second_info["projection_valid_reward"], 0.025)
    assert np.isclose(second_info["click_valid_reward"], 0.025)
    assert np.isclose(second_info["rule_action_valid_reward"], 0.05)


def test_submitted_click_ignores_trailing_actions_but_counts_them_invalid():
    env = DoudizhuSingleEnv(seed=1, env_config=_env_config())
    env.reset()

    candidate_action, click_valid_ratio, selected_indices, submit_kind = env._project_clicks_to_game_action(
        env.game.state,
        [[55, 870], [424, 758], [55, 870], [577, 758]],
    )

    assert candidate_action == "3"
    assert click_valid_ratio == 0.5
    assert selected_indices == [0]
    assert submit_kind == "play"


def test_renderer_buttons_are_centered_above_hand_and_clickable():
    env = DoudizhuSingleEnv(seed=1, env_config=_env_config())
    _obs, _info = env.reset()
    renderer = env.renderer
    state = env.game.state

    hitboxes = renderer.get_hitboxes(state)
    play_box = next(hitbox.box for hitbox in hitboxes if hitbox.kind == "play")
    pass_box = next(hitbox.box for hitbox in hitboxes if hitbox.kind == "pass")
    card_boxes = [hitbox.box for hitbox in hitboxes if hitbox.kind == "card"]

    assert play_box[1] < min(card_box[1] for card_box in card_boxes)
    assert pass_box[1] < min(card_box[1] for card_box in card_boxes)
    assert abs(((play_box[0] + pass_box[2]) / 2) - (renderer.width / 2)) <= 2

    assert renderer.hit_test(state, [], 424, 758).kind == "play"
    assert renderer.hit_test(state, [], 577, 758).kind == "pass"


def test_renderer_current_trick_actions_only_use_current_round():
    renderer = DoudizhuRenderer()

    completed_trick = [(0, "3"), (1, "pass"), (2, "pass")]
    assert renderer._current_trick_actions(completed_trick) == {0: "3", 1: "pass", 2: "pass"}

    new_trick = [(0, "3"), (1, "pass"), (2, "pass"), (0, "4")]
    assert renderer._current_trick_actions(new_trick) == {0: "4"}

    contested_trick = [(0, "3"), (1, "4"), (2, "pass")]
    assert renderer._current_trick_actions(contested_trick) == {0: "3", 1: "4", 2: "pass"}


def test_single_env_terminal_win_reward_and_payoffs():
    env = DoudizhuSingleEnv(seed=1, env_config=_env_config())
    env.reset()
    env.game.players[0].set_current_hand([Card("S", "3")])
    env.game.judger.playable_cards[0] = {"3"}
    env.game.round.current_player = 0
    env.game.round.greater_player = None
    env.game.winner_id = None
    env.game.state = env.game.get_state(0)

    _obs, reward, done, info = env.step({"clicks": [[500, 850], [424, 758]], "projection_valid": 1})

    assert done is True
    assert np.isclose(reward, 1.21)
    assert info["won"] == 1.0
    assert info["task_score"] == 1.0
    assert info["winner_id"] == 0
    assert info["payoffs"] == [1, 0, 0]
    assert info["rule_action_valid"] == 1.0
    assert info["hand_cards_reduced"] == 1
    assert np.isclose(info["hand_depletion_reward"], 0.01)
    assert np.isclose(info["win_reward"], 1.0)


def test_single_env_terminal_loss_payoffs_do_not_require_game_get_payoffs():
    env = DoudizhuSingleEnv(seed=1, env_config=_env_config())
    env.reset()
    env.game.winner_id = 1

    assert env._terminal_reward() == -1.0
    info = env._build_info(reward=-1.0)
    assert info["won"] == 0.0
    assert info["task_score"] == 0.0
    assert info["winner_id"] == 1
    assert info["payoffs"] == [0, 1, 1]


def test_vector_env_local_fallback_preserves_group_seed():
    env = build_doudizhu_envs(seed=3, env_num=1, group_n=2, is_train=True, env_config=_env_config(use_ray=False))
    obs, infos = env.reset()
    assert obs.shape == (2, 480, 640, 3)
    assert len(infos) == 2
    assert infos[0]["state_summary"]["hand"] == infos[1]["state_summary"]["hand"]

    obs, rewards, dones, infos = env.step(
        [
            {"clicks": [[55, 870], [424, 758]], "projection_valid": 1},
            {"clicks": [[577, 758]], "projection_valid": 1},
        ]
    )
    assert obs.shape == (2, 480, 640, 3)
    assert np.allclose(rewards, [0.21, 0.1])
    assert dones == [False, False]
    assert [info["fallback_used"] for info in infos] == [False, True]
    assert [info["hand_depletion_reward"] for info in infos] == [0.01, 0.0]
    env.close()


def test_doudizhu_manager_builds_visual_prompt_and_memory():
    envs = build_doudizhu_envs(seed=5, env_num=1, group_n=1, is_train=True, env_config=_env_config(use_ray=False))
    config = OmegaConf.create(
        {
            "env": {
                "doudizhu": {
                    "max_clicks": 8,
                    "max_memory_chars": 128,
                }
            }
        }
    )
    manager = DoudizhuEnvironmentManager(envs, partial(doudizhu_projection, max_clicks=8), config)

    obs, infos = manager.reset(kwargs=None)
    assert "<image>" in obs["text"][0]
    assert obs["image"].shape == (1, 480, 640, 3)
    assert obs["anchor"].shape == (1,)

    next_obs, rewards, dones, infos = manager.step([VALID_RESPONSE])
    assert "<image>" in next_obs["text"][0]
    assert "I led with a low card." in next_obs["text"][0]
    assert rewards.shape == (1,)
    assert dones.shape == (1,)
    assert infos[0]["is_projection_valid"].item() == 1
    assert infos[0]["chat"] == "Let's press them."
    success = manager.success_evaluator(
        total_infos=[[infos[0]]],
        total_batch_list=[[{"active_masks": True}]],
    )
    assert np.isclose(success["doudizhu_reward_projection_valid"][0], 0.05)
    assert np.isclose(success["doudizhu_reward_click_valid"][0], 0.05)
    assert np.isclose(success["doudizhu_reward_rule_action_valid"][0], 0.10)
    assert np.isclose(success["doudizhu_reward_hand_depletion"][0], 0.01)
    assert np.isclose(success["doudizhu_reward_win"][0], 0.0)
    manager.close()


def test_doudizhu_chinese_mode_uses_chinese_prompt_and_renderer_text_mode():
    envs = build_doudizhu_envs(
        seed=5,
        env_num=1,
        group_n=1,
        is_train=True,
        env_config=_env_config(use_ray=False, language="zh"),
    )
    config = OmegaConf.create(
        {
            "env": {
                "doudizhu": {
                    "language": "zh",
                    "max_clicks": 8,
                    "max_memory_chars": 128,
                }
            }
        }
    )
    manager = DoudizhuEnvironmentManager(envs, partial(doudizhu_projection, max_clicks=8), config)

    obs, _infos = manager.reset(kwargs=None)
    assert "你是一个斗地主游戏陪玩 agent" in obs["text"][0]
    assert "“出牌”和“不要”按钮" in obs["text"][0]
    assert "<plan>" in obs["text"][0]
    assert "<action>" in obs["text"][0]
    assert obs["image"].shape == (1, 480, 640, 3)

    local_env = DoudizhuSingleEnv(seed=1, env_config=_env_config(language="zh"))
    image_obs, _info = local_env.reset()
    assert local_env.renderer.language == "zh"
    assert local_env.renderer.text["play"] == "出牌"
    assert str(local_env.renderer.font.path).endswith("NotoSansCJKsc-Regular.otf")
    assert image_obs.shape == (480, 640, 3)
    manager.close()
