import copy
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ray

from agent_system.environments.env_package.doudizhu.envs import (
    DoudizhuSingleEnv,
    _cfg_get,
    _to_plain_config,
)

from .projection import doudizhu_grounding_projection


CARD_LABELS = {"T": "10", "B": "BJ", "R": "RJ"}


def _pretty_action(action: Optional[str], language: str = "en") -> str:
    if action is None:
        return ""
    if action == "pass":
        return "不要" if language == "zh" else "pass"
    return " ".join(CARD_LABELS.get(card, card) for card in action)


class DoudizhuGroundingSingleEnv(DoudizhuSingleEnv):
    """Canonical Dou Dizhu game for command-to-click grounding.

    Model actions are scored against the current teacher command, but the game
    state is advanced only by the teacher action.
    """

    def __init__(self, seed: int = 0, env_config: Optional[Dict[str, Any]] = None):
        super().__init__(seed=seed, env_config=env_config)
        env_config = _to_plain_config(env_config) or {}
        grounding_cfg = _cfg_get(env_config, "doudizhu_grounding", {}) or {}
        reward_cfg = _cfg_get(grounding_cfg, "reward", {}) or {}

        self.reward_projection = float(_cfg_get(reward_cfg, "projection_valid", 0.1))
        self.reward_click = float(_cfg_get(reward_cfg, "click_valid", 0.2))
        self.reward_submit = float(_cfg_get(reward_cfg, "submit_correct", 0.2))
        self.reward_target = float(_cfg_get(reward_cfg, "target_action_match", 1.0))
        self.teacher_policy = str(_cfg_get(grounding_cfg, "teacher_policy", "rule_v1"))
        self.target_action: Optional[str] = None
        self.episode_id = ""
        self.grounding_step = 0

    def reset(self, seed: Optional[int] = None, kwargs: Optional[Dict[str, Any]] = None):
        obs, _info = super().reset(seed=seed, kwargs=kwargs)
        self.episode_id = str(uuid.uuid4())
        self.grounding_step = 0
        self.target_action = self._teacher_action(self.game.state)
        return obs, self._grounding_info(reward=0.0)

    def score_group(self, actions: Sequence[Dict[str, Any]]):
        if self.game is None:
            raise RuntimeError("DoudizhuGroundingSingleEnv must be reset before step().")

        infos, rewards = [], []
        done = bool(self.done)
        for repeat_index, action in enumerate(actions):
            reward, info = self._score_action(action if isinstance(action, dict) else {}, repeat_index=repeat_index)
            rewards.append(reward)
            infos.append(info)
        return rewards, [done for _ in actions], infos

    def advance_teacher(self):
        if self.game is None:
            raise RuntimeError("DoudizhuGroundingSingleEnv must be reset before step().")
        if self.done:
            return self._render(), self._grounding_info(reward=0.0)

        action = self.target_action
        legal_actions = set(self.game.state.get("actions", []))
        if action not in legal_actions:
            action = self._fallback_action(self.game.state)

        self.game.step(action)
        bot_turns, bot_limit_reached = self._run_bots_until_player_turn()
        self.done = bool(self.game.is_over() or bot_limit_reached)
        self.grounding_step += 1

        if bot_limit_reached:
            self.last_message = "Bot turn limit reached; episode stopped."
        else:
            self.last_message = (
                f"指挥动作已执行：{self._pretty_action(action)}"
                if self.language == "zh"
                else f"Teacher executed: {self._pretty_action(action)}"
            )

        self.target_action = None if self.done else self._teacher_action(self.game.state)
        return self._render(), self._grounding_info(reward=0.0, bot_turns=bot_turns, bot_limit_reached=bot_limit_reached)

    def _teacher_action(self, state: Dict[str, Any]) -> str:
        if self.teacher_policy != "rule_v1":
            raise NotImplementedError(f"Unsupported doudizhu grounding teacher_policy: {self.teacher_policy}")
        action = self.bot_agents[1].step({"raw_obs": state})
        if action not in state.get("actions", []):
            action = self._fallback_action(state)
        return action

    def _score_action(self, action: Dict[str, Any], repeat_index: int = 0) -> Tuple[float, Dict[str, Any]]:
        if self.done or self.target_action is None:
            info = self._grounding_info(reward=0.0, repeat_index=repeat_index)
            info["won"] = 0.0
            info["done_before_scoring"] = True
            return 0.0, info

        raw_state = self.game.state
        candidate_action, click_valid_ratio, selected_indices, submit_kind = self._project_clicks_to_game_action(raw_state, action.get("clicks", []))
        target_action = self.target_action
        projection_valid = float(action.get("projection_valid", 0))
        submit_correct = float((target_action == "pass" and submit_kind == "pass") or (target_action != "pass" and submit_kind == "play"))
        target_action_match = float(candidate_action == target_action)
        selected_cards = "".join(raw_state.get("current_hand", "")[idx] for idx in selected_indices)

        reward = (
            self.reward_projection * projection_valid
            + self.reward_click * float(click_valid_ratio)
            + self.reward_submit * submit_correct
            + self.reward_target * target_action_match
        )

        info = self._grounding_info(
            reward=reward,
            repeat_index=repeat_index,
            projection_valid=projection_valid,
            click_valid_ratio=float(click_valid_ratio),
            submit_correct=submit_correct,
            target_action_match=target_action_match,
            predicted_action=candidate_action,
            selected_cards=selected_cards,
            selected_indices=sorted(selected_indices),
            submit_kind=submit_kind,
            plan=action.get("plan", ""),
            tool_calls=action.get("tool_calls", []),
            tool_calling=float(action.get("tool_calling", 0)),
            raw_tool_call_text=action.get("raw_tool_call_text", ""),
        )
        info["won"] = target_action_match
        info["task_score"] = target_action_match
        return float(reward), info

    def _grounding_info(self, reward: float, repeat_index: int = 0, **overrides):
        info = self._build_info(reward=reward)
        target_action = self.target_action
        info.update(
            {
                "won": 0.0,
                "task_score": 0.0,
                "target_action": target_action,
                "target_action_pretty": _pretty_action(target_action, self.language),
                "teacher_policy": self.teacher_policy,
                "grounding_step": self.grounding_step,
                "repeat_index": repeat_index,
                "grpo_uid": f"{self.episode_id}:step:{self.grounding_step}",
                "projection_valid": 0.0,
                "click_valid_ratio": 0.0,
                "submit_correct": 0.0,
                "target_action_match": 0.0,
                "predicted_action": None,
                "selected_cards": "",
                "selected_indices": [],
                "submit_kind": None,
                "plan": "",
                "raw_tool_call_text": "",
                "tool_calls": [],
                "tool_calling": 0.0,
                "done_before_scoring": False,
            }
        )
        info.update(overrides)
        return info


class DoudizhuGroundingWorker:
    def __init__(self, seed: int, env_config: Optional[Dict[str, Any]] = None):
        self.env = DoudizhuGroundingSingleEnv(seed=seed, env_config=env_config)

    def reset(self, seed_for_reset=None, kwargs=None):
        return self.env.reset(seed=seed_for_reset, kwargs=kwargs)

    def step_group(self, actions):
        rewards, _dones, infos = self.env.score_group(actions)
        obs, next_info = self.env.advance_teacher()
        dones = [bool(self.env.done) for _ in actions]
        return obs, rewards, dones, infos, next_info


class DoudizhuGroundingVectorEnv:
    def __init__(
        self,
        seed=0,
        env_num=1,
        group_n=1,
        is_train=True,
        env_config=None,
        resources_per_worker=None,
    ):
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.env_num = int(env_num)
        self.group_n = int(group_n)
        self.num_processes = self.env_num * self.group_n
        self.is_train = bool(is_train)
        self.env_config = _to_plain_config(env_config) or {}
        self.resources_per_worker = resources_per_worker or {"num_cpus": 0.1, "num_gpus": 0}
        grounding_cfg = _cfg_get(self.env_config, "doudizhu_grounding", {}) or {}
        self.use_ray = bool(_cfg_get(grounding_cfg, "use_ray", False))
        self.ray_init_error = None

        if self.use_ray:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
            except Exception as exc:
                self.use_ray = False
                self.ray_init_error = str(exc)

        if self.use_ray:
            worker_cls = ray.remote(**self.resources_per_worker)(DoudizhuGroundingWorker)
            self.workers = [worker_cls.remote(self.seed + idx, self.env_config) for idx in range(self.env_num)]
        else:
            self.workers = [DoudizhuGroundingWorker(self.seed + idx, self.env_config) for idx in range(self.env_num)]

    def reset(self, kwargs=None):
        seeds, per_env_kwargs = self._reset_inputs(kwargs)
        if self.use_ray:
            results = ray.get(
                [
                    worker.reset.remote(seed_for_reset=seeds[idx], kwargs=per_env_kwargs[idx])
                    for idx, worker in enumerate(self.workers)
                ]
            )
        else:
            results = [worker.reset(seed_for_reset=seeds[idx], kwargs=per_env_kwargs[idx]) for idx, worker in enumerate(self.workers)]

        obs_list, info_list = [], []
        for group_idx, (obs, info) in enumerate(results):
            for repeat_idx in range(self.group_n):
                expanded_info = copy.deepcopy(info)
                expanded_info["group_index"] = group_idx
                expanded_info["repeat_index"] = repeat_idx
                obs_list.append(obs)
                info_list.append(expanded_info)
        return np.array(obs_list, dtype=np.uint8), info_list

    def step(self, actions):
        assert len(actions) == self.num_processes
        grouped_actions = [actions[idx * self.group_n : (idx + 1) * self.group_n] for idx in range(self.env_num)]
        if self.use_ray:
            results = ray.get([worker.step_group.remote(grouped_actions[idx]) for idx, worker in enumerate(self.workers)])
        else:
            results = [worker.step_group(grouped_actions[idx]) for idx, worker in enumerate(self.workers)]

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for group_idx, (next_obs, rewards, dones, infos, next_info) in enumerate(results):
            for repeat_idx in range(self.group_n):
                info = copy.deepcopy(infos[repeat_idx])
                info["group_index"] = group_idx
                info["repeat_index"] = repeat_idx
                info["next_target_action"] = next_info.get("target_action")
                info["next_target_action_pretty"] = next_info.get("target_action_pretty")
                obs_list.append(next_obs)
                reward_list.append(float(rewards[repeat_idx]))
                done_list.append(bool(dones[repeat_idx]))
                info_list.append(info)
        return np.array(obs_list, dtype=np.uint8), reward_list, done_list, info_list

    def _reset_inputs(self, kwargs):
        if kwargs is None:
            if self.is_train:
                seeds = self.rng.randint(0, 2**16 - 1, size=self.env_num).astype(np.int64).tolist()
            else:
                seeds = self.rng.randint(2**16, 2**32 - 1, size=self.env_num, dtype=np.uint32).astype(np.int64).tolist()
            return seeds, [None for _ in range(self.env_num)]

        kwargs_list = list(kwargs)
        if len(kwargs_list) == self.num_processes:
            kwargs_list = kwargs_list[:: self.group_n]
        if len(kwargs_list) != self.env_num:
            raise ValueError(f"Expected {self.env_num} canonical env kwargs or {self.num_processes} repeated kwargs, got {len(kwargs_list)}")

        seeds = []
        for idx, item in enumerate(kwargs_list):
            if isinstance(item, dict) and item.get("seed") is not None:
                seeds.append(int(item["seed"]))
            else:
                seeds.append(self.seed + idx)
        return seeds, kwargs_list

    def close(self):
        workers = getattr(self, "workers", [])
        if getattr(self, "use_ray", False):
            for worker in workers:
                ray.kill(worker)
        self.workers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_doudizhu_grounding_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=None, resources_per_worker=None):
    return DoudizhuGroundingVectorEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
        resources_per_worker=resources_per_worker,
    )


__all__ = [
    "DoudizhuGroundingSingleEnv",
    "DoudizhuGroundingVectorEnv",
    "build_doudizhu_grounding_envs",
    "doudizhu_grounding_projection",
]
