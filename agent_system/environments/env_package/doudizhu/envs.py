from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ray

from .core import Game
from .core.rule_agent import DouDizhuRuleAgentV1
from .core.utils import INDEX
from .renderer import DoudizhuRenderer


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_plain_config(config):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except Exception:
        pass
    return config


def _is_chinese_mode(doudizhu_cfg) -> bool:
    if bool(_cfg_get(doudizhu_cfg, "chinese_mode", False)):
        return True
    language = str(_cfg_get(doudizhu_cfg, "language", "en")).lower()
    return language in ("zh", "zh-cn", "chinese", "cn")


class DoudizhuSingleEnv:
    """Single Dou Dizhu game controlled through in-memory GUI clicks."""

    def __init__(self, seed: int = 0, env_config: Optional[Dict[str, Any]] = None):
        self.base_seed = int(seed)
        self.env_config = _to_plain_config(env_config) or {}
        doudizhu_cfg = _cfg_get(self.env_config, "doudizhu", {}) or {}
        reward_cfg = _cfg_get(doudizhu_cfg, "reward", {}) or {}

        self.max_bot_turns = int(_cfg_get(doudizhu_cfg, "max_bot_turns", 256))
        self.max_clicks = int(_cfg_get(doudizhu_cfg, "max_clicks", 8))
        self.language = "zh" if _is_chinese_mode(doudizhu_cfg) else "en"
        self.renderer = DoudizhuRenderer(
            width=int(_cfg_get(doudizhu_cfg, "image_width", 640)),
            height=int(_cfg_get(doudizhu_cfg, "image_height", 480)),
            language=self.language,
        )
        self.reward_projection = float(_cfg_get(reward_cfg, "projection_valid", 0.05))
        self.reward_click = float(_cfg_get(reward_cfg, "click_valid", 0.05))
        self.reward_rule_action = float(_cfg_get(reward_cfg, "rule_action_valid", 0.10))
        self.reward_hand_depletion = float(_cfg_get(reward_cfg, "hand_depletion", 0.01))
        self.reward_win = float(_cfg_get(reward_cfg, "win", 1.0))
        self.reward_loss = float(_cfg_get(reward_cfg, "loss", -1.0))

        self.rng = np.random.RandomState(self.base_seed)
        self.game: Optional[Game] = None
        self.bot_agents: List[Optional[DouDizhuRuleAgentV1]] = []
        self.done = False
        self.last_message = ""
        self._reset_reward_trackers()

    def reset(self, seed: Optional[int] = None, kwargs: Optional[Dict[str, Any]] = None):
        if kwargs and kwargs.get("seed") is not None:
            seed = int(kwargs["seed"])
        if seed is None:
            seed = self.base_seed

        self.rng = np.random.RandomState(int(seed))
        self.game = Game()
        self.game.np_random = self.rng
        state, player_id = self.game.init_game()
        if player_id != 0:
            raise RuntimeError(f"Expected fixed landlord/player0 start, got player {player_id}")
        self.bot_agents = [None, DouDizhuRuleAgentV1(self.rng), DouDizhuRuleAgentV1(self.rng)]
        self.done = False
        self.last_message = ""
        self._reset_reward_trackers()
        return self._render(), self._build_info(reward=0.0)

    def step(self, action: Dict[str, Any]):
        if self.game is None:
            raise RuntimeError("DoudizhuSingleEnv must be reset before step().")
        if self.done:
            return self._render(), 0.0, True, self._build_info(reward=0.0)

        action = action if isinstance(action, dict) else {}
        raw_state = self.game.state
        prev_hand_count = int(raw_state.get("num_cards_left", [self._num_cards_left(0)])[0])
        candidate_action, click_valid_ratio, selected_indices, submit_kind = self._project_clicks_to_game_action(raw_state, action.get("clicks", []))
        legal_actions = set(raw_state.get("actions", []))

        fallback_used = candidate_action not in legal_actions
        if fallback_used:
            game_action = self._fallback_action(raw_state)
            rule_action_valid = 0.0
        else:
            game_action = candidate_action
            rule_action_valid = 1.0

        self.game.step(game_action)
        bot_turns, bot_limit_reached = self._run_bots_until_player_turn()
        hand_cards_reduced = max(0, prev_hand_count - self._num_cards_left(0))
        hand_depletion_reward = self.reward_hand_depletion * hand_cards_reduced if not fallback_used else 0.0
        self.done = bool(self.game.is_over() or bot_limit_reached)
        if bot_limit_reached:
            self.last_message = (
                "Bot 回合达到上限，当前 episode 已停止。"
                if self.language == "zh"
                else "Bot turn limit reached; episode stopped."
            )
        elif fallback_used:
            self.last_message = (
                f"已执行兜底动作：{self._pretty_action(game_action)}"
                if self.language == "zh"
                else f"Fallback executed: {self._pretty_action(game_action)}"
            )
        else:
            self.last_message = (
                f"你出了：{self._pretty_action(game_action)}"
                if self.language == "zh"
                else f"You played: {self._pretty_action(game_action)}"
            )

        terminal_reward = self._terminal_reward() if self.done else 0.0
        projection_valid = float(action.get("projection_valid", 0))
        validity_reward_delta, validity_reward_average = self._update_validity_reward_average(
            projection_valid=projection_valid,
            click_valid_ratio=float(click_valid_ratio),
            rule_action_valid=rule_action_valid,
        )
        reward = (
            validity_reward_delta
            + hand_depletion_reward
            + terminal_reward
        )

        info = self._build_info(
            reward=reward,
            click_valid_ratio=click_valid_ratio,
            rule_action_valid=rule_action_valid,
            fallback_used=fallback_used,
            game_action=game_action,
            hand_cards_reduced=hand_cards_reduced,
            hand_depletion_reward=hand_depletion_reward,
            selected_cards="".join(raw_state.get("current_hand", "")[idx] for idx in selected_indices),
            submit_kind=submit_kind,
            bot_turns=bot_turns,
            bot_limit_reached=bot_limit_reached,
            projection_valid=projection_valid,
            validity_reward_delta=validity_reward_delta,
            validity_reward_average=validity_reward_average,
            chat=action.get("chat", ""),
            memory=action.get("memory", ""),
        )
        return self._render(), reward, self.done, info

    def _reset_reward_trackers(self):
        self.validity_step_count = 0
        self.projection_valid_sum = 0.0
        self.click_valid_ratio_sum = 0.0
        self.rule_action_valid_sum = 0.0

    def _validity_reward_average(self) -> float:
        if self.validity_step_count <= 0:
            return 0.0
        inv_count = 1.0 / float(self.validity_step_count)
        return (
            self.reward_projection * self.projection_valid_sum * inv_count
            + self.reward_click * self.click_valid_ratio_sum * inv_count
            + self.reward_rule_action * self.rule_action_valid_sum * inv_count
        )

    def _update_validity_reward_average(
        self,
        projection_valid: float,
        click_valid_ratio: float,
        rule_action_valid: float,
    ) -> Tuple[float, float]:
        previous_average = self._validity_reward_average()
        self.validity_step_count += 1
        self.projection_valid_sum += float(projection_valid)
        self.click_valid_ratio_sum += float(click_valid_ratio)
        self.rule_action_valid_sum += float(rule_action_valid)
        current_average = self._validity_reward_average()
        return current_average - previous_average, current_average

    def _project_clicks_to_game_action(self, state: Dict[str, Any], clicks: Sequence[Sequence[float]]):
        selected = set()
        hit_scores = []
        submitted = None

        if not isinstance(clicks, Sequence):
            clicks = []
        bounded_clicks = list(clicks)[: self.max_clicks]
        for click_idx, click in enumerate(bounded_clicks):
            if not isinstance(click, Sequence) or len(click) != 2:
                hit_scores.append(0.0)
                continue
            hitbox = self.renderer.hit_test(state, sorted(selected), click[0], click[1])
            hit_scores.append(1.0 if hitbox is not None else 0.0)
            if hitbox is None:
                continue
            if hitbox.kind == "card":
                if hitbox.payload in selected:
                    selected.remove(hitbox.payload)
                else:
                    selected.add(hitbox.payload)
            elif hitbox.kind in ("play", "pass"):
                submitted = hitbox.kind
                hit_scores.extend([0.0] * (len(bounded_clicks) - click_idx - 1))
                break

        click_valid_ratio = float(np.mean(hit_scores)) if hit_scores else 0.0
        if submitted == "pass":
            return "pass", click_valid_ratio, sorted(selected), submitted
        if submitted == "play":
            hand = state.get("current_hand", "")
            return "".join(hand[idx] for idx in sorted(selected)), click_valid_ratio, sorted(selected), submitted
        return None, click_valid_ratio, sorted(selected), submitted

    def _run_bots_until_player_turn(self) -> Tuple[int, bool]:
        bot_turns = 0
        while not self.game.is_over() and self.game.get_player_id() != 0:
            if bot_turns >= self.max_bot_turns:
                return bot_turns, True
            state = self.game.state
            player_id = int(state["self"])
            bot_action = self.bot_agents[player_id].step({"raw_obs": state})
            if bot_action not in state.get("actions", []):
                bot_action = self._fallback_action(state)
            self.game.step(bot_action)
            bot_turns += 1
        return bot_turns, False

    def _fallback_action(self, state: Dict[str, Any]) -> str:
        legal_actions = list(state.get("actions", []))
        if not legal_actions:
            return "pass"
        if "pass" in legal_actions:
            return "pass"
        return sorted(legal_actions, key=self._action_sort_key)[0]

    def _action_sort_key(self, action: str):
        if action == "pass":
            return (999, 999, action)
        ranks = [INDEX.get(card, 99) for card in action]
        return (len(action), max(ranks) if ranks else 99, sum(ranks), action)

    def _terminal_reward(self) -> float:
        if not self.game.is_over():
            return 0.0
        return self.reward_win if self._payoffs()[0] > 0 else self.reward_loss

    def _payoffs(self):
        if self.game is None or self.game.winner_id is None:
            return np.array([0, 0, 0], dtype=np.int32)
        return self.game.judger.judge_payoffs(self.game.round.landlord_id, self.game.winner_id)

    def _render(self):
        state = self._safe_state()
        return self.renderer.render(state, message=self.last_message)

    def _safe_state(self):
        if self.game is None or self.game.state is None:
            return {
                "self": 0,
                "landlord": 0,
                "current_hand": "",
                "seen_cards": "",
                "trace": [],
                "played_cards": ["", "", ""],
                "num_cards_left": [0, 0, 0],
                "actions": [],
            }
        return self.game.state

    def _build_info(self, reward: float, **overrides):
        state = self._safe_state()
        is_over = bool(self.game.is_over()) if self.game is not None else False
        payoffs = self._payoffs().tolist() if is_over else [0, 0, 0]
        info = {
            "won": float(is_over and payoffs[0] > 0),
            "task_score": float(payoffs[0]) if is_over else 0.0,
            "reward": float(reward),
            "current_player": int(state.get("self", 0)),
            "landlord": int(state.get("landlord", 0)),
            "num_cards_left": list(state.get("num_cards_left", [0, 0, 0])),
            "legal_actions": list(state.get("actions", [])),
            "trace": list(state.get("trace", [])),
            "winner_id": None if not is_over else int(self.game.winner_id),
            "payoffs": payoffs,
            "state_summary": self.compact_state(),
            "click_valid_ratio": 0.0,
            "rule_action_valid": 0.0,
            "fallback_used": False,
            "game_action": None,
            "hand_cards_reduced": 0,
            "hand_depletion_reward": 0.0,
            "selected_cards": "",
            "submit_kind": None,
            "bot_turns": 0,
            "bot_limit_reached": False,
            "projection_valid": 0.0,
            "validity_reward_delta": 0.0,
            "validity_reward_average": self._validity_reward_average(),
            "chat": "",
            "memory": "",
        }
        info.update(overrides)
        return info

    def compact_state(self):
        state = self._safe_state()
        return {
            "self": state.get("self", 0),
            "landlord": state.get("landlord", 0),
            "hand": state.get("current_hand", ""),
            "num_cards_left": list(state.get("num_cards_left", [0, 0, 0])),
            "trace": list(state.get("trace", []))[-6:],
        }

    def _pretty_action(self, action: Optional[str]) -> str:
        if action is None:
            return "none"
        if action == "pass":
            return "不要" if self.language == "zh" else "PASS"
        labels = {"T": "10", "B": "BJ", "R": "RJ"}
        return " ".join(labels.get(card, card) for card in action)

    def _num_cards_left(self, player_id: int) -> int:
        if self.game is None:
            return 0
        return len(self.game.players[player_id].current_hand)


class DoudizhuWorker:
    def __init__(self, seed: int, env_config: Optional[Dict[str, Any]] = None):
        self.env = DoudizhuSingleEnv(seed=seed, env_config=env_config)

    def reset(self, seed_for_reset=None, kwargs=None):
        return self.env.reset(seed=seed_for_reset, kwargs=kwargs)

    def step(self, action):
        return self.env.step(action)


class DoudizhuVectorEnv:
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
        doudizhu_cfg = _cfg_get(self.env_config, "doudizhu", {}) or {}
        self.use_ray = bool(_cfg_get(doudizhu_cfg, "use_ray", True))
        self.ray_init_error = None

        if self.use_ray:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
            except Exception as exc:
                self.use_ray = False
                self.ray_init_error = str(exc)

        if self.use_ray:
            worker_cls = ray.remote(**self.resources_per_worker)(DoudizhuWorker)
            self.workers = [
                worker_cls.remote(self.seed + idx, self.env_config)
                for idx in range(self.num_processes)
            ]
        else:
            self.workers = [DoudizhuWorker(self.seed + idx, self.env_config) for idx in range(self.num_processes)]

    def reset(self, kwargs=None):
        seeds, per_env_kwargs = self._reset_inputs(kwargs)
        if self.use_ray:
            futures = [
                worker.reset.remote(seed_for_reset=seeds[idx], kwargs=per_env_kwargs[idx])
                for idx, worker in enumerate(self.workers)
            ]
            results = ray.get(futures)
        else:
            results = [
                worker.reset(seed_for_reset=seeds[idx], kwargs=per_env_kwargs[idx])
                for idx, worker in enumerate(self.workers)
            ]
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list, dtype=np.uint8), info_list

    def step(self, actions):
        assert len(actions) == self.num_processes
        if self.use_ray:
            futures = [worker.step.remote(action) for worker, action in zip(self.workers, actions)]
            results = ray.get(futures)
        else:
            results = [worker.step(action) for worker, action in zip(self.workers, actions)]
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return np.array(obs_list, dtype=np.uint8), reward_list, done_list, info_list

    def _reset_inputs(self, kwargs):
        if kwargs is None:
            if self.is_train:
                base_seeds = self.rng.randint(0, 2**16 - 1, size=self.env_num)
            else:
                base_seeds = self.rng.randint(2**16, 2**32 - 1, size=self.env_num, dtype=np.uint32)
            seeds = np.repeat(base_seeds, self.group_n).astype(np.int64).tolist()
            return seeds, [None for _ in range(self.num_processes)]

        kwargs_list = list(kwargs)
        if len(kwargs_list) == self.env_num and self.group_n > 1:
            kwargs_list = [item for item in kwargs_list for _ in range(self.group_n)]
        if len(kwargs_list) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} env kwargs, got {len(kwargs_list)}")

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


def build_doudizhu_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=None, resources_per_worker=None):
    return DoudizhuVectorEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
        resources_per_worker=resources_per_worker,
    )
