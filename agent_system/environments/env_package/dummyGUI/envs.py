# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import numpy as np
from PIL import Image
from pathlib import Path

class DummyGUIEnv():
    def __init__(self, max_steps):
        self.state_id = None
        self.step_count = None
        self.max_steps = max_steps
        self.script = [
                        {"screen": "lobby", "target_action": "click[start_match]"},
                        {"screen": "ready_room", "target_action": "click[ready]"},
                        {"screen": "turn_page", "target_action": "click[end_turn]"},
                        {"screen": "black", "target_action": "noop"}
                    ]
        self.available_actions = ["click[start_match]", "click[ready]", "click[end_turn]", "noop"]
        self.images = [
            self._read_image("lobby.jpg"),
            self._read_image("ready_room.jpg"),
            self._read_image("turn_page.jpg"),
            self._read_image("black.jpg")
        ]

    def reset(self):
        self.state_id = 0
        self.step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.state_id is None:
            raise Exception("Environment NOT activate")
        reward = 0
        done = False
        is_action_valid = False
        if self.state_id < len(self.script) - 1 and self.step_count < self.max_steps:
            self.step_count += 1
            target_action = self.script[self.state_id]["target_action"]
            if action in self.available_actions:
                is_action_valid = True
                if action == target_action:
                        self.state_id += 1
            else:
                reward -= 0.1

        if self.state_id == len(self.script) - 1:
            reward += 1
            done = True
        elif self.step_count == self.max_steps:
            done = True
        else:
            done = False
        
        obs = self._get_obs()
        info = self._get_info()
        # The projection is responsible for judging the format validity, 
        # while this is responsible for judging the legitimacy of the action in the environment, which 
        # is included in the environment reward.
        info["is_action_valid"] = is_action_valid
        
        return obs, reward, done, info

    def render(self):
        return self.images[self.state_id]

    def _get_obs(self):
        return self.render()

    def _get_info(self):
        info = {
            "won": 1.0 if self.state_id == len(self.script) - 1 else 0.0,
            "available_actions": self.available_actions,
            "screen_name": self.script[self.state_id]["screen"],
            "text_obs": f'current screen is {self.script[self.state_id]["screen"]}'
        }
        return info

    def _read_image(self, name):
        current_dir = Path(__file__).resolve().parent
        image_path = current_dir / "figures" / name
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        return img_array

class DummyGUIWorker:
    def __init__(self, max_steps):
        self.env = DummyGUIEnv(max_steps)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs, info = self.env.reset()
        return obs, info


class DummyGUIMultiProcessEnv:
    def __init__(self,
                 max_steps,
                 env_num,
                 group_n=1,
                 resources_per_worker={"num_cpus": 0.1}):
        if not ray.is_initialized():
            ray.init()
        
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n

        env_worker = ray.remote(**resources_per_worker)(DummyGUIWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(max_steps)
            self.workers.append(worker)
    
    def step(self, actions):
        assert len(actions) == self.num_processes

        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)
        
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        return obs_list, reward_list, done_list, info_list
    
    def reset(self):
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote()
            futures.append(future)
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list
    
    def close(self):
        for worker in self.workers:
            ray.kill(worker)
    
    def __del__(self):
        self.close()


def build_dummy_gui_envs(
        max_steps,
        env_num=1,
        group_n=1,
        resources_per_worker={"num_cpus": 0.1}):
    return DummyGUIMultiProcessEnv(max_steps, env_num, group_n, resources_per_worker)