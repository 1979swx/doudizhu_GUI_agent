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




DUMMY_GUI_TEMPLATE = """
You are an game playing and companion agent operating in a game GUI environment. Your goal is to navigate through the game through actions and chat with the users.

# Current Step
Your current observation is shown in the image: <image>
Your previous memory is: {previous_memory}
Your actions should be based on the observation in the image.

Now it's your turn to make a move. Click ONE BUTTON for the current step by output "click[button_name]" (if no button, output "noop").
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
Output your understanding of the current situation and your strategic thoughts, and use this as the memory to reference for future turns. You MUST enclose it in <memory> </memory> tags.
You should chat with the user when playing the game. You MUST enclose your your chat output in <chat> </chat> tags.

# Example Output
<think>some reasoning process</think><action>click[start_match]</action><memory>some strategy</memory><chat>Let's go and win the game!</chat>
"""


from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from typing import List, Tuple, Dict, Union, Any

class DummyGUIEnvironmentManager(EnvironmentManagerBase):

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self.memory = None

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        self.memory = [None for _ in range(len(obs))]
        obs = np.array(obs, obs[0].dtype)
        observations = {
            'text': self.build_text_obs(infos, init=True), 
            'image': obs,   
            'anchor': obs
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        structured_response, projection_valids = self.projection_f(text_actions)
        actions = structured_response["action"]
        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_projection_valid'] = to_numpy(projection_valids[i])

        self.memory = structured_response["memory"]
        
        next_obs = np.array(next_obs, next_obs[0].dtype)
        next_observations = {
            'text': self.build_text_obs(infos),  
            'image': next_obs,
            'anchor': next_obs 
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        for i in range(len(infos)):
            if init:
                obs = DUMMY_GUI_TEMPLATE.format(
                    previous_memory="this is the initial state, no memory.",
                )
            else:
                obs = DUMMY_GUI_TEMPLATE.format(
                    previous_memory=self.memory[i],
                )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs