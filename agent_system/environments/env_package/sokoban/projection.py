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

import torch
import random
from typing import List
import re


def sokoban_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    Sokoban action mappings:
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """

    action_pools = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = 0  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()

            for act in action_pools.keys():
                if act in extracted_action:
                    actions[i] = action_pools[act]
                    # if found legal action, set valids to 1
                    valids[i] = 1
                    break

            # If no valid action found, randomly select from pool
            if valids[i] == 0:
                actions[i] = 0

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = 0

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return actions, valids


def extract_tag(text: str, tag: str):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return -1
    return match.group(1).strip()

def sokoban_projection_my_modification(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
        <chat>some chat...</chat><memory>some memory...</memory>
    Sokoban action mappings:
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """

    action_pools = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        action_tag = "action"
        think_tag = "think"
        chat_tag = "chat"
        memory_tag = "memory"
        try:
            extracted_action = extract_tag(actions[i], action_tag)
            if extracted_action == -1:
                actions[i] = 0
                continue
            elif extracted_action in action_pools:
                actions[i] = action_pools[extracted_action]
                valids[i] = 1
            else:
                actions[i] = 0

            if extract_tag(original_str, think_tag) == -1:
                valids[i] = 0
            if extract_tag(original_str, chat_tag) == -1:
                valids[i] = 0
            if extract_tag(original_str, memory_tag) == -1:
                valids[i] = 0
        except:
            # randomly choose an action from the action list if illegal
            actions[i] = 0

    return actions, valids