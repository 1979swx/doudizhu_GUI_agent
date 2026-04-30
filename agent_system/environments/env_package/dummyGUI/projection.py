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

from typing import List
import re

def extract_tag(text: str, tag: str):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return -1
    return match.group(1).strip()

def dummy_gui_projection(actions: List[str]):
    # These fuction judges the format correctness and extract actions
    n = len(actions)
    action_tag = "action"
    think_tag = "think"
    chat_tag = "chat"
    memory_tag = "memory"
    action_list = [None] * n
    think_list = [None] * n
    chat_list = [None] * n
    memory_list = [None] * n
    projection_valids = [1] * n
    structured_response = {
        action_tag: action_list,
        think_tag: think_list,
        chat_tag: chat_list,
        memory_tag: memory_list
    }
    for i in range(len(actions)):
        # Attempt to extract the action
        try:
            extracted_action = extract_tag(actions[i], action_tag)
            if extracted_action == -1:
                action_list[i] = "action extraction failed"
                projection_valids[i] = 0
            else:
                action_list[i] = extracted_action
        except:
            action_list[i] = "action extraction failed"
            projection_valids[i] = 0

        # Attempt to extract the think
        try:
            extracted_think = extract_tag(actions[i], think_tag)
            if extracted_think == -1:
                think_list[i] = "think extraction failed"
                projection_valids[i] = 0
            else:
                think_list[i] = extracted_think
        except:
            think_list[i] = "think extraction failed"
            projection_valids[i] = 0
        
        # Attempt to extract the chat
        try:
            extracted_chat = extract_tag(actions[i], chat_tag)
            if extracted_chat == -1:
                chat_list[i] = "chat extraction failed"
                projection_valids[i] = 0
            else:
                chat_list[i] = extracted_chat
        except:
            chat_list[i] = "chat extraction failed"
            projection_valids[i] = 0

        # Attempt to extract the memory
        try:
            extracted_memory = extract_tag(actions[i], memory_tag)
            if extracted_memory == -1:
                memory_list[i] = "memory extraction failed"
                projection_valids[i] = 0
            else:
                memory_list[i] = extracted_memory
        except:
            memory_list[i] = "memory extraction failed"
            projection_valids[i] = 0

    return structured_response, projection_valids