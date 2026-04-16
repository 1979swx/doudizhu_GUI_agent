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
                actions[i] = "some crazy action"
            else:
                actions[i] = extracted_action
                valids[i] = 1

            if extract_tag(original_str, think_tag) == -1:
                valids[i] = 0
            if extract_tag(original_str, chat_tag) == -1:
                valids[i] = 0
            if extract_tag(original_str, memory_tag) == -1:
                valids[i] = 0
        except:
            actions[i] = "some crazy action"

    return actions, valids