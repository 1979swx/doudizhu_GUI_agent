DOUDIZHU_GROUNDING_TEMPLATE = """<image>
You are controlling the Dou Dizhu GUI by normalized left clicks.

Commanded card action: {command}

Your task is only to execute the commanded action on the screenshot. Do not choose a different card action.
- If the commanded action is pass, click only the PASS button.
- Otherwise, click each matching card in your bottom hand, then click the PLAY button.
- Coordinates must be normalized numbers from 0 to 1000, where [0,0] is the top-left corner and [1000,1000] is the bottom-right corner.
- Output one turn only.

Return exactly one XML tag. Do not output any explanation or other tags:
<tool_call>left_click([x1,y1],[x2,y2])</tool_call>
"""


DOUDIZHU_GROUNDING_TEMPLATE_ZH = """
你正在通过鼠标点击来控制斗地主 GUI。

指挥动作：{command}

<image>你的任务是在截图中执行这个指挥动作。不要自行选择其它出牌。
- 你通过 [x,y] 坐标来进行点击动作，坐标必须是 0 到 1000 范围内的归一化数字，[0,0] 代表左上角，[1000,1000] 代表右下角。
- 游戏页面的底部有手牌，其上方有‘出牌’和‘不要’按钮，这是主要交互区域。
- 如果指挥动作是“不要”或 pass，只点击“不要”按钮。
- 如果指挥动作是出牌，则依次点击底部手牌中与指挥动作匹配的每张牌，然后点击“出牌”按钮。
- 也就是说，每一轮动作的最后必须以点击“出牌”或“不要”两个按钮之一结尾。
- 工具定义：left_click([x1,y1],[x2,y2],...,[xN,yN]) ，支持批量点击，每个坐标对代表一次点击，N个坐标对代表N次点击。一次性输出本轮的所有点击。

输出一个名为 <tool_call> </tool_call> 的 XML 标签，标签内是一个 left_click([x1,y1],[x2,y2],...,[xN,yN]) 调用。

例子：
指挥动作：不要
输出：<tool_call>left_click([566,764])</tool_call>

指挥动作：3 3
输出：<tool_call>left_click([55,850],[100,860],[430,755])</tool_call>

当前指挥动作：{command}
"""
