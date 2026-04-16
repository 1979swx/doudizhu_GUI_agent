import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time  # 引入 time 模块用于控制渲染速度
import os    # 引入 os 模块用于清屏（可选，让动画更干净）

class TreasureHuntEnv(gym.Env):
    """
    一维寻宝环境
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.grid_size = 10
        self.agent_pos = 0
        self.max_steps = 20
        self.current_step = 0
        
        self.render_mode = render_mode

        # 1. 动作空间：0 表示向左，1 表示向右
        self.action_space = spaces.Discrete(2)

        # 2. 观测空间：智能体的一维坐标 (0 到 9 的整数)
        # 使用 Box 来表示坐标，形状为 (1,)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(1,), dtype=np.float32
        )

    def _get_obs(self):
        # 封装一下获取观测的方法，保证返回的数据类型与 observation_space 匹配
        return np.array([self.agent_pos], dtype=np.float32)

    def _get_info(self):
        # 可以返回智能体距离目标的距离等辅助信息
        return {"distance_to_treasure": self.grid_size - 1 - self.agent_pos}

    def reset(self, seed=None, options=None):
        # 初始化随机数种子 (Gymnasium 标准要求)
        super().reset(seed=seed)
        
        # 重置环境状态
        self.agent_pos = 0
        self.current_step = 0

        # 返回 observation 和 info
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1

        # 根据动作更新状态
        if action == 0:   # 向左
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1: # 向右
            self.agent_pos = min(self.grid_size - 1, self.agent_pos + 1)

        # 计算奖励和终止状态
        terminated = False
        truncated = False
        reward = -1.0  # 每走一步扣1分，时间惩罚

        if self.agent_pos == self.grid_size - 1:
            terminated = True
            reward = 100.0  # 找到宝藏！
        
        # 检查是否超时 (截断)
        if self.current_step >= self.max_steps:
            truncated = True

        # 如果需要渲染，在这里调用
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        # 简单的终端可视化
        grid = ["_"] * self.grid_size
        grid[self.grid_size - 1] = "T" # Treasure
        grid[self.agent_pos] = "A"     # Agent
        
        if self.render_mode == "ansi":
            return "".join(grid)
        elif self.render_mode == "human":
            print("".join(grid))

def main():
    from gymnasium.utils.env_checker import check_env

    # 1. 核心改变：实例化时显式传入 render_mode="human"
    env = TreasureHuntEnv(render_mode="human")

    # 运行检查器
    check_env(env)
    print("环境检查通过！没有任何报错。\n")
    print("================ 开始寻宝演示 ================")

    # 2. 重置环境，并手动渲染第一帧（因为原代码的 reset 中没有调用 render）
    obs, info = env.reset()
    print("\n初始状态:")
    env.render()
    time.sleep(1.0)  # 暂停1秒，让你看清初始位置

    # 3. 运行测试循环
    for step_num in range(25):
        print(f"\n--- 第 {step_num + 1} 步 ---")
        
        # 从动作空间中随机采样一个动作
        # action = env.action_space.sample() 
        action = 1
        
        # 将动作含义打印出来，方便人类阅读
        action_name = "向右(1)" if action == 1 else "向左(0)"
        print(f"智能体决定: {action_name}")
        
        # 你的 step 方法内部包含了 if self.render_mode == "human": self.render()
        # 所以这一步不仅会更新状态，还会直接在终端打印出新的网格图案
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"当前观测坐标: {obs}, 获得奖励: {reward}")
        
        # 核心改变：增加 0.8 秒的停顿，制造“逐帧播放”的动画感
        time.sleep(0.3) 
        
        # 检查是否结束
        if terminated or truncated:
            if terminated:
                print("\n🎉 游戏结束：成功找到宝藏！")
            else:
                print("\n⏳ 游戏结束：步数耗尽，被截断！")
            break

if __name__ == "__main__":
    main()