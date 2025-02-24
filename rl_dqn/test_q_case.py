import numpy as np
import os
import sys
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 用于可视化
import os
import sys

os.environ["SUMO_HOME"] = "sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from network import QNetwork



# 规则模型
# RL 训练好的模型
RL_MODEL_PATH = "testbed_demo/rl_dqn/save/qnet_checkpoint.pth"

# 固定的测试数据
lane_bots = {
    2: [432, 429],
    4: [435],
    6: [],
    8: [428, 425, 436]
}
bots_obs_times = {
    432: 3.5,
    429: 15.6,
    435: 4.8,
    428: 7.0,
    425: 25.9,
    436: 10.3
}

# 车道 ID，和 SUMO 里的对应
lane_ids = ["N2C_0", "C2E_0", "C2S_0", "W2C_0"]

def get_state_from_fixed_data():
    """
    直接用固定的 lane_bots 数据转换成 state
    """
    lane_times = []
    lane_nums = []
    for lane_id in [2, 4, 6, 8]:
        max_wait = max([bots_obs_times[bot] for bot in lane_bots.get(lane_id, [])], default=0)
        lane_times.append(max_wait)
        lane_nums.append(len(lane_bots.get(lane_id, [])))
    
    return np.array(lane_times + lane_nums, dtype=np.float32)



def get_rl_model_action(qnet):
    """
    RL Policy 计算 traffic light action
    """
    state = get_state_from_fixed_data()
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = qnet(state_t)
        rl_action = q_values.argmax(dim=1).item()  # 0 or 1
    return [1, 0, 1, 0] if rl_action == 0 else [0, 1, 0, 1]

if __name__ == "__main__":
    
    # 1. 加载 RL 训练好的模型
    qnet = QNetwork(state_dim=8, action_dim=2)
    qnet.load_state_dict(torch.load(RL_MODEL_PATH))
    qnet.eval()

    # 2. 计算 RL Policy 预测的 Traffic Light Action
    rl_action = get_rl_model_action(qnet)
    print(f"RL Policy Model Action: {rl_action}")


# RL Policy Model Action: [0, 1, 0, 1]