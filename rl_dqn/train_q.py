# train_q.py
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

os.environ["SUMO_HOME"] = "project/sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci  # SUMO 仿真控制
from network import QNetwork

ACTIONS = [0, 1]

def start_sumo(sumo_cfg):
    traci.start(["sumo", "-c", sumo_cfg])
    
def get_state():
    lane_ids = ["N2C_0", "C2E_0", "C2S_0", "W2C_0"]
    lane_times = []
    lane_nums = []
    for lane_id in lane_ids:
        max_wait = 0
        veh_list = traci.lane.getLastStepVehicleIDs(lane_id)
        for vid in veh_list:
            waiting_time = traci.vehicle.getWaitingTime(vid)
            if waiting_time > max_wait:
                max_wait = waiting_time
        lane_times.append(max_wait)
        lane_nums.append(len(veh_list))
    return np.array(lane_times + lane_nums, dtype=np.float32)

def set_traffic_light(action):
    if action == 0:
        traci.trafficlight.setRedYellowGreenState("C", "GrGr")
    else:
        traci.trafficlight.setRedYellowGreenState("C", "rGrG")

def get_reward():
    all_veh_ids = traci.vehicle.getIDList()
    if len(all_veh_ids) == 0:
        return 0.0
    total_wait = sum(traci.vehicle.getWaitingTime(vid) for vid in all_veh_ids)
    avg_wait = total_wait / len(all_veh_ids)
    return -avg_wait

def train_one_episode(qnet, optimizer, epsilon=0.1, max_steps=1000):
    traci.start(["sumo", "-c", "testbed_demo/rl_dqn/config/4way.sumocfg"])
    total_reward = 0.0
    for step in range(max_steps):
        state = get_state()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            with torch.no_grad():
                action = qnet(state_t).argmax(dim=1).item()
        set_traffic_light(action)
        traci.simulationStep()
        next_state = get_state()
        reward = get_reward()
        gamma = 0.99
        with torch.no_grad():
            td_target = reward + gamma * qnet(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).max(dim=1)[0].item()
        q_values = qnet(state_t)
        q_sa = q_values[0, action]
        loss = F.mse_loss(q_sa, torch.tensor(td_target, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_reward += reward
        if traci.simulation.getMinExpectedNumber() <= 0:
            break
    traci.close()
    return total_reward

if __name__ == "__main__":
    qnet = QNetwork(state_dim=8, action_dim=2)
    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
    num_episodes = 50
    rewards = []

    # 确保 results 文件夹存在
    os.makedirs("results", exist_ok=True)

    # 记录 Reward 到文件
    reward_log_path = "results/reward_log.txt"
    with open(reward_log_path, "w") as f:
        for ep in range(num_episodes):
            start_sumo("/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/config/4way.sumocfg")
            traci.close()
            ep_reward = train_one_episode(qnet, optimizer, epsilon=0.1, max_steps=1000)
            rewards.append(ep_reward)
            f.write(f"Episode {ep+1}/{num_episodes}, total_reward={ep_reward:.2f}\n")
            print(f"Episode {ep+1}/{num_episodes}, total_reward={ep_reward:.2f}")

    torch.save(qnet.state_dict(), "/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/save/qnet_checkpoint.pth")
    print("Training finished, model saved to qnet_checkpoint.pth")

    # 📊 绘制 Reward 变化图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes+1), rewards, marker="o", linestyle="-", color="b", label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/save/reward_plot.png")  # 📌 保存图片
    # plt.show()





# netconvert --node-files=/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/config/4way.net.xml --edge-files=/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/config/4way.net.xml --output-file=/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/config/4way.net.xml