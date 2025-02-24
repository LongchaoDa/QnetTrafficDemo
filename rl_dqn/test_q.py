# test_q.py

import numpy as np
import torch
import traci
from network import QNetwork

# 同上
ACTIONS = [0, 1]

def start_sumo(sumo_cfg):
    traci.start(["sumo", "-c", sumo_cfg])

def get_state():
    lane_ids = ["lane2", "lane4", "lane6", "lane8"]
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
        traci.trafficlight.setRedYellowGreenState("TL", "GrGr") 
    else:
        traci.trafficlight.setRedYellowGreenState("TL", "rGrG")

def run_eval_episode(qnet, max_steps=1000):
    # 重新启动仿真 (可以自行决定配置文件或流量文件)
    traci.load(["-c", "4way.sumocfg"])
    # 记录所有车辆的出场时间 {veh_id: start_time}
    start_times = {}

    for step in range(max_steps):
        # 统计新出现的车辆, 记录其start_time
        for vid in traci.simulation.getDepartedIDList():
            start_times[vid] = traci.simulation.getTime()

        # 根据网络决策
        state = get_state()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = qnet(state_t)
            action = q_values.argmax(dim=1).item()
        set_traffic_light(action)

        traci.simulationStep()

        if traci.simulation.getMinExpectedNumber() <= 0:
            # 没有新车会出现，且所有车辆都到达终点或退出了
            break

    # 统计每辆车的travel time
    end_time = traci.simulation.getTime()
    final_travel_times = []
    for vid, stime in start_times.items():
        # 如果车已经抵达, getArrivalTime(vid) 可以拿到到达时间
        # 如果车还在路网里可以把当前time当做到达时间(不完美,只是演示)
        if traci.vehicle.getIDCount() > 0 and vid in traci.vehicle.getIDList():
            arrival_t = end_time
        else:
            arrival_t = traci.vehicle.getArrivalTime(vid)
        tt = arrival_t - stime
        final_travel_times.append(tt)

    traci.close()

    if len(final_travel_times) == 0:
        return 0.0
    avg_tt = sum(final_travel_times) / len(final_travel_times)
    return avg_tt

if __name__ == "__main__":
    # 1. 初始化并加载训练好的权重
    qnet = QNetwork(state_dim=8, action_dim=2)
    qnet.load_state_dict(torch.load("/home/local/ASURITE/longchao/Desktop/project/testbed_demo/rl_dqn/save/qnet_checkpoint.pth"))
    qnet.eval()

    # 2. 运行一个或多个episode来评估
    n_eval = 5
    avg_tts = []
    for i in range(n_eval):
        start_sumo("4way.sumocfg")
        traci.close()  # 先关, 避免冲突

        avg_tt = run_eval_episode(qnet, max_steps=1000)
        avg_tts.append(avg_tt)
        print(f"Test Episode {i+1}, average travel time = {avg_tt:.2f}")

    overall_avg_tt = sum(avg_tts) / len(avg_tts)
    print(f"Overall average travel time across {n_eval} runs: {overall_avg_tt:.2f}")
