# load_and_test_rule_model.py

import torch

if __name__ == "__main__":
    # 1. 直接加载完整模型
    loaded_model = torch.load("testbed_demo/rule_nn/rule_model_entire.pth")
    loaded_model.eval()

    # 2. 示例测试数据
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

    # 3. 推理
    color_tensor = loaded_model(lane_bots, bots_obs_times)
    print("Predicted color list:", color_tensor.tolist())


# Predicted color list: [0, 1, 0, 1]