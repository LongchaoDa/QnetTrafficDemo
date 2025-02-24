# rule_based_model.py
import torch
import torch.nn as nn

class RuleBasedModel(nn.Module):
    def __init__(self):
        super(RuleBasedModel, self).__init__()

    def forward(self, lane_bots, bots_obs_times):
        # 这里就是你原先的 rule-based 逻辑
        longest_time_lane = {2: 0, 4: 0, 6: 0, 8: 0}
        for lane_id, bot_ids in lane_bots.items():
            for bot_id in bot_ids:
                if bots_obs_times[bot_id] > longest_time_lane[lane_id]:
                    longest_time_lane[lane_id] = bots_obs_times[bot_id]

        lane_times = [
            longest_time_lane[2],
            longest_time_lane[4],
            longest_time_lane[6],
            longest_time_lane[8]
        ]

        lane_nums = [
            len(lane_bots[2]),
            len(lane_bots[4]),
            len(lane_bots[6]),
            len(lane_bots[8])
        ]

        X = max(lane_times[0], lane_times[2]) + 2*(lane_nums[0] + lane_nums[2])
        Y = max(lane_times[1], lane_times[3]) + 2*(lane_nums[1] + lane_nums[3])

        if X > Y:
            color_tensor = torch.tensor([1, 0, 1, 0], dtype=torch.int)
        elif X < Y:
            color_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.int)
        else:
            color_tensor = torch.tensor([0, 1, 0, 1], dtype=torch.int)
        return color_tensor
