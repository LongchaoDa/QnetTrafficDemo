# train_and_save_rule_model.py

import torch
import sys
sys.path.append("testbed_demo/rule_nn")
from rule_model import RuleBasedModel  # 从你刚才的文件中导入类

if __name__ == "__main__":
    # 1. 初始化模型
    model = RuleBasedModel()

    # 2. （可选）训练或其他逻辑，这里省略；rule-based不需要真正训练

    # 3. 保存整个模型（结构+参数）
    torch.save(model, "testbed_demo/rule_nn/rule_model_entire.pth")
    print("Model has been saved as rule_model_entire.pth")
