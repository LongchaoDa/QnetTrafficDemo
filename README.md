# RL demo for intersection control

Example file: 
`testbed_demo/yiran_testfile.py`


## Replicated Rule Model: 

Material is at: `testbed_demo/rule_nn`

The network file is: 
`testbed_demo/rule_nn/train_rule_model.py`

The training file is (no need to train, just load and run): 
`testbed_demo/rule_nn/train_rule_model.py`

The test file is: 
`testbed_demo/rule_nn/load_test_rule_model.py`


The pre-saved model weight is: 
`testbed_demo/rule_nn/rule_model_entire.pth`



## Q-network RL policy is trained on a demo simulation. 

Simulation config: 

`testbed_demo/rl_dqn/config`

But no need to train by yourself, you can just directly load the pre-trained model (please learn from the script below): 
`testbed_demo/rl_dqn/test_q_case.py`

This scriipt is using the RL model to do the same test as the rule-based model. 

Just directly load the model weight from: 

`testbed_demo/rl_dqn/save/qnet_checkpoint.pth`

Besides, the training reward is saved as in image: 

`testbed_demo/rl_dqn/save/reward_plot.png`

Here the reward is just the negative average waiting time,. 