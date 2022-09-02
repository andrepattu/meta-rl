# Meta-RL
This README contains the instructions to setup an environment and run the code for my dissertation on "An Exploration of Policy Gradient Methods to Generalise Learning in Meta Reinforcement Learning. 

# Setup
Set up virtual python environment using conda or venv. Activate the environment and run `pip install -r requirements.txt'

Potential additional steps:
- If torch module is not found, install pytorch according to the instructions on pytorch.org
- You may be required to `accept-rom-license` when installing gym
- You may be required to install `Box2D-kengz` for the gym environment

# Usage
## PPO
To train ppo from scratch:
`python main_ppo.py`

To test ppo:
`python main_ppo.py --mode test --actor_model <ppo_actor.pth>`

To continue training with existing actor and critic models:
`python main_ppo.py --actor_model <ppo_actor.pth> --critic_model <ppo_critic.pth>`

## DDPG
To train ddpg from scratch:
`python main_ddpg.py`

To test ddpg:
`python main_ddpg.py --mode test --actor_model <ddpg_actor.pth>`

To continue training with existing actor and critic models:
`python main_ddpg.py --actor_model <ddpg_actor.pth> --critic_model <ddpg_critic.pth>`

## Meta-PPO
To meta-train meta-PPO from scratch:
`python main_meta_ppo.py`

To meta-test meta-PPO:
`python main_meta_ppo.py --mode test --loss_fn <loss_fn.pth>`
