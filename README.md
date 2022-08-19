# Meta-RL

# Setup
Set up virtual environment using conda or venv. Activate the environment and run `pip install -r requirements.txt'

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

## Meta-DDPG
