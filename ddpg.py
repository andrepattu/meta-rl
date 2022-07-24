import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, x0=None, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.x0 = x0
        self.dt = dt

        # reset x_prev to x0
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self): # overwriting the call function
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __repr__(self):
        return f"OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})"

class ReplayBuffer(object):
    def __init__(self, max_size, in_dim, act_dim):
        self.mem_size = max_size
        self.mem_counter = 0 # memory counter
        self.state_mem = np.zeros((self.mem_size, *in_dim))
        self.new_state_mem = np.zeros((self.mem_size, *in_dim))
        self.act_mem = np.zeros((self.mem_size, act_dim))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_counter % self.mem_size
        self.state_mem[idx] = state
        self.new_state_mem[idx] = next_state
        self.act_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_memory[idx] = 1 - done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_mem[batch]
        actions = self.act_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.new_state_mem[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, in_dim, fc1_dims, fc2_dims, act_dim, name,
                 chkpt_dir='models'):
        super(CriticNetwork, self).__init__()
        self.in_dim = in_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.act_dim = act_dim
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg_test.pth')
        self.fc1 = nn.Linear(*self.in_dim, self.fc1_dims)
        init_val1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -init_val1, init_val1)
        nn.init.uniform_(self.fc1.bias.data, -init_val1, init_val1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        init_val2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -init_val2, init_val2)
        nn.init.uniform_(self.fc2.bias.data, -init_val2, init_val2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.act_val = nn.Linear(self.act_dim, self.fc2_dims)
        init_val3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -init_val3, init_val3)
        nn.init.uniform_(self.q.bias.data, -init_val3, init_val3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_val = self.fc1(state)
        state_val = self.bn1(state_val)
        state_val = F.relu(state_val)
        state_val = self.fc2(state_val)
        state_val = self.bn2(state_val)

        act_val = F.relu(self.act_val(action))
        state_act_val = F.relu(torch.add(state_val, act_val))
        state_act_val = self.q(state_act_val)

        return state_act_val

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alph, in_dim, fc1_dims, fc2_dims, act_dim, name,
                 chkpt_dir='models'):
        super(ActorNetwork, self).__init__()
        self.in_dim = in_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.act_dim = act_dim
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg_test.pth')
        self.fc1 = nn.Linear(*self.in_dim, self.fc1_dims)
        init_val1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -init_val1, init_val1)
        nn.init.uniform_(self.fc1.bias.data, -init_val1, init_val1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        init_val2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -init_val2, init_val2)
        nn.init.uniform_(self.fc2.bias.data, -init_val2, init_val2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        init_val3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.act_dim)
        nn.init.uniform_(self.mu.weight.data, -init_val3, init_val3)
        nn.init.uniform_(self.mu.bias.data, -init_val3, init_val3)

        self.optimizer = optim.Adam(self.parameters(), lr=alph)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class DDPG(object):
    def __init__(self, alph, beta, in_dim, tau, env_name, gamma=0.99,
                 act_dim=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, in_dim, act_dim)
        self.batch_size = batch_size

        self.env_name = env_name

        self.actor = ActorNetwork(alph, in_dim, layer1_size,
                                  layer2_size, act_dim=act_dim,
                                  name=env_name+'Actor')
        self.critic = CriticNetwork(beta, in_dim, layer1_size,
                                    layer2_size, act_dim=act_dim,
                                    name=env_name+'Critic')

        self.target_actor = ActorNetwork(alph, in_dim, layer1_size,
                                         layer2_size, act_dim=act_dim,
                                         name=env_name+'TargetActor')
        self.target_critic = CriticNetwork(beta, in_dim, layer1_size,
                                           layer2_size, act_dim=act_dim,
                                           name=env_name+'TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(act_dim))

        self.update_network_parameters(tau=1)

    def query_action(self, observation):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(),
                                 dtype=torch.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # convert to dictionaries for easier iteration
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        print("...saving models...")
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    # def check_actor_params(self):
    #     current_actor_params = self.actor.named_parameters()
    #     current_actor_dict = dict(current_actor_params)
    #     original_actor_dict = dict(self.original_actor.named_parameters())
    #     original_critic_dict = dict(self.original_critic.named_parameters())
    #     current_critic_params = self.critic.named_parameters()
    #     current_critic_dict = dict(current_critic_params)
    #     print('Checking Actor parameters')

    #     for param in current_actor_dict:
    #         print(param, torch.equal(original_actor_dict[param], current_actor_dict[param]))
    #     print('Checking critic parameters')
    #     for param in current_critic_dict:
    #         print(param, torch.equal(original_critic_dict[param], current_critic_dict[param]))
    #     input()