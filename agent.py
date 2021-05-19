import torch
from models import Actor, Critic
import gym
import numpy as np
import environment
import doctest
import torch.nn.functional as F
import torch.optim as optim
# Entropy loss : H(X) = \mathbb E_X[I(x)] = - \sum_{x\in \mathbb X} p(x) log p(x)
# Maximization Entropy

ID = "Pendulum-v0" # configuration


### Deterministic 인지 Stochastic인지 continuous인지 드등?


class Agent:
    """

    ---
    # test scenario 1.
    >>> env = environment.gym_env(ID)
    >>> agent = Agent(env)
    >>> state = agent.env.reset()

    # >>> agent.select_action(state)

    >>> agent.train(100000)
    """

    def __init__(self, env: gym.Env, hidden_dims = 128):
        self.env = env
        obs_dim = env.observation_space.shape[0] ## TODO 외우기
        action_dim = env.action_space.shape[0]


        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = 0.95 # hyperparameter
        self.entropy_weight = 1e-2 # hyperparameter

        self.actor = Actor(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic = Critic(obs_dim, hidden_dims).to(self.device)# 넣어주시

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.done = False
        self.score = 0

        self.transition_store = list()

    def select_action(self, state:np.ndarray, train = 'train'):
        """
        여기 쫌 난해하네
        :param state:
        :param train:
        :return:
        """
        state_tensor = torch.FloatTensor(state).to(self.device)# 요런거 외우기
        action, dist = self.actor(state_tensor)
        action_map = {
            'train': action,
            'test' : dist.mean
                  }
        selected_action = action_map[train]
        log_prob = dist.log_prob(selected_action).sum(dim=-1) # entropy ### sum을 왜 하는겨????

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy(), log_prob #array([-2.], dtype=float32) # detach가 필요한 이유는?
        # selected_action.clamp(-2.0, 2.0).cpu().detach().numpy(),
        # dist.log_prob(action_map)

    def train(self, number_frames): #number frames = 500000 (얘가 총 프레임수인가벼), plotting interval = 100

        state = self.env.reset()
        for i in range(1, number_frames):
            self.env.render()
            action, log_prob = self.select_action(state, 'train')
            next_state, reward, done, info = self.env.step(action)
            self.transition_store.append((state, next_state, reward, done))
            state = next_state
            if done:
                # 얜 당연히 SGD임?
                self.update(self.transition_store)
                self.transition_store = []
                state = self.env.reset()

        self.env.close()

    def update(self, store):
        for experience in store:
            state, next_state, reward, done = experience
            next_state = torch.FloatTensor(next_state).to(self.device)
            state = torch.FloatTensor(state).to(self.device)

            pred_value = self.critic(state)
            targ_value = reward + self.gamma * self.critic(next_state) * (1-done)
            value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())

            # update value
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            # advantage = Q_t - V(s_t)
            _, log_prob = self.select_action(state, 'train')
            advantage = (targ_value - pred_value).detach()  # not backpropagated
            policy_loss = -advantage * log_prob
            policy_loss += self.entropy_weight * -log_prob  # entropy maximization

            # update policy
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        # return policy_loss.item(), value_loss.item()


    def log(self): #score, actor loss, critic loss to tensorboard
        NotImplemented
            # tensorboard에 기록 남기기

    def check(self):
        NotImplemented
            # 중간중간 저장하기?

    def test(self):
        NotImplemented
            # tensorboard에 영상 남기기


doctest.testmod()