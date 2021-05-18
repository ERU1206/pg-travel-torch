import torch
from models import Actor, Critic
import gym
import numpy as np

# Entropy loss : H(X) = \mathbb E_X[I(x)] = - \sum_{x\in \mathbb X} p(x) log p(x)
# Maximization Entropy
class Agent:
    """

    ---
    >>> env = gym.make()
    >>> Agent(env, gamma, entropy_weight)
    """
    def __init__(self, env: gym.Env, hidden_dims = 128):
        self.env = env
        obs_dim = env.observation_space.shape[0] ## TODO 외우기
        action_dim = env.action_space.shape[0] ## TODO 외우기


        self.actor = Actor(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic = Critic(obs_dim, hidden_dims).to(self.device)# 넣어주시
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.done = False
        self.score = 0

        self.trainsition_store = list()

    def select_action(self, state:np.ndarray, train = 'train'):
        """
        여기 쫌 난해하네
        :param state:
        :param train:
        :return:
        """
        state_tensor = torch.FloatTensor(state).to(self.device)# 요런거 외우기
        action, dist = self.actor(state)
        action_map = {
            'train': action,
            'test' : dist.mean
                  }
        selected_action = action_map[train]
        log_prob = dist.log_prob(selected_action).sum(dim=-1)
        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy(),log_prob #
        # dist.log_prob(action_map)

    def train(self, number_frames, plotting_interval): #number frames = 500000 (얘가 총 프레임수인가벼), plotting interval = 100
        while 1-self.done:
            state = self.env.reset()
            for i in range(1, number_frames):
                action, log_prob = self.select_action(state, 'train')
                next_state, reward, done, info = self.env.step(action)
                self.env.render()


                state = next_state
            if done:
                state = self.env.reset()
                print(score)
                score = 0
            actor_loss, critic_loss = self.update_model()
        self.env.close()

    def log(self): #score, actor loss, critic loss to tensorboard
        NotImplemented

    def test(self):
        NotImplemented


