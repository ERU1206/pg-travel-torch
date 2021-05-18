import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Actor is network that have 1 hidden layer
# 1st layer -> relu->
#                  ->
class Actor(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dimension = 128):
        """
        :param in_dims: Model input's dimension
        :param out_dims: Model output's dimension
        """
        super(Actor, self).__init__()

        self.hidden1= nn.Linear(in_dims, hidden_dimension)
        self.mu_layer = nn.Linear(hidden_dimension, out_dims)
        self.log_std_layer = nn.Linear(hidden_dimension, out_dims)

        self.initialize_uniformly(self.mu_layer)
        self.initialize_uniformly(self.log_std_layer)

    def forward(self, state:torch.Tensor):
        x = F.relu(self.hidden1(state))
        mu = torch.tanh(self.mu_layer(x))*2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()
        return action, dist

    def initialize_uniformly(self, layer:nn.Linear, init_w: float = 3e-3) :
        layer.weight.data.uniform_(-init_w, init_w) # 자기 자신 바꿀땐 _ 잊지 말자
        layer.bias.data.uniform_(-init_w, init_w)

# Critic is one layer network that have relu activate function
class Critic(nn.Module):
    def __init__(self, in_dims, hidden_dims,):
        super(Critic, self).__init__()
        self.hidden1 = nn.Linear(in_dims, hidden_dims)
        self.outputs = nn.Linear(hidden_dims, 1)
        self.initialize_uniformly(self.outputs)

    def forward(self, state):
        x = F.relu(self.hidden1(state))
        return self.out(x)

    def initialize_uniformly(self, layer:nn.Linear, init_w: float = 3e-3) :
        layer.weight.data.uniform_(-init_w, init_w) # 자기 자신 바꿀땐 _ 잊지 말자
        layer.bias.data.uniform_(-init_w, init_w)



