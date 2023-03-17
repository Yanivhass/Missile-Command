import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        '''
        state_dim: dimension of state
        action_dim: dimension of action
        max_action: maximum value of action
        hidden_dim: dimension of hidden layer
        '''
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.init_weights()

    def init_weights(self):
        '''
        Initialize weights of network
        '''
        torch.nn.init.kaiming_normal_(self.l1.weight)
        torch.nn.init.kaiming_normal_(self.l2.weight)
        torch.nn.init.kaiming_normal_(self.mean_linear.weight)
        torch.nn.init.kaiming_normal_(self.log_std_linear.weight)
        self.l1.bias.data.fill_(0.01)
        self.l2.bias.data.fill_(0.01)
        self.mean_linear.bias.data.fill_(0.01)
        self.log_std_linear.bias.data.fill_(0.01)

    def forward(self, state):
        '''
        Forward propagation
        state: state
        '''
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        '''
        Sample action from policy
        state: state
        Returns: action sampled from policy
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.max_action
        return action, log_prob, mean


class Qnetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        '''
        state_dim: dimension of state
        action_dim: dimension of action
        hidden_dim: dimension of hidden layer
        '''
        super(Qnetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.init_weights()

    def init_weights(self):
        '''
        Initialize the weights of the network
        '''
        self.fc1.weight.data.kaiming_normal_(nonlinearity='relu')
        self.fc2.weight.data.kaiming_normal_(nonlinearity='relu')
        self.fc3.weight.data.kaiming_normal_(nonlinearity='relu')
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, state):
        '''
        Forward pass of the network
        Returns: Action distribution
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x