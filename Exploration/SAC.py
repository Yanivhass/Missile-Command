import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models import Policy, Qnetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        self.policy = Policy(state_dim, action_dim, max_action, hidden_dim)
        self.critic = Qnetwork(state_dim, action_dim, hidden_dim)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy.sample(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_freq=2):

def save(self, filename, directory):
    torch.save(self.policy.state_dict(), '%s/%s_policy.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))