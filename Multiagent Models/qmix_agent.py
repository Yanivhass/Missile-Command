import torch
import torch.nn as nn
import qmix
from qmix import QMixer


class qmix_agent(nn.Module):

    def __init__(self, defender_args, attacker_args, attacker_model, defender_model):
        super(qmix_agent, self).__init__()
        """

        Args:
            defender_args: Mixing arguments for attackers
            attacker_args: Mixing arguments for defenders
        """
        self.n_attackers = attacker_args.n_agents
        self.n_defenders = defender_args.n_agents
        self.model_att = attacker_model
        self.model_def = defender_model

        self.mixing_att = QMixer(attacker_args)
        self.mixing_def = QMixer(defender_args)

    def forward(self, attackers_obvs, def_obvs, states):
        """

        Args:
            attackers_obvs: Observations of all attackers
            def_obvs: Observation of all defenders
            states: The total state of the network

        Returns: Output of mixing - total Q function for both defenders and attackers

        """
        q_attackers = self.model_att(attackers_obvs)
        q_defenders = self.model_def(def_obvs)

        Qtot_attack = self.mixing_att(q_attackers, states)
        Qtot_defenders = self.mixing_def(q_defenders, states)

        return Qtot_attack, Qtot_defenders
