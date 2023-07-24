import torch.nn as nn
import torch.nn.functional as F
import torch
from .dgcnn_agent import DGCNNAgent, DGCNNAgent_logits

class SACAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SACAgent, self).__init__()
        self.args = args
        self.device = args.device
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        
        self.agents = torch.nn.ModuleList([DGCNNAgent(input_shape, args) for _ in range(self.n_agents)])
        #To include parameter sharing modify this code

        
    # TODO:
    def init_hidden(self):
        # make hidden states on same device as model
        #In out context, hidden states have no meaning as we dont have RNNs
        return torch.cat([a.init_hidden() for a in self.agents])

    # TODO: forward hidden_state
    # TODO: forward each agent
    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return torch.cat(qs), torch.cat(hiddens).unsqueeze(0)
        else:
            breakpoint()
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return torch.cat(qs, dim=-1).view(-1, q.size(-1)), torch.cat(hiddens, dim=1)

    def cuda(self):
        for a in self.agents:
            a.cuda(device=self.device)

class MLP_agent_ontop(nn.Module):
    def __init__(self, num_points, num_layers=1):
        super(MLP_agent_ontop, self).__init__()
        self.num_layers = num_layers
        self.num_points = num_points

        self.final_linear = nn.Linear(self.num_points, self.num_points)
        self.layers_list = nn.ModuleList([nn.Sequential(nn.Linear(self.num_points, self.num_points), 
                                                        nn.ReLU()) for _ in range(num_layers-1)])
        
    def forward(self, x):
        y = x.clone()

        for i in range(self.num_layers-1):
            y = self.layers_list[i](y)
        y = self.final_linear(y)
        return y


#TODO: This class implements a shared base DGCNN and gives separate MLPs on top of it to each agent.
class SACAgent_sharedbase(nn.Module):
    def __init__(self, input_shape, args):
        super(SACAgent_sharedbase, self).__init__()
        self.args = args
        self.device = args.device
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        #TODO:
        self.base_net = DGCNNAgent_logits(input_shape, args)
        self.net_on_top = torch.nn.ModuleList([MLP_agent_ontop(num_points=100, num_layers=2) for _ in range(self.n_agents)])
        
    # TODO:
    def init_hidden(self):
        # make hidden states on same device as model
        #In our context, hidden states have no meaning as we dont have RNNs
        return torch.cat([self.base_net.init_hidden() for a in self.net_on_top])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []

        #TODO: Check input size conditions
        if inputs.size(0) == self.n_agents:
            base_out, h = self.base_net(inputs[0], hidden_state[:,0])
            for i in range(self.n_agents):
                q = self.net_on_top[i](base_out)
                hiddens.append(h)
                qs.append(q)
            return torch.cat(qs), torch.cat(hiddens).unsqueeze(0)
        else:
            breakpoint()
        
    def cuda(self):
        self.base_net.cuda(device=self.device)
        for a in self.net_on_top:
            a.cuda(device=self.device)