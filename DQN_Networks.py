import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN_net(nn.Module):
    def __init__(self,num_edge,num_pos,num_evader,num_action, params):
        super(DQN_net, self).__init__()
        self.params = params
        self.num_edge=num_edge
        self.num_pos=num_pos
        self.num_evader=num_evader
        self.num_action=num_action
        self.batch_size = self.params["GAN_batch_size"]
        self.fc_n1 = nn.Linear(num_pos+num_pos*self.num_evader+num_edge+1, 32)


        self.conv1_link=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=2,padding=1)
        self.conv2_link = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=3, padding=1)
        self.conv3_link = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.fc_link = nn.Linear(49, 24)
        

        self.fc_hid1 = nn.Linear(32+24, 48)
        self.fc_action = nn.Linear(48, num_action)


    def forward(self, steps,ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos):
        # ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos
        
        all_input=torch.cat((steps,torch.flatten(ego_pos,1),torch.flatten(all_evaders_pos,1),torch.flatten(traffic_state,1)),1)
        feature = F.elu(self.fc_n1(all_input))
        topo=F.relu(self.conv1_link(topo_link_array.view(-1,1,self.num_edge,self.num_edge)))
        topo = F.relu(self.conv2_link(topo))
        topo = F.relu(self.conv3_link(topo))
        topo=F.elu(self.fc_link(topo.view(-1,49)))

        all_features=torch.cat((feature,topo),1)
        all_features=F.elu(self.fc_hid1(all_features))
        Q_values = F.softmax(self.fc_action(all_features), dim=1)
        #print("######## Net Q_values:", Q_values.size(), Q_values)
        return Q_values
    