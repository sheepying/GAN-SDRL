import numpy as np

from Agent_GDQN import GDQN
import torch
import os
import torch.optim as optim
from copy import deepcopy as dc
from env.utils import calculate_dis
from buffer import buffer ######buffer
from torch.distributions import Normal, Categorical
import torch.nn.functional as F

import collections
class Muti_agent():
    def __init__(self, params):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_steps_num=collections.deque(maxlen=10)

        self.params=params
        self.train_model = self.params["train_model"]
        
        ##创建buffer_dic
        self.replay_buffer_dict = {}
        for pursuer_id in self.params["pursuer_ids"]:
            self.replay_buffer_dict[pursuer_id] = buffer(params)

        # self.replay_buffer=buffer(params)
        self.agent = GDQN(self.params)



    def process_state(self,state):
        #return all_states dic for agent networks:{pursuer_id: {ego_pos: [ego_pos], target_pos:[target_pos],traffic_state: [traffic_state],topo_link_array: [topo_link_array],all_evaders_pos:[all_evaders_pos] } }
        #pro_state add"target"
        pro_state = dc(state)
        all_states={}

        if not ("target" in state):
            pursuer_target = {}
        else:
            pursuer_target=state["target"]

        all_evaders_pos = []
        for evader_id in self.params["evader_ids"]:
            all_evaders_pos.append(dc(state["evader_pos"][evader_id]))

        all_evaders_pos=[all_evaders_pos]
        traffic_state=[dc(state["background_veh"])]
        topo_link_array=[dc(state["topology_array"])]

        for pursuer_id in self.params["pursuer_ids"]:
            ego_pos = [dc(state["pursuer_pos"][pursuer_id])]
            if "target" in state:
                target_pos=[dc(state["evader_pos"][state["target"][pursuer_id]])]
            else:
                ego_pos_tensor = torch.tensor(ego_pos, dtype=torch.float,device=self.device)
                all_evaders_pos_tensor = torch.tensor(all_evaders_pos, dtype=torch.float,device=self.device)
                # target_code = self.agent.DQN_Net.select_target(ego_pos_tensor,
                #                                                                   all_evaders_pos_tensor)[0]
                # if state["evader_pos"][self.params["evader_ids"][target_code]][0] == -1:
                target_id = self.min_dis_evader(state, pursuer_id)  # TODO：
                # else:
                #     target_id=self.params["evader_ids"][target_code]
                pursuer_target[pursuer_id]=target_id
                target_pos = [dc(state["evader_pos"][target_id])]
            ego_state={
                "ego_pos":ego_pos,
                "target_pos":target_pos,
                "traffic_state":traffic_state,
                "topo_link_array":topo_link_array,
                "all_evaders_pos":all_evaders_pos,
                "steps":[[dc(state["steps"])]]
            }
            all_states[pursuer_id]= ego_state
        pro_state["target"]=dc(pursuer_target)
        return pro_state,all_states


    def select_action(self,pro_state, pro_all_states):
        # pro_state, pro_all_states=self.process_state(state)
        actions={}
        actions_prob={}
        for pursuer_id in self.params["pursuer_ids"]:
            ego_state=pro_all_states[pursuer_id]
            action,action_prob=self.agent.select_action(ego_state)
            actions[pursuer_id]=action
            actions_prob[pursuer_id]=action_prob
        return actions,actions_prob,pro_state["target"]


    def min_dis_evader(self,state,pursuer_id):
        min_dis = float('inf')
        min_evader_id = None
        for eva_index, evader_id in enumerate(self.params["evader_ids"]):
            if state["evader_pos"][evader_id][0] !=-1:
                eva_x, eva_y = state["evader_xy"][evader_id]["x"], state["evader_xy"][evader_id]["y"]
                dis = calculate_dis(eva_x, eva_y,
                                          state["pursuer_xy"][pursuer_id]["x"], state["pursuer_xy"][pursuer_id]["y"])
                if dis <= min_dis:
                    min_dis = dis
                    min_evader_id = evader_id
        return min_evader_id
    
    def Cumul_R(self):
        r_dic={}
        R_dic={}
        for pursuer_id in self.params["pursuer_ids"]:
            r_list = []
            R_list = []
            R = 0 
            mp_id = dc(self.replay_buffer_dict[pursuer_id].memory_pool)
            for i in range(len(mp_id)):
                r_list.append(mp_id[i]["reward"])
            r_dic[pursuer_id] = r_list

            for r in r_list:
                R = R + r[0]
                R_list.insert(0,R)
                
            R_dic[pursuer_id] = R_list
        
        return R_dic

    def train_agents(self):
        print("prepare for training......")
        Cumul_Reward = self.Cumul_R()
        Qloss, Gloss, Dloss, Wd = [], [], [], []
        for pursuer_id in self.params["pursuer_ids"]:
            train_set = dc(self.replay_buffer_dict[pursuer_id].memory_pool)       
            Q_loss, G_loss, D_loss, W_d =self.agent.update(train_set,Cumul_Reward[pursuer_id])
            del self.replay_buffer_dict[pursuer_id].memory_pool[:]  #clear the current buffer
            Qloss.append(Q_loss)
            Gloss.append(G_loss)
            Dloss.append(D_loss)
            Wd.append(W_d)
            # if critic_loss<self.agent_list[pursuer_id].best_critic_loss:
            #     self.agent_list[pursuer_id].save_critic_param()
            #     self.agent_list[pursuer_id].best_critic_loss=critic_loss
        print("#####loss list: Q, G, D, W######", Qloss, Gloss, Dloss, Wd)
        return np.array(Qloss).mean(), np.array(Gloss).mean(), np.array(Dloss).mean(), np.array(Wd).mean()

    def select_train_set(self, net_param):
        buffer_length = self.replay_buffer.get_length()
        ego_pos_input = np.zeros((buffer_length, self.params["max_steps"], self.params["lane_code_length"] + 1))
        eva_pos_input = np.zeros((buffer_length, self.params["max_steps"], self.params["num_evader"],
                                  self.params["lane_code_length"] + 1))
        action_input = np.zeros((buffer_length, self.params["max_steps"]))
        reward_input = np.zeros((buffer_length, self.params["max_steps"]))
        param_input = dc(net_param)
        # critic_param_input=dc(critic_param)
        for i in range(buffer_length):
            if i != 0:
                param_input = torch.cat((param_input, dc(net_param)), 0)
                # critic_param_input = torch.cat((critic_param_input, dc(critic_param)), 0)
            exp = dc(self.replay_buffer.memory_pool[i])
            if "ego_pos_eva_input" not in list(exp.keys()):
                exp_steps = len(exp["action"])
                ego_pos_input[i][0:exp_steps] = exp["state"]["ego_pos"]
                action_input[i][0:exp_steps] = exp["action"]
                reward_input[i][0:exp_steps] = exp["reward"]
                # print(eva_pos_input[i][:][0:exp_steps][:].shape)
                # print(np.array(exp["state"]["all_evaders_pos"]).shape)
                eva_pos_input[i][0:exp_steps] = exp["state"]["all_evaders_pos"]

                # for evader in range(len(self.params["evader_ids"])):
                #     for step in range(exp_steps):
                #         # a=exp["state"]["all_evaders_pos"][step][:]
                #         # print(a)
                #         eva_pos_input[i][evader][step][:] = exp["state"]["all_evaders_pos"][step][evader][:]
                self.replay_buffer.memory_pool[i]["ego_pos_eva_input"] = dc(ego_pos_input[i])
                self.replay_buffer.memory_pool[i]["eva_pos_eva_input"] = dc(eva_pos_input[i])
                self.replay_buffer.memory_pool[i]["action_eva_input"] = dc(action_input[i])
                self.replay_buffer.memory_pool[i]["reward_eva_input"] = dc(reward_input[i])
            else:
                # print("=====")
                ego_pos_input[i] = exp["ego_pos_eva_input"]
                eva_pos_input[i] = exp["eva_pos_eva_input"]
                action_input[i] = exp["action_eva_input"]
                reward_input[i] = exp["reward_eva_input"]
        self.evaluate_exp_net.eval()
        value = self.evaluate_exp_net(param_input.type(torch.float32).to(self.device),
                                      torch.from_numpy(ego_pos_input).type(torch.float32).to(self.device),
                                      torch.from_numpy(eva_pos_input.swapaxes(1, 2)).type(torch.float32).to(
                                          self.device),
                                      torch.from_numpy(reward_input).type(torch.float32).to(self.device),
                                      torch.from_numpy(action_input).type(torch.float32).to(self.device)).T
        # print("=================")
        # print(value)
        value = F.softmax(value, dim=1)
        # print(value)
        # print("=================")
        c = Categorical(value)
        exp_index = c.sample().item()
        chosen_exp = {
            "param_input": net_param.type(torch.float32).view(1, -1),
            "ego_pos_input": torch.from_numpy(np.expand_dims(ego_pos_input[exp_index], axis=0)).type(torch.float32),
            "eva_pos_input": torch.from_numpy(np.expand_dims(eva_pos_input[exp_index].swapaxes(0, 1), axis=0)).type(
                torch.float32),
            "reward_input": torch.from_numpy(np.expand_dims(reward_input[exp_index], axis=0)).type(torch.float32),
            "action_input": torch.from_numpy(np.expand_dims(action_input[exp_index], axis=0)).type(torch.float32)
        }
        return self.replay_buffer.memory_pool[exp_index], chosen_exp

    def train_evaluate_net(self):

        all_loss = 0

        if self.evaluate_net_buffer.get_length()>self.params["evaluate_net_batch_size"]*0:
            print("training evaluate network......")
            for i in range(self.params["evaluate_net_update_times"]):
                self.evaluate_exp_net.eval()
                param_input, \
                ego_pos_input, eva_pos_input, \
                reward_input, action_input, target_output=\
                    self.evaluate_net_buffer.get_train_batch(self.params["evaluate_net_batch_size"])

                output =self.evaluate_exp_net(param_input.to(self.device),ego_pos_input.to(self.device),eva_pos_input.to(self.device),reward_input.to(self.device),action_input.to(self.device))
                loss = torch.nn.functional.mse_loss(output.view(-1,1), target_output.view(-1,1).to(self.device))
                self.evaluate_net_optimizer.zero_grad()
                loss.backward()
                self.evaluate_net_optimizer.step()
                all_loss += loss.item()
            self.save_evaluate_net(all_loss/self.params["evaluate_net_update_times"])
        return all_loss/self.params["evaluate_net_update_times"]

    def save_evaluate_net(self,loss):
        dir_path = 'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.evaluate_exp_net.state_dict(),
                   'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]+'/' +'evaluate_net.pth')

    def load_evaluate_net(self):
        file_path = 'agent_param/' + self.params["env_name"] + '/' + self.params[
            "exp_name"] + '/' +'evaluate_net.pth'
        if os.path.exists(file_path):
            print("loading evaluate_net from param file....")
            self.evaluate_exp_net.load_state_dict(torch.load(file_path))
            self.evaluate_exp_net.to(self.device)
        else:
            print("creating new param for evaluate_net....")

    def load_params(self):
        # for pursuer_id in self.params["pursuer_ids"]:
        self.agent.load_param()

    # def load_critics_param(self):
    #     for pursuer_id in self.params["pursuer_ids"]:
    #         self.agent_list[pursuer_id].load_critic_param()
    #
    # def load_all_networks(self):
    #     self.load_actors_param()
    #     self.load_critics_param()









