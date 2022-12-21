import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flask import Flask
from splendor_env import *
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(np.array(done_mask_lst))
    
    def size(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(121, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 27)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Agent():
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.model.load_state_dict(self.model.state_dict())
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.action = [[1,1,1,0,0, 0,0], # 보석 토큰 가져오기
                       [1,1,0,1,0, 0,0],
                       [1,1,0,0,1, 0,0],
                       [1,0,1,1,0, 0,0],
                       [1,0,1,0,1, 0,0],
                       [1,0,0,1,1, 0,0],
                       [0,1,1,1,0, 0,0],
                       [0,1,1,0,1, 0,0],
                       [0,1,0,1,1, 0,0],
                       [0,0,1,1,1, 0,0],
                       [2,0,0,0,0, 0,0],
                       [0,2,0,0,0, 0,0],
                       [0,0,2,0,0, 0,0],
                       [0,0,0,2,0, 0,0],
                       [0,0,0,0,2, 0,0],
                       
                       [0,0,0,0,0, 1,0], # i = 15 개발 카드 구매
                       [0,0,0,0,0, 1,1],
                       [0,0,0,0,0, 1,2],
                       [0,0,0,0,0, 1,3],
                       [0,0,0,0,0, 2,0],
                       [0,0,0,0,0, 2,1],
                       [0,0,0,0,0, 2,2],
                       [0,0,0,0,0, 2,3],
                       [0,0,0,0,0, 3,0],
                       [0,0,0,0,0, 3,1],
                       [0,0,0,0,0, 3,2],
                       [0,0,0,0,0, 3,3]] # i = 26
                       
    def filter_action(self, state_dict):
        possible_action_list = []
        
        # 보석 획득하는 경우
        if sum(state_dict['player_state'][1]) < 9:
            for i in range(15):
                possible_action_list.append(i)

        # 카드 구매하는 경우
        for i in range(15, 27):
            card_level = self.action[i][-2] - 1
            card_order = self.action[i][-1]

            #카드 구매 조건 확인
            card = state_dict['cards'][card_level][card_order]
            my_gems = state_dict['player_state'][1]
            my_cards = state_dict['player_state'][0]
            #구매 불가능할 경우 action 다시 선택
            if card == [0 for i in range(7)]:
                pass
            elif card[0] > my_gems[0] + my_cards[0] or card[1] > my_gems[1] + my_cards[1] or card[2] > my_gems[2] + my_cards[2] or card[3] > my_gems[3] +  my_cards[3] or card[4] > my_gems[4] + my_cards[4]:
                pass
            else:
                possible_action_list.append(i)
        
        return possible_action_list
        
    def select_action(self, obs, epsilon, state_dict):
        out = self.model.forward(obs)
        possible_action_list = self.filter_action(state_dict)
        action_num = random.randint(0, len(possible_action_list) - 1)

        return possible_action_list[action_num]

    def train(self, q, q_target, memory, optimizer):
        for i in range(10):
            s,a,r,s_prime,done_mask = memory.sample(batch_size)
            q_out = q(s)
            q_a = q_out.gather(1,a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            
            
def state2np(state_dict):
    new_state = np.array([])
    for state in state_dict.values():
        state = np.array(state).flatten()
        new_state = np.concatenate((new_state, state), axis=0)
    return new_state

def main():
    GM = GameManager("Aircraft")
    GM.join_game()
    # GM.join_game()
    GM.start_game()
    env = GM.game

    print_interval = 10
    score = 0.0
    average_turn = 0

    agent = Agent()

    
    for n_epi in range(200):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        state_dict = s
        s = state2np(s)

        done = False
        turn = 0

        while not done:
            turn += 1
            s_tensor = torch.from_numpy(s).float()
            a = agent.select_action(s_tensor, epsilon, state_dict)
            #print(a)
            s_prime, r, done, info = env.step(agent.action[a])
            state_dict = s_prime
            s_prime = state2np(s_prime)
            done_mask = 0.0 if done else 1.0
            agent.memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            
            if done:
                #print(f"epi {n_epi} My Cards: {state_dict['player_state'][0]}| My Gems: {state_dict['player_state'][1]} My score: {state_dict['score'][0]} Turn to end: {turn}")
                average_turn += turn
                turn =  0
                break
            time.sleep(0.01)
            if agent.memory.size()>1000: #train 시험용으로 낮춤
                agent.train(agent.model, agent.target_model, agent.memory, agent.optimizer)
        #print("Done!")
        #torch.save(agent.model, "./weight/model.pt")
        if n_epi%print_interval==0 and n_epi!=0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            print("n_episode :{}, score : {:.1f}, turn: {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, average_turn/print_interval, agent.memory.size(), epsilon*100))
            score = 0.0
            average_turn = 0
            
    #env.close()

if __name__ == '__main__':
    main()