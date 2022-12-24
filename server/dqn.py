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
import matplotlib.pyplot as plt

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.9
buffer_limit  = 10000
batch_size    = 8
EPISODE = 4000
print_interval = 50

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
        self.fc1 = nn.Linear(126, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 29)

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
                       [1,0,0,1,1, 0,0],#i=5
                       [0,1,1,1,0, 0,0],
                       [0,1,1,0,1, 0,0],
                       [0,1,0,1,1, 0,0],
                       [0,0,1,1,1, 0,0],
                       [2,0,0,0,0, 0,0],#i=10
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
        def check_avaiable_gems(gem_index):
            gem_count = state_dict['player_state'][4][gem_index]
            if gem_count < 2:
                for action_num in possible_action_list[:]:
                    if self.action[action_num][gem_index] > gem_count:
                        possible_action_list.remove(action_num)

        possible_action_list = []
        
        # 보석 획득하는 경우
        if sum(state_dict['player_state'][1]) < 8:
            for i in range(0, 15):
                possible_action_list.append(i)

        elif sum(state_dict['player_state'][1]) < 9:
            for i in range(10, 15):
                possible_action_list.append(i)
            
        # 가져올 수 있는 보석보다 많이 가져오는 액션 제거
        for i in range(5):
            check_avaiable_gems(i)


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
        coin = random.random()

        possible_action_list = self.filter_action(state_dict)
        if coin < epsilon:
            action_num = random.randint(0, len(possible_action_list) - 1)
            return possible_action_list[action_num]

        else:
            #out 중 가능한 값 key, value 추출
            possible_out = dict()
            for i in possible_action_list:
                possible_out[i] = float(out[i])
            action_num = max(possible_out, key=possible_out.get)
            return action_num



    def train(self, q, q_target, memory, optimizer):
        for i in range(10):
            s,a,r,s_prime,done_mask = memory.sample(batch_size)
            q_out = q(s)
            q_a = q_out.gather(1,a) #모델이 예측한 a의 Q value
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) #다음 상황에서의 max Q value
            target = r + gamma * max_q_prime  * done_mask
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

    score = 0.0
    reward = 0
    average_turn = 0

    agent = Agent()

    score_list  = []
    reward_list = []
    turn_list = []
    epsilon_list = []

    for n_epi in range(EPISODE):
        epsilon = max(0.02, 0.2 - 0.02*(n_epi/100))
        s = env.reset()
        state_dict = s
        s = state2np(s)

        done = False
        turn = 0
        while not done:
            turn += 1
            s_tensor = torch.from_numpy(s).float()
            a = agent.select_action(s_tensor, epsilon, state_dict)
            #print(f"Trun {turn} My Cards: {state_dict['player_state'][0]}|My Gems: {state_dict['player_state'][1]} My score: {state_dict['score'][0]}")
            #print(f"cards: {state_dict['cards'][0]}")
            #print(f'selected action: {agent.action[a]}')
            s_prime, r, done, info = env.step(agent.action[a])
            state_dict = s_prime
            s_prime = state2np(s_prime)
            done_mask = 0.0 if done else 1.0
            agent.memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            reward += r
            #print(f'reward: {reward}')
            
            if done:
                #if n_epi%20 == 1:
                #print(f"epi {n_epi} My Cards: {state_dict['player_state'][0]}|My Gems: {state_dict['player_state'][1]} My score: {state_dict['score'][0]} Turn to end: {turn}")
                average_turn += turn
                score += state_dict['score'][0]
                turn =  0
                break
        if agent.memory.size()>1000:
            agent.train(agent.model, agent.target_model, agent.memory, agent.optimizer)
        #torch.save(agent.model, "./weight/model.pt")

        if n_epi%print_interval==0 and n_epi!=0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            print("epi {}, reward : {:.1f}, turn: {:.1f}, score : {}, eps : {:.1f}%, buffer: {}".format(
                n_epi, reward/print_interval, average_turn/print_interval, score/print_interval, epsilon*100, agent.memory.size()))
                 
            reward_list.append(reward/print_interval)
            score_list.append(score/print_interval)
            turn_list.append(average_turn/print_interval)
            epsilon_list.append(epsilon*100)
            
            
            reward = 0
            score = 0
            average_turn = 0

    episodes = [i for i in range(print_interval, EPISODE, print_interval)]
    #plt.plot(episodes, reward_list, label = 'reward')
    plt.plot(episodes, score_list, label = 'score')
    #plt.plot(episodes, turn_list, label = 'turn to end')
    #plt.plot(episodes, epsilon_list, label = 'epsilon')

    plt.title('train result')
    plt.xlabel('episode')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()