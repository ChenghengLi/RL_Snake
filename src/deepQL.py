import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from game.snake import Direction, Point, SnakeGame
from tqdm import tqdm

# Define the neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Manhattan distance function
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Reward function
def manhattan_reward(state, next_state, food, eaten, game_over):
    dist_1 = manhattan_distance(state[:2], food)
    dist_2 = manhattan_distance(next_state[:2], food)
    
    if game_over:
        return -100
    elif eaten:
        return 30
    elif dist_2 < dist_1:
        return 1
    else:
        return -5

class DeepQLearning:
    def __init__(self, game: SnakeGame, alpha=0.001, gamma=0.9, epsilon=0.1, buffer_size=10000, batch_size=64):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        input_dim = 6  
        output_dim = 3  
        self.policy_net = DQN(input_dim=input_dim, output_dim=output_dim).to(self.device)
        self.target_net = DQN(input_dim=input_dim, output_dim=output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_actions(self):
        return ["left", "right", "forward"]

    def state_to_tensor(self, state):
        x, y, direction = state
        direction_one_hot = [0, 0, 0, 0]
        if direction == Direction.UP:
            direction_one_hot[0] = 1
        elif direction == Direction.DOWN:
            direction_one_hot[1] = 1
        elif direction == Direction.LEFT:
            direction_one_hot[2] = 1
        elif direction == Direction.RIGHT:
            direction_one_hot[3] = 1
        return torch.tensor([x, y] + direction_one_hot, dtype=torch.float32).to(self.device)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.get_actions())
        else:
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return self.get_actions()[torch.argmax(q_values).item()]

    def store_experience(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def sample_experiences(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_experiences()
        states = torch.stack([self.state_to_tensor(state) for state in states])
        next_states = torch.stack([self.state_to_tensor(state) for state in next_states])
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states)
        q_value = q_values.gather(1, torch.tensor(actions).unsqueeze(1).to(self.device)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, max_steps, reward_function=manhattan_reward):
        scores = []
        for episode in tqdm(range(num_episodes)):
            self.game.reset()
            state = self.game.get_state()
            game_over = False
            step = 0
            while not game_over and step < max_steps:
                action = self.choose_action(state)
                eaten, score, game_over = self.game.play_step(action)
                next_state = self.game.get_state()
                reward = reward_function(state, next_state, action, self.game.get_food(), eaten, game_over)
                self.store_experience(state, action, reward, next_state, game_over)
                self.train_step()
                state = next_state
                step = 0 if eaten else step + 1
            scores.append(score)
            if episode % 1000 == 0:
                print(f'Episode {episode} finished - Max Score: {max(scores)} - Last Score: {score} - Mean Score: {sum(scores)/len(scores)}')
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        return self.policy_net

    def get_movement(self, state):
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return self.get_actions()[torch.argmax(q_values).item()]

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(torch.load(filename))
        return self.policy_net