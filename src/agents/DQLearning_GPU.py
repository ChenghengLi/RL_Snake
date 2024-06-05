import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game.snake import SnakeGame
from tqdm import tqdm
import json

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepQLearning:
    def __init__(self, game: SnakeGame, alpha=0.0001, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, tau=0.001):
        # Set device to GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.policy_net = DQN(6, 3).to(self.device)  # Move model to the correct device
        self.target_net = DQN(6, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64
        self.memory_capacity = 100000
        self.update_target_steps = 1000
        self.steps_done = 0

    def get_actions(self):
        return ["left", "right", "forward"]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.get_actions())
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action_index = torch.argmax(q_values).item()
            return self.get_actions()[action_index]

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor([self.get_actions().index(action) for action in actions], dtype=torch.int64, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states_tensor).max(1)[0]
        target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_steps == 0:
            self.update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, num_episodes, max_steps, reward_function, verbose=False):
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
                reward = reward_function(state, next_state, action, self.game.get_food(), eaten, game_over, score, step)
                if step == max_steps - 1:
                    reward = -50
                self.store_transition(state, action, reward, next_state, game_over)
                self.learn()
                state = next_state
                step = 0 if eaten else step + 1
            scores.append(score)
            if episode % 1000 == 0 and verbose:
                print(f'Episode {episode} finished - Max Score: {max(scores)} - Last Score: {score} - Mean Score: {sum(scores)/len(scores)}')
        print(max(scores))
        return self.policy_net

    def get_movement(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_tensor)
        action_index = torch.argmax(q_values).item()
        return self.get_actions()[action_index]

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval()

