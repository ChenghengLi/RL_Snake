from game.snake import Direction, Point, SnakeGame
import random
from tqdm import tqdm
from rewards import naive_reward, manhattan_distance_reward

class QLearning():
    def __init__(self, game: SnakeGame, alpha=0.1, gamma=0.6, epsilon=0.1, reward_function="naive_reward"):
        self.game = game
        self.q_table = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_function = reward_function


    def get_actions(self):
        return ["left", "right", "forward"]

    def add_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.get_actions()}

    def choose_action(self, state):        
        if random.random() < self.epsilon:
            return random.choice(self.get_actions())
        else:
            return max(self.q_table[state], key = lambda x:self.q_table[state][x])

    def learn(self, state, action, reward, next_state):
        self.add_state(next_state)
        max_next_action = max(self.q_table[next_state], key = lambda x:self.q_table[next_state][x])
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (reward + self.gamma * self.q_table[next_state][max_next_action] - self.q_table[state][action])
        pass


    def train(self, num_episodes, max_steps):
        for episode in tqdm(range(num_episodes)):
            self.game.reset()
            state = self.game.head
            self.add_state(state)
            game_over = False
            counter = 0
            prev_score = 0
            while not game_over:
                counter += 1
                action = self.choose_action(state)
                prev_state = self.game.head
                eaten, score, game_over = self.game.play_step(action)
                if score > prev_score:
                    prev_score = score
                    counter = 0
                next_state = self.game.head
                food = self.game.food

                # Change functionality based on reward function
                if self.reward_function == "naive_reward":
                    reward = naive_reward(action, eaten, game_over)
                elif self.reward_function == "manhattan_reward":
                    reward = manhattan_distance_reward(action, eaten, game_over, prev_state, next_state, food)

                if counter > max_steps:
                    reward = -2
                    game_over = True
                self.learn(state, action, reward, next_state)
                state = next_state

            
        return self.q_table


    def get_movement(self, state):
        return max(self.q_table[state], key = lambda x:self.q_table[state][x])
