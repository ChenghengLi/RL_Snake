import random
from game.snake import SnakeGame
from tqdm import tqdm


class QLearning:
    def __init__(self, game: SnakeGame, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.game = game
        self.q_table = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    

    def get_actions(self):
        return ["left", "right", "forward"]

    def add_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.get_actions()}

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.get_actions())
        else:
            return max(self.q_table[state], key=lambda x: self.q_table[state][x])

    def learn(self, state, action, reward, next_state):
        self.add_state(next_state)
        max_next_action = max(self.q_table[next_state], key=lambda x: self.q_table[next_state][x])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][max_next_action] - self.q_table[state][action])

    def train(self, num_episodes, max_steps, reward_function, verbose=False):
        scores = []
        for episode in tqdm(range(num_episodes)):
            self.game.reset()
            state = self.game.get_state()
            self.add_state(state)
            game_over = False
            step = 0
            while not game_over and step < max_steps:
                action = self.choose_action(state)
                eaten, score, game_over = self.game.play_step(action)
                next_state = self.game.get_state()
                reward = reward_function(state, next_state, action, self.game.get_food(), eaten, game_over, score)
                if step == max_steps - 1:
                    reward = -50
                self.learn(state, action, reward, next_state)
                state = next_state
                step = 0 if eaten else step + 1
            scores.append(score)
            if episode % 1000 == 0 and verbose:
                print(f'Episode {episode} finished - Max Score: {max(scores)} - Last Score: {score} - Mean Score: {sum(scores)/len(scores)}')
        print(max(scores))
        return self.q_table

    def get_movement(self, state):
        return max(self.q_table[state], key=lambda x: self.q_table[state][x])

    def save_q_table(self, filename):
        with open(filename, 'w') as f:
            for key, value in self.q_table.items():
                f.write(f'{key}:{value}\n')

    def load_q_table(self, filename):
        with open(filename, 'r') as f:
            self.q_table = eval(f.read())
        return self.q_table
