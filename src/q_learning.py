from game.snake import Direction, Point, SnakeGame
import random


class QLearning():
    def __init__(self, game: SnakeGame, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.game = game
        self.q_table = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def get_best_action(self, state):
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return random.choice(self.game.actions)

        best_action = None
        best_value = float('-inf')
        for action in self.game.actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = self.get_best_action(next_state)
        best_next_q_value = self.get_q_value(next_state, best_next_action)
        td_target = reward + self.gamma * best_next_q_value
        td_delta = td_target - self.get_q_value(state, action)
        new_q_value = self.get_q_value(state, action) + self.alpha * td_delta
        self.set_q_value(state, action, new_q_value)

    def train(self, num_episodes, max_steps):
        for episode in range(num_episodes):
            self.game.reset()
            state = self.game.get_state()
            for _ in range(max_steps):
                action = self.get_best_action(state)
                reward, game_over = self.game.play_step(action)
                next_state = self.game.get_state()
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                if game_over:
                    break

    def play(self):
        self.game.reset()
        state = self.game.get_state()
        while True:
            action = self.get_best_action(state)
            reward, game_over = self.game.play_step(action)
            state = self.game.get_state()
            if game_over:
                break