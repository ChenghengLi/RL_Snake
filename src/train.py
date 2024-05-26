from game import SnakeGame
from rewards import manhattan_reward, naive_reward
from qlearning import QLearning
from deepQL import DeepQLearning
import random
game = SnakeGame(200, 200)



policy_naive = QLearning(game).train(50000, 200, naive_reward)
#print(policy)

policy_manhattan = QLearning(game).train(50000, 200, manhattan_reward)




def benchmark(policy):
    scores = []
    for _ in range(10):

        step = 0
        game_over = False
        game.reset()
        while not game_over:
            step += 1
            state = game.get_state()
            if state not in policy:
                action = random.choice(["left", "right", "forward"])
            else:
                action = max(policy[state], key = lambda x:policy[state][x])
            _, score, game_over = game.play_step(action)

            if game_over:
                scores.append(score)

    print('Mean Score:', sum(scores)/len(scores))


"""CLI to play the snake game manually"""
import pygame

from game.snake import SnakeGame, player_to_snake_perspective


def play_snake():
    """Initialize and run the game loop"""
    pygame.init()

    game = SnakeGame()

    speed = 20

    stop = False

    game_over = False
    # game loop

    for _ in range(10):
        print("--------------------------------")
        print('Game:', _+1)
        step = 0
        game_over = False
        game.reset()
        while not game_over:
            step += 1
            state = game.get_state()
            if state not in policy:
                action = random.choice(["left", "right", "forward"])
            else:
                action = max(policy[state], key = lambda x:policy[state][x])
            _, score, game_over = game.play_step(action)

            if game_over:
                print('Score:', score, "- Steps:", step)
    print("--------------------------------")       


def play_snake_1():
    """Initialize and run the game loop"""
    pygame.init()

    game = SnakeGame()

    speed = 20
    clock = pygame.time.Clock()
    stop = False

    # game loop
    while True:
        state = game.get_state()
        print(state)
        action = max(policy[state], key = lambda x:policy[state][x])

        _, score, game_over = game.play_step(action)
        game.pygame_draw()
        clock.tick(speed)

        if game_over:
            print('Game Over')
            

    print('Final Score', score)
    pygame.quit()






if __name__ == '__main__':
    benchmark(policy_naive)
    benchmark(policy_manhattan)
