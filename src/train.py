from game import SnakeGame
from rewards import manhattan_reward, naive_reward
from qlearning import QLearning
from deepQL import DeepQLearning
import random
game = SnakeGame(200, 200)
from tqdm import tqdm


policy_naive = QLearning(game).train(5000, 200, naive_reward)
#print(policy)

policy_manhattan = QLearning(game).train(5000, 200, manhattan_reward)




def benchmark(policy):
    game = SnakeGame(200, 200)
    scores = []
    for _ in tqdm(range(1000)):

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
                break

    print('Mean Score:', sum(scores)/len(scores))


"""CLI to play the snake game manually"""
import pygame
from game.snake import SnakeGame, player_to_snake_perspective



def play_snake_1():
    """Initialize and run the game loop"""
    pygame.init()

    game = SnakeGame(640, 400)

    speed = 20
    clock = pygame.time.Clock()
    stop = False

    # game loop
    while True:
        state = game.get_state()
        action = max(policy_naive[state], key = lambda x:policy_naive[state][x])

        _, score, game_over = game.play_step(action)
        game.pygame_draw()
        clock.tick(speed)

        if game_over:
            print('Game Over')
            break
            

    print('Final Score', score)
    pygame.quit()






if __name__ == '__main__':
    benchmark(policy_naive)
    benchmark(policy_manhattan)

