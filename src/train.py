from game import SnakeGame
from rewards import manhattan_reward, naive_reward, advanced_naive_reward, euclidean_reward
from qlearning import QLearning
from deepQL import DeepQLearning
import random
game = SnakeGame(200, 200)
from tqdm import tqdm



#policy_naive = DeepQLearning(game).train(5000, 200, naive_reward)
#policy_advanced_naive = QLearning(game).train(5000, 200, advanced_naive_reward)
#policy_manhattan = QLearning(game).train(5000, 200, manhattan_reward)
model = DeepQLearning(game)
#model.load_model('manhattan_policy.txt')

policy_euclidean = model.train(500, 200, manhattan_reward)
#model.save_model('manhattan_policy.txt')
#print(policy)



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

    game = SnakeGame(400, 400)

    speed = 20
    clock = pygame.time.Clock()
    stop = False

    # game loop
    for i in range(10):
        game.reset()
        game_over = False
        while True:
            state = game.get_state()
            action = model.get_movement(state)

            _, score, game_over = game.play_step(action)
            game.pygame_draw()
            clock.tick(speed)

            if game_over:
                print('Game Over')
                break
            

        print('Final Score', score)
    pygame.quit()






if __name__ == '__main__':
    play_snake_1()
    #benchmark(policy_naive)
    #benchmark(policy_advanced_naive)
    #benchmark(policy_manhattan)
    #benchmark(policy_euclidean)

