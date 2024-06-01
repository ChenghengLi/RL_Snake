from game import SnakeGame
from tqdm import tqdm
import pygame
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def benchmark(model, W, H):
    game = SnakeGame(W, H)
    scores = []
    for _ in tqdm(range(1000)):

        step = 0
        game_over = False
        game.reset()
        while not game_over:
            state = game.get_state()
            action = model.get_movement(state)
            eaten , score, game_over = game.play_step(action)
            step = 0 if eaten else step + 1
            if game_over:
                scores.append(score)
                break
            if step == 200:
                break
    return scores, sum(scores) / len(scores) if len(scores) > 0 else 0


# Plotting the training progress
def plot(y, x = None, title = None, xlabel = None, ylabel = None):
    if x is None:
        x = list(range(len(y)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y,x=x, mode='lines', name='Scores'))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel
    )
    fig.show()



def play_snake(model, W, H):
    """Initialize and run the game loop"""
    pygame.init()

    game = SnakeGame(W, H)

    speed = 100
    clock = pygame.time.Clock()
    stop = False

    # game loop
    for i in range(20):
        game.reset()
        game_over = False
        while True:
            state = game.get_state()
            action = model.get_movement(state)

            _, score, game_over = game.play_step(action)
            game.pygame_draw()
            clock.tick(speed)

            if game_over:
                print('Game Over => Score:', score)
                break
    pygame.quit()