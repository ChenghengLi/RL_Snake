import pygame
from game.snake import SnakeGame, player_to_snake_perspective
from agents.QLearning import QLearning

def agent_play(game, model):
    scores = []
    game_over = False
    step = 0
    speed = 500
    clock = pygame.time.Clock()
    while not game_over:

        state = game.get_state()
        action = model.get_movement(state)
        eaten, score, game_over = game.play_step(action)
        step = 0 if eaten else step + 1
        game.pygame_draw()
        clock.tick(speed)
        if game_over or step == 200:
            scores.append(score)
            break

    return scores[-1]

def manual_play(game):
    score = 0
    speed = 15
    clock = pygame.time.Clock()
    stop = False

    while not stop:
        action = "forward"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.quit()
                stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game.quit()
                    stop = True
                elif event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                    player_direction = {
                        pygame.K_LEFT: "left",
                        pygame.K_RIGHT: "right",
                        pygame.K_UP: "up",
                        pygame.K_DOWN: "down"
                    }[event.key]
                    action = player_to_snake_perspective(game.direction, player_direction)

        if stop:
            break

        _, score, game_over = game.play_step(action)
        game.pygame_draw()
        clock.tick(speed)

        if game_over:
            break

    return score

def display_endgame(game, agent_score, player_score):
    font = pygame.font.Font(None, 36)
    result_text = f"AI points {agent_score} : {player_score} Player points"
    winner_text = "It's a tie!" if agent_score == player_score else ("Player wins!" if player_score > agent_score else "Agent wins!")
    instruction_text = "Press space to play again or Esc to quit"

    result_surface = font.render(result_text, True, (0,0,0))
    winner_surface = font.render(winner_text, True, (0,0,0))
    instruction_surface = font.render(instruction_text, True, (0,0,0))

    result_rect = result_surface.get_rect(center=((game.width+game.margin*2)  // 2, game.height // 2 - 40))
    winner_rect = winner_surface.get_rect(center=((game.width+game.margin*2) // 2, game.height // 2))
    instruction_rect = instruction_surface.get_rect(center=(game.width // 2, game.height // 2 + 40))

    game.display.blit(result_surface, result_rect)
    game.display.blit(winner_surface, winner_rect)
    game.display.blit(instruction_surface, instruction_rect)
    pygame.display.flip()

def play_snake():
    pygame.init()
    game = SnakeGame()
    model = QLearning(game)
    filename = 'src/policies/QL/best_advanced_naive_policy.txt'
    model.load_model(filename)

    while True:
        # Agent plays first
        agent_score = agent_play(game, model)
        print('Agent Score:', agent_score)

        # Display "Now is your turn. Press space to start"
        font = pygame.font.Font(None, 36)
        text1 = font.render('Now is your turn. Press space to start', True, (0,0,0))
        text2 = font.render('Use arrows to move the snake', True, (0,0,0))
        text1_rect = text1.get_rect(center=((game.width+game.margin*2) // 2, game.height // 2 - 20))
        text2_rect = text2.get_rect(center=((game.width+game.margin*2) // 2, game.height // 2 + 20))
        game.display.blit(text1, text1_rect)
        game.display.blit(text2, text2_rect)
        pygame.display.flip()

        # Wait for user to press Space
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False

        game.reset()

        # Player plays
        player_score = manual_play(game)
        print('Player Score:', player_score)

        # Display endgame results
        display_endgame(game, agent_score, player_score)

        # Wait for user to press Space to replay or Esc to quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

        game.reset()

if __name__ == '__main__':
    play_snake()
