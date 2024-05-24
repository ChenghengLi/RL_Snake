from game.snake import Direction, Point, SnakeGame

def naive_reward(action, eaten, game_over):
    if eaten:
        if action == "forward":
            return 2
        elif action == "left" or action == "right":
            return 1
    elif game_over:
        if action == "forward":
            return -2
        elif action == "left" or action == "right":
            return -1
    return 0

def manhattan_distance_reward(action, eaten, game_over, prev_state, next_state, food):
    if game_over:
        return -100  # Large negative reward for dying
    if eaten:
        return 50  # Large positive reward for eating food

    # Calculate the Manhattan distance from the snake's head to the food before and after the action
    distance_after = abs(next_state.coordx - food.coordx) + abs(next_state.coordy - food.coordy)
    distance_before = abs(prev_state.coordx - food.coordx) + abs(prev_state.coordy - food.coordy) 

    # Reward based on getting closer to or farther from the food
    if distance_after < distance_before:
        return 10  # Positive reward for getting closer to the food
    else:
        return -10  # Negative reward for getting farther from the food