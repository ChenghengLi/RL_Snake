import math

def naive_reward(state, next_state, action, food, eaten, game_over, score, steps):
    if eaten:
        return 1
    elif game_over:
        return -1
    return 0


def advanced_naive_reward(state, next_state, action, food, eaten, game_over, score, steps):
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


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def manhattan_reward(state, next_state, action, food, eaten, game_over, score, steps):
    dist_1 = manhattan_distance(state, food)
    dist_2 = manhattan_distance(next_state, food)
    
    if game_over:
        return -5
    elif eaten:
        return 5
    elif dist_2 < dist_1:
        return 1
    else:
        return -1


def euclidean_reward(state, next_state, action, food, eaten, game_over, score, steps):
    # From https://dr.ntu.edu.sg/bitstream/10356/89882/1/ICA2018SnakeGame.pdf
    if game_over:
        return -1
    elif eaten:
        return 1 
    else:
        snake_size = 3 + score # Snake starts as size 3 and grows by 1 for every score earned
        # Calculate the Euclidean distance from the snake's head to the food before and after the action
        dist_1 = euclidean_distance(state, food)
        dist_2 = euclidean_distance(next_state, food)
        return math.log((snake_size+dist_1)/(snake_size+dist_2), snake_size)
