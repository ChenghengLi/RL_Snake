def naive_reward(state, next_state, action, food, eaten, game_over):
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

def manhattan_reward(state, next_state, action, food, eaten, game_over):
    dist_1 = manhattan_distance(state, food)
    dist_2 = manhattan_distance(next_state, food)
    
    if game_over:
        return -100
    elif eaten:
        return 30
    elif dist_2 < dist_1:
        return 1
    else:
        return -5


