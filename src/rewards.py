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