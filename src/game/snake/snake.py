"""SnakeGame. It's interface is suitable for humans an AI programs'"""
# pylint: disable=too-many-instance-attributes

# Taken from
# https://github.com/python-engineer/python-fun/blob/master/snake-pygame/snake_game.py
import random
from enum import Enum
from collections import namedtuple
import pygame


class Direction(Enum):
    """Direction enum used to know where the snake is heading towards"""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'coordx, coordy')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 50)
BLACK = (0, 0, 0)




class SnakeGame:
    """Simple 2D snake game"""

    def __init__(self, width=640, height=400):
        """Initialize the graphic blocks and a snake of size 3 heading right"""
        self.block_size = 20
        self.width = width
        self.height = height
        self.margin = 30

        self.display = None
        self.font = None

        self.direction = Direction.RIGHT

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head,
                      Point(self.head.coordx - self.block_size,
                            self.head.coordy),
                      Point(self.head.coordx - (2 * self.block_size),
                            self.head.coordy)]

        self.score = 0
        self.food = None
        self._place_food()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head,
                      Point(self.head.coordx - self.block_size,
                            self.head.coordy),
                      Point(self.head.coordx - (2 * self.block_size),
                            self.head.coordy)]

        self.score = 0
        self._place_food()

    def get_width(self):
        return self.width // self.block_size
    
    def get_height(self):
        return self.height // self.block_size
    
    def _place_food(self):
        """Place the food randomly in a non-colliding coordinate"""
        while True:
            coordx = random.randint(0, (self.width - self.block_size) //
                                    self.block_size) * self.block_size
            coordy = random.randint(0, (self.height - self.block_size) //
                                    self.block_size) * self.block_size
            self.food = Point(coordx, coordy)
            if self.food not in self.snake:
                break

    def _choose_direction(self, action):
        """Rotate the snake given the provided `action` string"""
        clockwise = [Direction.UP, Direction.RIGHT,
                     Direction.DOWN, Direction.LEFT]
        current = clockwise.index(self.direction)

        if action == "left":
            return clockwise[(current - 1) % 4]
        if action == "right":
            return clockwise[(current + 1) % 4]
        if action == "forward":
            return self.direction

        raise ValueError("Unknown action: " + str(action))

    def play_step(self, action):
        """Receive an action and execute its effects of moving and colliding"""
        if action not in ["left", "right", "forward"]:
            raise ValueError("Unknown action: " + str(action))

        self.direction = self._choose_direction(action)

        # move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)  # update the snake

        # check if game over
        eaten = False
        game_over = False
        if self.is_collision(self.head):
            game_over = True
            return eaten, self.score, game_over

        # place new food or just move
        if self.head == self.food:
            self.score += 1
            eaten = True
            self._place_food()
        else:
            self.snake.pop()

        # return game over and score
        return eaten, self.score, game_over

    def is_collision(self, point):
        """Returns true if the current state is a collision"""
        # hits boundary
        if point.coordx > self.width - self.block_size or \
                point.coordy > self.height - self.block_size or \
                point.coordx < 0 or point.coordy < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False
    
    def _get_point(self, direction: Direction):

        coordx = self.head.coordx
        coordy = self.head.coordy
        if direction == Direction.RIGHT:
            coordx += self.block_size
        elif direction == Direction.LEFT:
            coordx -= self.block_size
        elif direction == Direction.DOWN:
            coordy += self.block_size
        elif direction == Direction.UP:
            coordy -= self.block_size

        return Point(coordx, coordy)

    
    def get_state_1(self):

        head = self.head

        # Get the coordinates in the grid
        x = head[0] // self.block_size
        y = head[1] // self.block_size

        return (x, y, str(self.direction))
    
    def _manhhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def get_state(self):
        """
        Returns:
            d_l: if there's danger at the left.
            d_r: if there's danger at the right.
            d_f: if there's danger at the front.
            f_f: if the food is forward.
            f_r: if the food is at the right.
            f_l: if the food is at the left.
        """
        d_l, d_r, d_f, f_f, f_r, f_l = 0, 0, 0, 0, 0, 0

        # Get the coordinates in the grid
        right = self._get_point(Direction.RIGHT)
        left = self._get_point(Direction.LEFT)
        up = self._get_point(Direction.UP)
        down = self._get_point(Direction.DOWN)

        if self.direction == Direction.RIGHT:
            d_f = self.is_collision(right)
            d_r = self.is_collision(down)
            d_l = self.is_collision(up)
            f_f = self.food[0] > self.head[0]
            f_r = self.food[1] > self.head[1]
            f_l = self.food[1] < self.head[1]

        elif self.direction == Direction.LEFT:
            d_f = self.is_collision(left)
            d_r = self.is_collision(up)
            d_l = self.is_collision(down)
            f_f = self.food[0] < self.head[0]
            f_r = self.food[1] < self.head[1]
            f_l = self.food[1] > self.head[1]
        elif self.direction == Direction.UP:
            d_f = self.is_collision(up)
            d_r = self.is_collision(right)
            d_l = self.is_collision(left)
            f_f = self.food[1] < self.head[1]
            f_r = self.food[0] > self.head[0]
            f_l = self.food[0] < self.head[0]
        else:
            d_f = self.is_collision(down)
            d_r = self.is_collision(left)
            d_l = self.is_collision(right)
            f_f = self.food[1] > self.head[1]
            f_r = self.food[0] < self.head[0]
            f_l = self.food[0] > self.head[0]

        return (d_f, d_r, d_l, f_f, f_r, f_l)
    
    def get_food(self):
        food = self.food
        x = food[0] // self.block_size
        y = food[1] // self.block_size
        return (x, y)

    def pygame_draw(self):
        """Uses pygame to draw the game on the screen"""
        if self.display is None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 25)
            self.display = pygame.display.set_mode((self.width + 2 * self.margin, self.height + 2 * self.margin + 50))
            pygame.display.set_caption('Snake')

        self.display.fill((128, 128, 128))

        # Draw chessboard pattern
        for y in range(0, self.height, self.block_size):
            for x in range(0, self.width, self.block_size):
                rect = pygame.Rect(x + self.margin, y + self.margin, self.block_size, self.block_size)
                color = WHITE if (x // self.block_size + y // self.block_size) % 2 == 0 else (192, 192, 192)
                pygame.draw.rect(self.display, color, rect)

        # Draw snake
        for body_point in self.snake:
            pygame.draw.rect(
                self.display,
                (0, 206, 34),
                pygame.Rect(
                    body_point.coordx + self.margin,
                    body_point.coordy + self.margin,
                    self.block_size,
                    self.block_size))

        # Draw food
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(
                self.food.coordx + self.margin,
                self.food.coordy + self.margin,
                self.block_size,
                self.block_size))

        # Draw score
        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width // 2 + self.margin, self.height + self.margin + 25))
        self.display.blit(text, text_rect)
        pygame.display.flip()


    def _move(self, direction):
        """Adds a new head according to the given direction"""
        coordx = self.head.coordx
        coordy = self.head.coordy
        if direction == Direction.RIGHT:
            coordx += self.block_size
        elif direction == Direction.LEFT:
            coordx -= self.block_size
        elif direction == Direction.DOWN:
            coordy += self.block_size
        elif direction == Direction.UP:
            coordy -= self.block_size

        self.head = Point(coordx, coordy)

    def quit(self):
        """Quit display"""
        if self.display:
            pygame.quit()

# player_direction: ['up', 'left', 'right', 'down']


def player_to_snake_perspective(snake_direction, player_direction):
    """Transforms universal directions (player) to local directions (snake)"""
    if snake_direction == Direction.UP:
        return {
            'up': 'forward',
            'left': 'left',
            'down': 'forward',  # <- no tail crash
            'right': 'right'
        }[player_direction]

    if snake_direction == Direction.LEFT:
        return {
            'up': 'right',
            'left': 'forward',
            'down': 'left',
            'right': 'forward'  # <- no tail crash
        }[player_direction]

    if snake_direction == Direction.DOWN:
        return {
            'up': 'forward',  # <- no tail crash
            'left': 'right',
            'down': 'forward',
            'right': 'left'
        }[player_direction]

    # Direction.RIGHT
    return {
        'up': 'left',
        'left': 'forward',  # <- no tail crash
        'down': 'right',
        'right': 'forward'
    }[player_direction]
