import gym
from gym import spaces
import numpy as np

# 10x20 board
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
SHAPES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1]],
}

def get_random_shape():
    shape = SHAPES[np.random.choice(list(SHAPES))]
    return np.array(shape)

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.current_piece = get_random_shape()
        self.piece_x = 3
        self.piece_y = 0
        self.action_space = spaces.Discrete(4)  # 0=left, 1=right, 2=rotate, 3=drop
        self.observation_space = spaces.Box(low=0, high=1, shape=(BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)

    def reset(self):
        self.board[:] = 0
        self.current_piece = get_random_shape()
        self.piece_x, self.piece_y = 3, 0
        return self._get_obs()

    def _get_obs(self):
        obs = self.board.copy()
        shape = self.current_piece
        for y in range(shape.shape[0]):
            for x in range(shape.shape[1]):
                if shape[y, x]:
                    if 0 <= y + self.piece_y < BOARD_HEIGHT and 0 <= x + self.piece_x < BOARD_WIDTH:
                        obs[y + self.piece_y, x + self.piece_x] = 1
        return obs

    def step(self, action):
        if action == 0: self.piece_x -= 1
        elif action == 1: self.piece_x += 1
        elif action == 2: self.current_piece = np.rot90(self.current_piece)
        elif action == 3:
            while not self._collision(): self.piece_y += 1
            self.piece_y -= 1
            self._place_piece()
            self.current_piece = get_random_shape()
            self.piece_x, self.piece_y = 3, 0
            lines_cleared = self._clear_lines()
            done = self._collision()
            return self._get_obs(), lines_cleared, done, {}

        if self._collision():
            if action == 0: self.piece_x += 1
            elif action == 1: self.piece_x -= 1
            elif action == 2: self.current_piece = np.rot90(self.current_piece, -1)

        return self._get_obs(), 0, False, {}

    def _collision(self):
        shape = self.current_piece
        for y in range(shape.shape[0]):
            for x in range(shape.shape[1]):
                if shape[y, x]:
                    ny = y + self.piece_y
                    nx = x + self.piece_x
                    if ny >= BOARD_HEIGHT or nx < 0 or nx >= BOARD_WIDTH or self.board[ny, nx]:
                        return True
        return False

    def _place_piece(self):
        shape = self.current_piece
        for y in range(shape.shape[0]):
            for x in range(shape.shape[1]):
                if shape[y, x]:
                    self.board[self.piece_y + y, self.piece_x + x] = 1

    def _clear_lines(self):
        lines = 0
        new_board = []
        for row in self.board:
            if all(row):
                lines += 1
            else:
                new_board.append(row)
        while len(new_board) < BOARD_HEIGHT:
            new_board.insert(0, np.zeros(BOARD_WIDTH))
        self.board = np.array(new_board)
        return lines

    def render(self, mode="human"):
        print("\n".join("".join("■" if x else "." for x in row) for row in self._get_obs()))
        print()
