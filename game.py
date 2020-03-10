import numpy as np
import cv2
from PIL import Image

class Tetris:
    """ Class of the game self """

    # The array that keeps track of all the possible pieces
    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # O
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        }
    }

    # The array that keeps track of all the possible colors of the pieces
    COLOURS = {
        0: (0,   0,   0),
        1: (255, 0,   0),
        2: (0,   150, 0),
        3: (0,   0,   255),
        4: (255, 120, 0),
        5: (255, 255, 0),
        6: (180, 0,   255),
        7: (0,   220, 220)
    }

    def __init__(self, width, height):
        """ Takes as input the width and height of the playing map """
        self.WIDTH = width
        self.HEIGHT = height

        self.reset()

    def reset(self):
        """ Reset the game: clear the board, reset the score """
        self._new_board()
        self.score = 0

    def _new_board(self):
        """ Cleans the board by replacing it with an array of zeros """
        self.board = np.zeros((self.HEIGHT, self.WIDTH,3))

    def _get_rotated_piece(self):
        """ Returns the current piece, including rotation """ 
        return self.TETROMINOS[self.current_piece][self.current_rotation]

    def _check_collision(self, piece, pos):
        """Check if there is a collision between the current piece and the board
        Inputs are a list of the piece coordinatesand the position """
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= self.WIDTH \
                    or y < 0 or y >= self.HEIGHT \
                    or np.sum(self.board[y][x][:]) != 0:
                return True
        return False

    def _add_piece_to_board(self, piece, colour, pos):
        """ Place a new piece on the board.
        Inputs are a list of the piece coordinates, the colour of the piece and the position """
        for x, y in piece:
            self.board[y + pos[1]][x + pos[0]][:] = colour

    def _get_full_board(self):
        """ Return the board, including the yet to be placed block """
        # TODO: Also include the falling block
        return self.board

    def render(self):
        """ Render the board """
        img = self._get_full_board().reshape(self.HEIGHT, self.WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((self.WIDTH * 25, self.HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)

    def _clear_lines(self):
        """ Clears completed lines in a board """
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in np.sum(self.board, axis=2) if np.count_nonzero(row) == self.WIDTH]
        if lines_to_clear:
            board = np.array([row for index, row in enumerate(board) if index not in lines_to_clear])
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(self.WIDTH)])
        return len(lines_to_clear), board

    def play(self, x, rotation, render=False, render_delay=None):
        """ Makes a play given a position and a rotation, returning the reward and if the game is over """
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score        
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self._get_colour(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over


tetris = Tetris(10, 20)
tetris._place_piece(tetris.TETROMINOS[2][90], tetris.COLOURS[2], (2,2))
tetris._place_piece(tetris.TETROMINOS[1][90], tetris.COLOURS[4], (7,15))
tetris.render()
# print(np.sum(tetris.board, axis=2))