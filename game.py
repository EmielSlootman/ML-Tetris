import numpy as np
import cv2
from PIL import Image
import random

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

        self.bag = list(range(len(self.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()

        self.colbag = list(range(len(self.COLOURS)))
        random.shuffle(self.colbag)
        self.colour = self.colbag.pop()

        self.reset()

    def reset(self):
        """ Reset the game: clear the board, reset the score """
        self._new_board()

        self.game_over = False
        self.score = 0
        self._new_round()

    def _new_board(self):
        """ Cleans the board by replacing it with an array of zeros """
        self.board = np.zeros((self.HEIGHT, self.WIDTH,3))

    def _get_rotated_piece(self):
        """ Returns the current piece, including rotation """ 
        return self.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_colour(self):
        return self.COLOURS[self.colour]

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
        board = self.board
        for x, y in piece:
            board[y + pos[1]][x + pos[0]][:] = colour
        return board
        

    def _get_complete_board(self):
        """ Return the board, including the yet to be placed block """
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        for x, y in piece:
            self.board[y][x][:] = self.colour
        return self.board

    def render(self):
        """ Render the board """
        img = self._get_complete_board().reshape(self.HEIGHT, self.WIDTH, 3).astype(np.uint8)
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
        board = self.board
        lines_to_clear = [index for index, row in enumerate(np.sum(self.board, axis=2)) if np.count_nonzero(row) == self.WIDTH]
        if lines_to_clear:
            board = np.array([row for index, row in enumerate(board) if index not in lines_to_clear])
            # Add new lines at the top
            board = np.concatenate((board, np.zeros((len(lines_to_clear), self.WIDTH, 3))), axis=0)
        return len(lines_to_clear), board

    def _new_round(self):
        """ Starts a new round (new piece) """
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(self.TETROMINOS)))
            random.shuffle(self.bag)

        if len(self.colbag) == 0:
            self.colbag = list(range(len(self.COLOURS)))
            random.shuffle(self.colbag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.colour = self.colbag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _rotate(self, angle):
        """ Change the current rotation.
        Input: angle """
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def get_game_score(self):
        """ Returns the current game score.
        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        """
        return self.score

    def _number_of_holes(self, board):
        """Number of holes in the board (empty sqquare with at least one block above it)"""
        holes = 0

        for col in zip(*np.sum(board,axis=2)):
            i = 0
            while i < self.HEIGHT and col[i] == 0:
                i += 1
            holes += len([x for x in col[i+1:] if x == 0])

        return holes


    def _bumpiness(self, board):
        """Sum of the differences of heights between pair of columns"""
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*np.sum(board, axis=2)):
            i = 0
            while i < self.HEIGHT and col[i] == 0:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        """Sum and maximum height of the board"""
        sum_height = 0
        max_height = 0
        min_height = self.HEIGHT

        for col in zip(*np.sum(board, axis=2)):
            i = 0
            while i < self.HEIGHT and col[i] == 0:
                i += 1
            height = self.HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_board_props(self, board):
        """Get properties of the board"""
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]


    def get_next_states(self):
        """Get all possible next states"""
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = self.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, self.WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, self._get_colour(), pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states


    def get_state_size(self):
        """Size of the state"""
        return 4

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
        print(self.board.shape)
        lines_cleared, self.board = self._clear_lines()
        score = 1 + (lines_cleared ** 2) * self.WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over
