#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Very simple tetris implementation
# 
# Control keys:
# Down - Drop stone faster
# Left/Right - Move stone
# Up - Rotate Stone clockwise
# Escape - Quit game
# P - Pause game
#
# Have fun!

# Copyright (c) 2010 "Kevin Chabowski"<kevin@kch42.de>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from random import randrange as rand
import numpy as np
import pygame, sys

# The configuration

class TetrisApp(object):

	# Define the shapes of the single parts
	TETROMINOS = [
		[[1, 1, 1],
		 [0, 1, 0]],
		
		[[0, 2, 2],
		 [2, 2, 0]],
		
		[[3, 3, 0],
		 [0, 3, 3]],
		
		[[4, 0, 0],
		 [4, 4, 4]],
		
		[[0, 0, 5],
		 [5, 5, 5]],
		
		[[6, 6, 6, 6]],
		
		[[7, 7],
		 [7, 7]]
	]
	
	# Defines the possible colors
	COLOURS = [
		(0,   0,   0  ),
		(255, 0,   0  ),
		(0,   150, 0  ),
		(0,   0,   255),
		(255, 120, 0  ),
		(255, 255, 0  ),
		(180, 0,   255),
		(0,   220, 220)
	]
	
	def __init__(self, cols, rows, speed, render, size, fps):
		""" Initialize the game and the screen 
			Input: [int colums] [int rows] [bool render (wether the game needs to be rendered)] [int size (size of screen)]"""
		self.cols = cols
		self.rows = rows
		self.render = render
		self.size = size
		self.fps = fps
		self.delay = speed
		if self.render:
			pygame.init()
			pygame.key.set_repeat(250,25)
			self.width = self.size * self.cols
			self.height = self.size * self.rows
		
			self.screen = pygame.display.set_mode((self.width, self.height))
		
			# We do not need the mouse movement events, so we block them
			pygame.event.set_blocked(pygame.MOUSEMOTION)
		self.reset()
		
	def _rotate_piece(self, shape):
		return [ [ shape[y][x] for y in range(len(shape)) ] for x in range(len(shape[0]) - 1, -1, -1) ]

	def check_collision(self, board, shape, offset):
		off_x, off_y = offset
		for cy, row in enumerate(shape):
			for cx, cell in enumerate(row):
				try:
					if cell and board[ cy + off_y, cx + off_x ]:
						return True
				except IndexError:
					return True
		return False

	def _clear_lines(self, board):
		""" Checks if lines can be cleared and clears them
			Output: Number of lines clear, new board"""
		lines_to_clear = [index for index, row in enumerate(board[:-1]) if np.count_nonzero(row)  == self.cols]
		if lines_to_clear:
			board = np.array([row for index, row in enumerate(board) if index not in lines_to_clear])
			# Add new lines at the top
			board = np.concatenate((np.zeros((len(lines_to_clear), self.cols), dtype=np.int8), board), axis=0)
		return len(lines_to_clear), board
		
	def join_matrixes(self, mat1, mat2, mat2_off):
		off_x, off_y = mat2_off
		for cy, row in enumerate(mat2):
			for cx, val in enumerate(row):
				mat1[cy+off_y-1	][cx+off_x] += val
		return mat1

	def new_board(self):
		board = np.zeros((self.rows + 1, self.cols), dtype=np.int8)
		board[-1, :] = 1
		return board
	
	def new_stone(self):
		self.stone = self.TETROMINOS[rand(len(self.TETROMINOS))]
		self.stone_x = int(self.cols / 2 - len(self.stone[0])/2)
		self.stone_y = 0
		
		if self.check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
			self.gameover = True
	
	def reset(self):
		self.board = self.new_board()
		self.new_stone()
		self.score = 0
	
	def _center_msg(self, msg):
		for i, line in enumerate(msg.splitlines()):
			msg_image =  pygame.font.SysFont("Arial", 20).render(line, False, (255,255,255), (0,0,0))
		
			msgim_center_x, msgim_center_y = msg_image.get_size()
			msgim_center_x //= 2
			msgim_center_y //= 2
		
			self.screen.blit(msg_image, (self.width // 2-msgim_center_x, self.height // 2-msgim_center_y+i*22))

	def _score_msg(self):
		msg_image =  pygame.font.SysFont("Arial", 20).render("Score: " + str(self.score), False, (255,255,255), (0,0,0))
		self.screen.blit(msg_image, (0,0))
		
	
	def _render_matrix(self, matrix, offset):
		off_x, off_y  = offset
		for y, row in enumerate(matrix):
			for x, val in enumerate(row):
				if val:
					pygame.draw.rect(self.screen, self.COLOURS[val], pygame.Rect((off_x+x) * self.size, (off_y+y) * self.size, self.size, self.size),0)
	
	def move(self, delta_x):
		if not self.gameover and not self.paused:
			new_x = self.stone_x + delta_x
			if new_x < 0:
				new_x = 0
			if new_x > self.cols - len(self.stone[0]):
				new_x = self.cols - len(self.stone[0])
			if not self.check_collision(self.board, self.stone, (new_x, self.stone_y)):
				self.stone_x = new_x
				
	def quit(self):
		self._center_msg("Exiting...")
		if self.render:
			pygame.display.update()
		sys.exit()
	
	def drop(self):
		if not self.gameover and not self.paused:
			self.stone_y += 1
			if self.check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
				self.board = self.join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
				self.new_stone()
				lines, self.board = self._clear_lines(self.board)
				self.score += 1 + lines**2 * self.cols
	
	def rotate_stone(self):
		if not self.gameover and not self.paused:
			new_stone = self._rotate_piece(self.stone)
			if not self.check_collision(self.board, new_stone, (self.stone_x, self.stone_y)):
				self.stone = new_stone
	
	def toggle_pause(self):
		self.paused = not self.paused
	
	def start_game(self):
		if self.gameover:
			self.reset()
			self.gameover = False

	
	def get_game_score(self):
		""" Returns the current game score.
		Each block placed counts as one.
		For lines cleared, it is used self.cols * lines_cleared ^ 2.
		"""
		return self.score

	def _number_of_holes(self, board):
		"""Number of holes in the board (empty sqquare with at least one block above it)"""
		holes = 0

		for col in zip(*board):
			i = 0
			while i < self.rows and col[i] == 0:
				i += 1
			holes += len([x for x in col[i+1:] if x == 0])

		return holes

	def _bumpiness(self, board):
		"""Sum of the differences of heights between pair of columns"""
		total_bumpiness = 0
		max_bumpiness = 0
		min_ys = []

		for col in zip(*board):
			i = 0
			while i < self.rows and col[i] == 0:
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
		min_height = self.rows

		for col in zip(*board):
			i = 0
			while i < self.rows and col[i] == 0:
				i += 1
			height = self.rows - i
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

	def get_state_size(self):
		"""Size of the state"""
		return 4

	def get_next_states(self):
		"""Get all possible next states"""
		states = {}
		piece = self.stone.copy()
		
		if len(piece[0]) == 2: 
			rotations = 1
		elif len(piece[0]) == 4:
			rotations = 2
		else:
			rotations = 4

		# For all rotations
		for rot in range(rotations):
			x_len = len(piece[0])

			# For all positions
			for x in range(self.cols - x_len + 1):
				pos = [x, 0]
				tempboard = self.board.copy()
				# Drop piece
				while not self.check_collision(tempboard, piece, pos):
					pos[1] += 1

				tempboard = self.join_matrixes(tempboard, piece, pos)
				states[(x, rot)] = self._get_board_props(tempboard)
			piece = self._rotate_piece(piece)
		return states

	def pcplace(self, x, rot):
		pos = [x, 0]
		for _ in range(rot):
			self.stone = self._rotate_piece(self.stone)
		if self.render:
			self.screen.fill((0,0,0))
			while not self.check_collision(self.board, self.stone, pos):
				self._render_matrix(self.board, (0,0))
				self._render_matrix(self.stone,(self.stone_x, self.stone_y))
				self._score_msg() 
				pos[1] += 1
				pygame.display.update()
				self.dont_burn_my_cpu.tick(self.fps)	
			self.board = self.join_matrixes(self.board, self.stone, pos)
		else:
			while not self.check_collision(self.board, self.stone, pos):
				pos[1] +=1
			self.board = self.join_matrixes(self.board, self.stone, pos)
		lines, self.board = self._clear_lines(self.board)
		self.score += 1 + lines**2 * self.cols
		self.new_stone()
		if self.gameover:
			self.score -= 2
		return self.score, self.gameover

	def pcrun(self):
		self.gameover = False
		self.paused = False
		if self.render:
			self.dont_burn_my_cpu = pygame.time.Clock()
	
	def run(self):
		key_actions = {
			'ESCAPE':	self.quit,
			'LEFT':		lambda:self.move(-1),
			'RIGHT':	lambda:self.move(+1),
			'DOWN':		self.drop,
			'UP':		self.rotate_stone,
			'p':		self.toggle_pause,
			'SPACE':	self.start_game
		}
		
		self.gameover = False
		self.paused = False
		pygame.time.set_timer(pygame.USEREVENT+1, self.delay)
		dont_burn_my_cpu = pygame.time.Clock()
		while 1:
			self.screen.fill((0,0,0))
			if self.gameover:
				self._center_msg("Game Over! \n You score was: " + str(self.score) + "\n Press space to continue")
			else:
				self._render_matrix(self.board, (0,0))
				self._render_matrix(self.stone,(self.stone_x, self.stone_y))
				self._score_msg()
				if self.paused:
					self._center_msg("Paused")
			pygame.display.update()

			for event in pygame.event.get():
				if event.type == pygame.USEREVENT+1:
					self.drop()
				elif event.type == pygame.QUIT:
					self.quit()
				elif event.type == pygame.KEYDOWN:
					for key in key_actions:
						if event.key == eval("pygame.K_"+key):
							key_actions[key]()
			dont_burn_my_cpu.tick(self.fps)

if __name__ == '__main__':
	App = TetrisApp(8, 16, 750, True, 40, 30)
	App.run()