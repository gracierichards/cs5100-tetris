# Do pip install gym-tetris

import sys
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import numpy as np

# Possible actions for a turn (only one turn per piece)
# One of:
# LEFT LEFT LEFT LEFT
# LEFT LEFT LEFT
# LEFT LEFT
# LEFT
# NOOP
# RIGHT
# RIGHT RIGHT
# RIGHT RIGHT RIGHT
# RIGHT RIGHT RIGHT RIGHT
# Plus one of these for rotations:
# NOOP
# A
# A A
# A A A
TRANSLATION_ACTIONS = ["4L", "3L", "2L", "1L", "0", "1R", "2R", "3R", "4R"]
# Rotation actions - number from 0 to 3?

# Number of orientations each piece can have
num_orientations = {"T": 4, "J": 4, "Z":2, "O":1, "S":2, "L":4, "I":2}

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Get the total number of pieces dropped. Takes in info[statistics]
def total_pieces(statistics):
  total = 0
  for type0 in statistics:
    total += statistics[type0]
  return total

# Helper for show_board()
def format_coord(x, y):
  return f"x={int(x)}, y={int(y)}"

# Shows the board and pixel coordinates
def show_board():
  state = env.reset()
  fig, ax = plt.subplots()
  plt.imshow(state)
  plt.title("Tetris Frame")
  ax.format_coord = format_coord
  plt.show()
  sys.exit()

# For getting the pixel coordinate boundaries of the board
# show_board()

# Turn the raw pixel data into a usable board state
def get_board(state):
  # Crops the screen to just the board
  board_img = state[46:208, 94:175]
  # Replace each RGB pixel with the average of the 3 color values
  gray = board_img.mean(axis=2)
  # Also number of pixels tall, number of pixels wide if that's easier
  num_pixels_y, num_pixels_x = gray.shape
  # A Tetris board is 20 by 10
  cellh = num_pixels_y // 20
  cellw = num_pixels_x // 10
  board_grid = np.zeros((20, 10))
  for r in range(20):
    for c in range(10):
      y1 = r * cellh
      y2 = (r + 1) * cellh
      x1 = c * cellw
      x2 = (c + 1) * cellw
      cell = gray[y1:y2, x1:x2]
      board_grid[r, c] = cell.mean()
  # Used to find the grayscale value threshold for a filled square. I found it to be around 40.
  # print("Number of grid cells < 10:", np.count_nonzero(board_grid <= 10))
  # print("Number of grid cells < 20:", np.count_nonzero((board_grid > 10) & (board_grid <= 20)))
  # print("Number of grid cells < 30:", np.count_nonzero((board_grid > 20) & (board_grid <= 30)))
  # print("Number of grid cells < 40:", np.count_nonzero((board_grid > 30) & (board_grid <= 40)))
  # print("Number of grid cells < 50:", np.count_nonzero((board_grid > 40) & (board_grid <= 50)))
  # print("Number of grid cells < 60:", np.count_nonzero((board_grid > 50) & (board_grid <= 60)))
  # print("Number of grid cells < 70:", np.count_nonzero((board_grid > 60) & (board_grid <= 70)))
  # print("Number of grid cells < 80:", np.count_nonzero((board_grid > 70) & (board_grid <= 80)))
  # print("Number of grid cells < 90:", np.count_nonzero((board_grid > 80) & (board_grid <= 90)))
  # print("Number of grid cells < 100:", np.count_nonzero((board_grid > 90) & (board_grid <= 100)))
  filled_empty_grid = (board_grid > 40).astype(int)
  return filled_empty_grid

# True for the first few frames after a new piece spawns. Actually just checks if two blocks in the middle
# of the first row are filled, they should always be filled when a piece spawns
def piece_spawned(board):
  return board[0][4] or board[0][5]

# Given a 20x10 binary board, return a list of the height of each column from left to right.
# The "height" is just the highest filled square for that column
# Only call this when a new piece has appeared, since the current piece's filled cells are not being tracked
# and would interfere
def get_column_heights(board):
  # Fields to ignore
  # O: [0][4], [0][5], [1][4], [1][5]
  # T: [0][4], [0][5], [0][6], [1][5]
  # I: [0][3], [0][4], [0][5], [0][6]
  # L: [0][4], [0][5], [0][6], [1][4]
  # S: [0][5], [0][6], [1][4], [1][5]
  # Z: [0][4], [0][5], [1][5], [1][6]
  # J: [0][4], [0][5], [0][6], [1][6]
  # Cells to ignore: [0][3], [0][4], [0][5], [0][6], [1][4], [1][5], [1][6]

  bcopy = board.copy()
  # Set the ignored cells to 0
  bcopy[0][3] = 0
  for row in [0, 1]:
    for col in [4, 5, 6]:
      bcopy[row][col] = 0
  # Finds the index of the first one for each column
  heights = board.argmax(axis=0)
  # Convert to 20-x for each element in heights
  return 20 - heights


"""Return the column of the first hole it finds, a hole being an unfilled cell
without any filled cells above it. Starts looking from the bottom row, so it will
find the deepest or at least tied for the deepest hole.
Input: the 20x10 grid, showing which cells are filled or not in binary"""
def find_hole(board):
  for row in range(19, 1, -1):
    # Ignore the top two rows, because this is calculated when the next piece spawns,
    # and each piece can take up up to two rows
    for col in range(10):
      if board[row][col] == 0:
        is_hole = True
        r = row
        while r >= 0:
          if r != 0:
            is_hole = False
            break
          r -= 1
        if is_hole:
          return col
  # Guaranteed to find a hole in the board, since rows that are completely filled are cleared


"""Find the number of holes across the entire board. Above is the inaccurate definition of a hole,
but this function uses the correct definition, which is that a hole is an unfilled cell with at least
one filled cell above it"""
def find_num_holes(board):
  # Loop for each column: go down each square one by one from the top. Once you hit a filled block,
  # start counting how many unfilled blocks are below it. Then just add all of them up
  num_holes = 0
  for column in board.T:  # The transpose
    start_counting = False
    for val in column:
      if start_counting and val == 0:
        num_holes += 1
      if not start_counting and val == 1:
        start_counting = True
  return num_holes


"""Part 1: Perform ten trials of choosing a random action each step.
  Find the average number of pieces dropped per trial."""
# state = env.reset()
# trials = 0
# total_totals = 0
# for step in range(10**4):
#   state, reward, done, info = env.step(env.action_space.sample())
#   env.render()
#   if done:
#     tp = total_pieces(info["statistics"])
#     print("Total pieces dropped:", tp)
#     total_totals += tp
#     state = env.reset()
#     trials += 1
#     if trials == 10:
#       break
# print("Average:", total_totals/10)


"""Testing reading the observation space (the board) and timing actions"""
# state = env.reset()
# prev_board = get_board(state)
# new_board = get_board(state)
# print(new_board)
# env.render()
# input("Paused here")  # To see the screen at this exact point without the program closing
# action_allowed = True
# for step in range(10**4):
#   # Time it so you can only perform an action at the rhythm the game moves
#   # if not np.array_equal(prev_board, new_board):
#   #   if action_allowed:
#   #     state, reward, done, info = env.step(env.action_space.sample())
#   #     action_allowed = False
#   #     print(new_board)
#   #     env.render()
#   #     input("Paused here")  # To see the screen at this exact point without the program closing
#   #   else:
#   #     state, reward, done, info = env.step(0)
#   if 1 in new_board[0]:
#     if action_allowed:
#       state, reward, done, info = env.step(env.action_space.sample())
#       action_allowed = False
#       print(new_board)
#       env.render()
#       input("Paused here")  # To see the screen at this exact point without the program closing
#     else:
#       state, reward, done, info = env.step(0)
#   else:
#     action_allowed = True
#     state, reward, done, info = env.step(0)
    
#   # env.render()
#   prev_board = new_board
#   new_board = get_board(state)
#   if done:
#     tp = total_pieces(info["statistics"])
#     print("Total pieces dropped:", tp)
#     state = env.reset()
#     break

# env.close()


"""Testing timing it to plan its set of actions one time per new piece"""
state = env.reset()
prev_board = get_board(state)
new_board = get_board(state)
total_lines_cleared = 0
for step in range(10**4):
  if piece_spawned(new_board) and not old_piece:
    # Score the quality of the previous move
    # Extract features:
    heights = get_column_heights(new_board)
    print("Column heights:", heights)
    max_height = max(heights)
    aggregate_height = sum(heights)
    # Number of lines completed with the current piece
    lines_cleared = info["number_of_lines"] - total_lines_cleared
    print("Lines cleared by the previous move:", lines_cleared)
    total_lines_cleared = info["number_of_lines"]
    # Height differential
    differential = max(heights) - min(heights)
    # Make correction - to holes in the code and the doc
    # Number of holes in the board
    num_holes = find_num_holes(new_board)
    print("Number of holes:", num_holes)

    state, reward, done, info = env.step(env.action_space.sample())
    old_piece = True
    # print("Current piece is:", info["current_piece"])
  else:
    if not piece_spawned(new_board):
      old_piece = False
    state, reward, done, info = env.step(0)
    
  env.render()
  prev_board = new_board
  new_board = get_board(state)
  if done:
    tp = total_pieces(info["statistics"])
    print("Total pieces dropped:", tp)
    state = env.reset()
    break

env.close()