# Do pip install gym-tetris

import sys
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import matplotlib.pyplot as plt
import numpy as np

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)

# instructions = [6, 3, 1, 6, 3, 2]

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

# Read the observation space (the board)
state = env.reset()
prev_board = get_board(state)
new_board = get_board(state)
action_allowed = True
for step in range(10**4):
  # Time it so you can only perform an action at the rhythm the game moves
  if not np.array_equal(prev_board, new_board):
    if action_allowed:
      state, reward, done, info = env.step(env.action_space.sample())
      action_allowed = False
      print(new_board)
      input("Paused here")  # To see the screen at this exact point without the program closing
    else:
      state, reward, done, info = env.step(0)
  else:
    action_allowed = True
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