# Do pip install gym-tetris

import sys
import random
import time
import pickle
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
# Represented as a translation value, -4 to 4, and a rotation value, from 0 to 3

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
  heights = bcopy.argmax(axis=0)
  # Nonempty columns need to be 20 - the value. Empty columns need to be set to 0.
  has_block = bcopy.any(axis=0)
  heights = np.where(has_block, 20-heights, 0)
  return heights


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


"""Returns the amount of translation actions to do, ranging from left 4 to right 4,
and a random number of rotations that makes sense for the current piece"""
def choose_random_action(cur_piece):
  translation = random.randint(-4, 4)
  id = cur_piece[0]
  num_rotations = random.randint(0, num_orientations[id] - 1)
  return (translation, num_rotations)


# Source: provided code for Assignment 2
def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)


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


"""Training"""
def train(num_episodes=100, gamma=0.9, epsilon=1, decay_rate=0.99999, render=True):
  # Actions are mapped by doing (translation number + 4) + (9 * rotation number) to get the index of the column
  Q_table = {}
  state_counts = {}
  episodes_completed = 0

  while episodes_completed < num_episodes:
    state = env.reset()
    state, reward, done, info = env.step(0)
    board = get_board(state)
    # total_lines_cleared = 0
    prev_score = 0  # Score of the previous step
    # old_piece = False  # True for a little bit after an action has been decided for a piece, to prevent picking an action again on the same piece
    first_iteration = True
    action_queue = []
    prev_piece = None
    # step_num = 0
    while not done:
      if first_iteration or info["current_piece"] != prev_piece:
        # Score the quality of the previous move
        # Extract features:
        heights = get_column_heights(board)
        heights_str = heights.astype(str)
        # max_height = max(heights)
        # aggregate_height = sum(heights)
        # Height differential
        # differential = max(heights) - min(heights)
        # Number of lines completed with the current piece
        # lines_cleared = info["number_of_lines"] - total_lines_cleared
        # print("Lines cleared by the previous move:", lines_cleared)
        # total_lines_cleared = info["number_of_lines"]
        # Change in score
        points = info["score"] - prev_score
        # print("Points obtained by the previous move:", points)
        # Number of holes in the board
        # num_holes = find_num_holes(board)
        # print("Number of holes:", num_holes)

        # For Q learning, the best state here would be the column heights array plus information on
        # the shape of the current piece
        # We can just give each piece type an ID and have it learn the best moves for each blindly
        # State = a string of the column heights with the piece letter code appended to it
        new_state = ",".join(heights_str) + info["current_piece"]
        # print("state hash is", new_state)

        if not first_iteration:
          if prev_state not in Q_table:
            Q_table[prev_state] = [0] * 36  # 36 is the number of translation + rotation combinations
          if prev_state not in state_counts:
            state_counts[prev_state] = [0] * 36
          if new_state not in Q_table:
            Q_table[new_state] = [0] * 36

          action_index = (mov_action + 4) + (9 * rot_action)
          prevQ = Q_table[prev_state][action_index]
          alpha = 1/(1 + state_counts[prev_state][action_index])
          # The max over all a' of Q(s', a') term
          best_future_q = max(Q_table[new_state])

          q = prevQ + alpha * (points + (gamma * best_future_q) - prevQ)
          Q_table[prev_state][action_index] = q
          state_counts[prev_state][action_index] += 1

        first_iteration = False

        prev_score = info["score"]
        # old_piece = True
        prev_piece = info["current_piece"]

        # If less than epsilon, choose randomly
        if random.random() < epsilon or new_state not in Q_table:
          mov_action, rot_action = choose_random_action(info["current_piece"])
        else:
          # Choose the highest Q value (if present)
          # Choose randomly among those that are tied
          cur_state_qs = Q_table[new_state]
          maxQ = max(cur_state_qs)
          matching_indices = [i for i,x in enumerate(cur_state_qs) if x == maxQ]
          i = random.choice(matching_indices)
          # Convert index to the right translation value and rotation value
          rot_action = i // 9
          mov_action = i % 9 - 4

        # print(f"Translation value is {mov_action}, rotation value is {rot_action}")
        for _ in range(20):
          action_queue.append(5)  # DOWN
        if mov_action > 0:
          mov_action_type = 3
          for _ in range(mov_action):
            action_queue.append(mov_action_type)
        elif mov_action < 0:
          mov_action_type = 4
          for _ in range(mov_action * -1):
            action_queue.append(mov_action_type)
        # mov_action == 0 is NOOP
        for _ in range(rot_action):
          action_queue.append(1)

        prev_state = new_state
        # Have to wait until the next new piece arrives in order to get the reward and update the Q table
      # else:
      #   # Waiting for a new piece to arrive, no learning here
      #   if not piece_spawned(board):
      #     old_piece = False

      if action_queue:
        executed_action = action_queue.pop()
        # print("Top row occupied?", board[0].any())
        # print(f"Executing action {executed_action}")
        state, reward, done, info = env.step(executed_action)
      else:
        state, reward, done, info = env.step(0)
        # if step_num % 2 == 0:
        #   state, reward, done, info = env.step(0) # NOOP
        # else:
        #   state, reward, done, info = env.step(5) # Down
        
      if render:
        env.render()
      board = get_board(state)
      # step_num += 1
    
    episodes_completed += 1
    epsilon = epsilon * decay_rate

  # Save to file
  with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""Evaluation"""
def eval(num_episodes=100):
  Q_table = np.load('Q_table.pickle', allow_pickle=True)
  scores = []
  line_totals = []
  piece_totals = []
  # Source: provided code in Assignment 2
  for _ in tqdm(range(num_episodes)):
    state = env.reset()
    state, reward, done, info = env.step(0)
    board = get_board(state)
    prev_piece = None
    action_queue = []
    while not done:
      if info["current_piece"] != prev_piece:
        heights = get_column_heights(board)
        heights_str = heights.astype(str)
        hashed_state = ",".join(heights_str) + info["current_piece"]
        try:
          action = np.random.choice(36, p=softmax(Q_table[hashed_state]))  # Select action using softmax over Q-values
          # Convert index to the right translation value and rotation value
          rot_action = action // 9
          mov_action = action % 9 - 4
        except KeyError:
          # Fallback to random action if state not in Q-table
          mov_action, rot_action = choose_random_action(info["current_piece"])

        for _ in range(20):
          action_queue.append(5)  # DOWN
        if mov_action > 0:
          mov_action_type = 3
          for _ in range(mov_action):
            action_queue.append(mov_action_type)
        elif mov_action < 0:
          mov_action_type = 4
          for _ in range(mov_action * -1):
            action_queue.append(mov_action_type)
        # mov_action == 0 is NOOP
        for _ in range(rot_action):
          action_queue.append(1)

      if action_queue:
        executed_action = action_queue.pop()
        # print("Top row occupied?", board[0].any())
        # print(f"Executing action {executed_action}")
        state, reward, done, info = env.step(executed_action)
      else:
        state, reward, done, info = env.step(0)
        # if i % 3 == 0:
        #   state, reward, done, info = env.step(0) # NOOP
        # else:
        #   state, reward, done, info = env.step(5) # Down

      board = get_board(state)

    total_score = info["score"]
    lines_cleared = info["number_of_lines"]
    tp = total_pieces(info["statistics"])
    print(f"Score: {total_score}\tLines Cleared: {lines_cleared}\tTotal Pieces Dropped: {tp}")

  # avg_reward = sum(rewards)/len(rewards)
  # print("Average reward:", avg_reward)
  # avg_ep_len = sum(ep_lengths)/len(ep_lengths)
  # print("Average episode length:", avg_ep_len)
  # print("Number of states not found in the Q table:", len(states_not_in_Q))
  # total_actions_taken = sum(ep_lengths)
  # percent_not_in_Q = 100 * len(states_not_in_Q)/total_actions_taken
  # print(f"Percent of actions that were chosen randomly due to failure to find the state in the Q table: {percent_not_in_Q:.2f}%")

start_time = time.time()
train(render=False)
print(f"Training took {time.time() - start_time} seconds")
eval()

env.close()