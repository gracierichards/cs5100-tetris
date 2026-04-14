# Do pip install gym-tetris

import sys
import gym
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np
import random
import pickle

env = gym_tetris.make("TetrisA-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

possible_orientations = {'T': ['Tu', 'Tr', 'Td', 'Tl'],
                         'J': ['Jl', 'Ju', 'Jr', 'Jd'],
                         'Z': ['Zh', 'Zv'],
                         'O': ['O'],
                         'S': ['Sh', 'Sv'],
                         'L': ['Lr', 'Ld', 'Ll', 'Lu'],
                         'I': ['Iv', 'Ih']}


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
  filled_empty_grid = (board_grid > 40).astype(int)
  return filled_empty_grid


# Given a 20x10 binary board, return a list of the height of each column from left to right.
# The "height" is just the highest filled square for that column
def get_column_heights(board):
  # Finds the index of the first one for each column
  heights = board.argmax(axis=0)
  # Nonempty columns need to be 20 - the value. Empty columns need to be set to 0.
  has_block = board.any(axis=0)
  heights = np.where(has_block, 20-heights, 0)
  return heights


# Takes in the info dictionary from gym and returns the piece id without the orientation info
def get_current_piece(info):
  if not info["current_piece"]:
    return None
  return info["current_piece"][0]


"""Training"""
def train(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999, render=True, debug=False, load_pickle=True):
  if load_pickle:
    Q_table = np.load('Q_table.pickle', allow_pickle=True)
    state_counts = np.load('state_counts.pickle', allow_pickle=True)
    state_actions = np.load('state_actions.pickle', allow_pickle=True)
  else:
    Q_table = {}
    state_counts = {}
    state_actions = {}
  episodes_completed = 0
  while episodes_completed < num_episodes:
    obs = env.reset()
    obs, reward, done, info = env.step(0)
    prev_frame_board = np.zeros((20, 10))
    # prev_frame_board stores the board of the most recent previous frame.
    # prev_board is the board after the second to last piece was dropped

    target_rot = None
    target_col = None
    reached_target = True

    done = False
    first_iteration = True
    prev_piece = None
    prev_score = 0
    prev_max_height = 0
    prev_top_row = 0
    frame = 1
    while not done:
      if debug:
        if frame > 400:
          sys.exit()
        print("Frame:", frame)
      board = get_board(obs)

      if first_iteration:
        active = board > 0
      else:
        active = (board > 0) & (cur_board_before_spawn == 0)
      # Note that part of the piece can be hidden if rotated while at the top
      # It falls every 48 frames if down is not being held
      # Every 2 frames if it is
      # Two times it took 85-88 frames for a piece to fall one row after being rotated its first move
      # First action usually takes effect on the 4th frame
      # Another action was able to be input 16 frames later (confirmed)
      # And then another one 6 frames later
      # And then another one 6 frames later
      # Piece stays on the bottom for 11 frames before the new piece spawns
      piece_rows, piece_cols = np.where(active)
      if debug:
        print("Piece rows", piece_rows)
        print("Piece cols", piece_cols)
      spawn_detected = False
      if len(piece_rows) != 0:
        current_top_row = int(np.min(piece_rows))
        # If the piece is high up when on the previous frame it was low, spawn detected
        if current_top_row <= 2 and prev_top_row >= 5:
            spawn_detected = True
        prev_top_row = current_top_row

      if spawn_detected or first_iteration:
        # This code is accessed the frame a new piece appears at the top. We want to save the state
        # of the board right before that piece appeared, to compare with much later (so save the previous frame)
        cur_board_before_spawn = prev_frame_board.copy()

        # Score the quality of the previous move
        if not first_iteration:
          points = info["score"] - prev_score
          max_height = max(get_column_heights(cur_board_before_spawn))
          change_in_height = max_height - prev_max_height
          if points < 40:
            points = 0  # You get bonus points for holding DOWN, which should not be factored in
          if change_in_height < 0:
            reward = points
          else:
            reward = points - (change_in_height * max_height/2)
          if points != 0:
            print(f"Episode: {episodes_completed}  Frame: {frame}  Points: {points}  Max Height: {max_height}  Change in Height: {change_in_height}  Reward: {reward}")
          prev_score = info["score"]
          prev_max_height = max_height

          # The state will be the column heights plus the piece type, the action will be the column
          # the piece was dropped in + 2d tuple with the shape of the piece (for orientation)
          prev_state = (tuple(get_column_heights(prev_board)), prev_piece)
          new_state = (tuple(get_column_heights(cur_board_before_spawn)), get_current_piece(info))
          action_hash = (target_col, target_rot)
          if (prev_state, action_hash) not in Q_table:
            prevQ = 0
          else:
            prevQ = Q_table[(prev_state, action_hash)]
          if (prev_state, action_hash) not in state_counts:
            state_counts[(prev_state, action_hash)] = 0
          alpha = 1/(1 + state_counts[(prev_state, action_hash)])
          if prev_state not in state_actions:
            state_actions[prev_state] = set()
          state_actions[prev_state].add(action_hash)
          # The max over all a' of Q(s', a') term
          if new_state in state_actions:
              best_future_q = max(Q_table[(new_state, a)] for a in state_actions[new_state])
          else:
              best_future_q = 0.0

          q = prevQ + alpha * (reward + (gamma * best_future_q) - prevQ)
          Q_table[(prev_state, action_hash)] = q
          state_counts[(prev_state, action_hash)] += 1

        first_iteration = False
        reached_target = False

        # If less than epsilon, choose randomly
        if random.random() < epsilon or new_state not in state_actions:
          target_col = random.randint(0, 9)
          piece_type = get_current_piece(info)
          target_rot = random.choice(possible_orientations[piece_type])
        else:
          # Among the actions done before for new_state, query the Q_table for their q values and
          # pick the action with the max value. All of this is done in this line.
          action_with_highest_q = max(state_actions[new_state], key=lambda a: Q_table[(new_state, a)])
          target_col, target_rot = action_with_highest_q

        prev_board = cur_board_before_spawn
        obs, reward, done, info = env.step(0)
      elif len(piece_cols) == 0:
        obs, reward, done, info = env.step(0)
      else:
        # During the frame where spawn_detected is true, both the old and new piece are counted as the new piece,
        # so if we run this code on that frame it will break
        first_col_of_piece = int(np.min(piece_cols))  # leftmost column of the piece
        piece_width = int(np.max(piece_cols)) - first_col_of_piece + 1
        if debug:
          print("Reached target:", reached_target)
          print("Target column:", target_col)
          print("Target orientation:", target_rot)
          print("Current orientation:", info["current_piece"])
        # Each frame, do a rotation if the orientation isn't correct, if it is, do a translation if the
        # column position isn't correct
        if not reached_target:
          if info["current_piece"] != target_rot:
            # Need to avoid spamming the rotate button (mysteries of the Tetris DAS system -
            # using mod 6 doesn't work)
            if frame % 5 == 0:
              obs, reward, done, info = env.step(1)  # rotate clockwise
            else:
              obs, reward, done, info = env.step(0)
          else:
            if debug:
              print("Column location:", first_col_of_piece)
              print("Piece width:", piece_width)
            if first_col_of_piece > target_col:
              obs, reward, done, info = env.step(4)  # try left
            elif first_col_of_piece < target_col:
              if first_col_of_piece >= 10 - piece_width:  # can't go any further to the right
                reached_target = True
              else:
                obs, reward, done, info = env.step(3)  # try right
            else:
              reached_target = True
        else:
          obs, reward, done, info = env.step(5)  # press down until new piece appears

      prev_piece = get_current_piece(info)
      if not prev_piece:
        obs, reward, done, info = env.step(0)
        prev_piece = get_current_piece(info)
      prev_frame_board = board
      if render:
        env.render()

      frame += 1
        
    episodes_completed += 1
    epsilon = epsilon * decay_rate

  # Save to file
  with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('state_counts.pickle', 'wb') as handle:
    pickle.dump(state_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('state_actions.pickle', 'wb') as handle:
    pickle.dump(state_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)

train(render=False)
env.close()