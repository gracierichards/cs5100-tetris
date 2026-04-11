# Do pip install gym-tetris

import sys
import gym
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
import numpy as np
import random

env = gym_tetris.make("TetrisA-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Action meanings (SIMPLE_MOVEMENT):
# 0: NOOP
# 1: right
# 2: left
# 3: down
# 4: A (rotate)
# 5: B (rotate other way)


def get_board(obs):
  return obs["board"] if isinstance(obs, dict) else obs


# Have to multiply all actions by ten, as it appears that every 10 steps is the max rate to input actions
def build_action_queue(target_col, target_rot):
  actions = []

  # Rotate first
  actions += [1] * target_rot * 10

  # Move horizontally
  delta = target_col - 5  # Assuming the center column is 5
  if delta > 0:
    actions += [3] * delta * 10
  else:
    actions += [4] * (-delta) * 10

  # Then drop
  actions += [5] * 35  # the minimum number of DOWNs needed to drop a piece from the
                       # first row to the last row. Determined by trial and error.

  return actions


# Given a 20x10 binary board, return a list of the height of each column from left to right.
# The "height" is just the highest filled square for that column
def get_column_heights(board):
  # Finds the index of the first one for each column
  heights = board.argmax(axis=0)
  # Nonempty columns need to be 20 - the value. Empty columns need to be set to 0.
  has_block = board.any(axis=0)
  heights = np.where(has_block, 20-heights, 0)
  return heights


"""Training"""
def train(num_episodes=100, gamma=0.9, epsilon=1, decay_rate=0.99999, render=True):
  Q_table = {}
  state_counts = {}
  state_actions = {}
  episodes_completed = 0
  while episodes_completed < num_episodes:
    obs = env.reset()
    obs, reward, done, info = env.step(0)
    prev_frame_board = get_board(obs)
    # prev_frame_board stores the board of the most recent previous frame.
    # prev_board is the board after the second to last piece was dropped

    action_queue = []
    target_rot = None
    target_col = None

    done = False
    first_iteration = True
    just_finished_queue = False
    prev_piece = None
    prev_score = 0
    prev_max_height = 0

    while not done:
      board = get_board(obs)

      if info["current_piece"] != prev_piece or first_iteration:
        # This code is accessed the frame a new piece appears at the top. We want to save the state
        # of the board right before that piece appeared, to compare with much later (so save the previous frame)
        cur_board_before_spawn = prev_frame_board.copy()

        # Score the quality of the previous move
        if not first_iteration:
          points = info["score"] - prev_score
          max_height = max(get_column_heights(cur_board_before_spawn))
          change_in_height = max_height - prev_max_height
          if change_in_height < 0:
            reward = points
          else:
            reward = points - (change_in_height * max_height/2)
          print(f"Points: {points}  Max Height: {max_height}  Change in Height: {change_in_height}  Reward: {reward}")
          prev_score = info["score"]
          prev_max_height = max_height

          # The state will be the column heights plus the piece type, the action will be the column
          # the piece was dropped in + 2d tuple with the shape of the piece (for orientation)
          prev_state = (tuple(get_column_heights(prev_board)), prev_piece)
          new_state = (tuple(get_column_heights(cur_board_before_spawn)), info["current_piece"])
          action_hash = (first_col_of_piece, rotation_signature)
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

        # If less than epsilon, choose randomly
        if random.random() < epsilon or new_state not in state_actions:
          target_col = random.randint(0, 9)
          target_rot = random.randint(0, 3)
        else:
          # Among the actions done before for new_state, query the Q_table for their q values and
          # pick the action with the max value. All of this is done in this line.
          action_with_highest_q = max(state_actions[new_state], key=lambda a: Q_table[(new_state, a)])
          target_col, target_rot = action_with_highest_q

        # target_col = random.randint(0, 9)
        # target_rot = random.randint(0, 3)
        action_queue = build_action_queue(target_col, target_rot)
        committed = True
        prev_board = cur_board_before_spawn

      if action_queue:
        action = action_queue.pop(0)
        if not action_queue:
          just_finished_queue = True
      else:
        action = 0  # NOOP

      prev_piece = info["current_piece"]
      obs, reward, done, info = env.step(action)
      if just_finished_queue:
        just_finished_queue = False
        active = (board > 0) & (cur_board_before_spawn == 0)
        piece_rows, piece_cols = np.where(active)
        if len(piece_cols) > 0:
          first_col_of_piece = int(np.min(piece_cols))  # leftmost column of the piece
          min_r, max_r = piece_rows.min(), piece_rows.max()
          min_c, max_c = piece_cols.min(), piece_cols.max()
          shape = active[min_r:max_r+1, min_c:max_c+1]
          rotation_signature = tuple(map(tuple, shape.astype(int)))
          # Representing the final orientation of the piece as a tuple of tuples

      if committed and not action_queue:
          committed = False
      prev_frame_board = board
      env.render()
        
    episodes_completed += 1

  env.close()