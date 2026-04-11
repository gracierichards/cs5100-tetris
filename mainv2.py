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


"""Training"""
def train(num_episodes=100, gamma=0.9, epsilon=1, decay_rate=0.99999, render=True):
  Q_table = {}
  state_counts = {}
  episodes_completed = 0
  while episodes_completed < num_episodes:
    obs = env.reset()
    obs, reward, done, info = env.step(0)
    prev_board = get_board(obs)

    action_queue = []
    target_rot = None
    target_col = None

    done = False
    first_iteration = True
    just_finished_queue = False
    prev_piece = None

    while not done:
      board = get_board(obs)

      if info["current_piece"] != prev_piece or first_iteration:
        first_iteration = False
        target_col = random.randint(0, 9)
        target_rot = random.randint(0, 3)
        action_queue = build_action_queue(target_col, target_rot)
        committed = True
        # This block is accessed the frame a new piece appears at the top. We want to save the state
        # of the board right before that piece appeared, to compare with much later (so save the previous frame)
        last_round_board = prev_board.copy()

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
        active = (board > 0) & (last_round_board == 0)
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
      prev_board = board
      env.render()
        
    episodes_completed += 1

  env.close()