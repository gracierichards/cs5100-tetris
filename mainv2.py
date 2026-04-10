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


def get_piece_width(rotation):
  # crude approximation (good enough for random agent)
  # cycles through common widths
  widths = [2, 3, 4, 3]  # varies by rotation
  return widths[rotation % 4]


def build_action_queue(target_col, target_rot):
  actions = []

  # Rotate first
  actions += [1] * target_rot

  # Move horizontally
  delta = target_col - 5  # Assuming the center column is 5
  if delta > 0:
    actions += [3] * delta * 10
  else:
    actions += [4] * (-delta) * 10

  # Then drop
  actions += [5] * 35  # spam down

  return actions


num_episodes = 5
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
  prev_piece = None

  while not done:
    board = get_board(obs)

    if info["current_piece"] != prev_piece or first_iteration:
      first_iteration = False
      target_col = random.randint(0, 9)
      target_rot = random.randint(0, 3)

      action_queue = build_action_queue(target_col, target_rot)
      committed = True

    if action_queue:
      action = action_queue.pop(0)
    else:
      action = 0  # NOOP

    prev_piece = info["current_piece"]
    obs, reward, done, info = env.step(action)
    if committed and not action_queue:
        committed = False
    prev_board = board
    env.render()
      
  episodes_completed += 1

env.close()