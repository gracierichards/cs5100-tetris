# Do pip install gym-tetris

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)

instructions = [6, 3, 1, 6, 3, 2]

done = True
state = env.reset()
for step in range(10**4):
  if step % 3 == 0:
    action = 6
  elif step % 3 == 1:
    action = 3
  else:
    action = 1
  state, reward, done, info = env.step(action)
  env.render()
  if done:
    state = env.reset()
    break

env.close()