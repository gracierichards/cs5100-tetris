# Do pip install gym-tetris

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)

# instructions = [6, 3, 1, 6, 3, 2]

# Get the total number of pieces dropped. Takes in info[statistics]
def total_pieces(statistics):
  total = 0
  for type0 in statistics:
    total += statistics[type0]
  return total


done = True
state = env.reset()
for step in range(10**4):
  state, reward, done, info = env.step(env.action_space.sample())
  env.render()
  if done:
    print("Total pieces dropped:", total_pieces(info["statistics"]))
    state = env.reset()
    break

env.close()