from splendor_env import *

GM = GameManager("Aircraft")
GM.join_game()
GM.join_game()
GM.start_game()

env = GM.game

s = env.reset()
print(env.step([1,1,1,0,0,0,0]))
print()
print(env.step([0,0,1,1,1,0,0]))
print()
print(env.step([1,1,1,0,0,0,0]))
print()
print(env.step([0,0,1,1,1,0,0]))
print()
print(env.step([1,1,1,0,0,0,0]))
print()
print(env.step([0,0,1,1,1,0,0]))
print()


print(env.step([0,0,0,0,0,1,0]))