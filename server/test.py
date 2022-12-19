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

#step 메소드 테스트
#game.step([1,1,0,1,0,0,0])
#game.step([0,0,0,0,0,1,3])

# print(env.step([0,0,0,0,0,1,0]))