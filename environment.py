import gym

env = gym.make('CartPole-v1')

while True: # 코드를 멈춰서 꺼줘야 함...
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
env.close()