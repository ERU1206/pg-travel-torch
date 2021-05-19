import gym

ID = "Pendulum-v0"

# Customize environment
def gym_env(ID):
    env = gym.make(ID)
    return env


if __name__ == '__main__':
    env = gym_env('Pendulum-v0')
    env.reset()
    print(env.action_space)
    # while True: # 코드를 멈춰서 꺼줘야 함...
    #     env.reset()
    #     while True:
    #         env.render()
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             break
    env.close()