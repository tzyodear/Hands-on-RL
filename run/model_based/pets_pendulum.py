from rl_utils import ReplayBuffer
from model_based.pets import PETS
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    buffer_size = 100000
    n_sequence = 50
    elite_ratio = 0.2
    plan_horizon = 25
    num_episodes = 10
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    replay_buffer = ReplayBuffer(buffer_size)
    pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes)
    return_list = pets.train()

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PETS on {}'.format(env_name))
    plt.show()
