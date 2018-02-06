import cogle_mavsim
import gym
import math
import numpy as np
import os
from gym_recording.wrappers import TraceRecordingWrapper

def calculate_expert_action(obs):
    *ms, ra, dist, angle = obs
    print('MS: ' + str(ms) + '\r')

    delta_angle = 180. / (len(ms) - 1)
    ms_angles = [i * delta_angle - 90 for i in range(len(ms))]

    if any(m > -0.9 for m in ms) and \
       (-90 < angle * 180.) and (angle * 180. < 90):
        lambda_weight = 0.75
        dirs_error = [abs(msa - lambda_weight * angle * 180.) for msa in ms_angles]
        angle = math.radians(dirs_error.index(min(dirs_error)) * delta_angle - 90.) / math.pi

    alt_error = ra - 600. / 3000
    pitch = -max(min(20 * alt_error, 1), -1)
    yaw = -angle

    return np.array([pitch, yaw])


def expert_agent():
    env = gym.make('CoGLEM1-virtual-v0')
    os.makedirs('./traces', exist_ok=True)
    env = TraceRecordingWrapper(env, directory='./traces/', buffer_batch_size=10)
    
    ITERATIONS = 10

    for x in range(ITERATIONS):
        obs = env.reset()
        done = False
        while not done:
                env.render()
                action = calculate_expert_action(obs)
                print('x: ',x, 'Doing action: ', action, ' ', env.env._elapsed_steps, '\r')
                obs, reward, done, info = env.step(action)
                print('observations: ', obs, ' ', reward, ' ', done, '\r')

    env.close()


if __name__ == '__main__':
    expert_agent()