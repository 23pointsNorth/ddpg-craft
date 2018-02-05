import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from expert_agent import calculate_expert_action
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import timeit

import gym
import cogle_mavsim

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import random
import numpy as np 

class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

OU = OU()       # Ornstein-Uhlenbeck Process

def runSimulation(train_indicator=False, id=None, dagger_eps=.0):
    BUFFER_SIZE = 400
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 0.001/10     # Target Network HyperParameters
    LRA = 0.0001    # Learning rate for Actor
    LRC = 0.001     # Lerning rate for Critic
    MODEL_DIR = './models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make('CoGLEM1-v0')

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    np.random.seed(0)

    print('Action dim {}\r'.format(action_dim))
    print('State dim {}\r'.format(state_dim))

    EXPLORE = 1000.
    noise_eps = 0.1
    episode_count = 1000
    max_steps = 50
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)

    if (not train_indicator) and id:
        print('Load netowrks!\r')
        try:
            actor.model.load_weights(os.path.join(MODEL_DIR, 'actormodel_' + str(id) + '.h5'))
            critic.model.load_weights(os.path.join(MODEL_DIR, 'criticmodel_' + str(id) + '.h5'))
            actor.target_model.load_weights(os.path.join(MODEL_DIR, 'actormodel_' + str(id) + '.h5'))
            critic.target_model.load_weights(os.path.join(MODEL_DIR, 'criticmodel_' + str(id) + '.h5'))
            print('Weight load successfully.\r')
        except:
            print('Cannot find the weight for id: \'{}\' at {}\r'.format(id, 
                    os.path.join(MODEL_DIR, 'actormodel_' + str(id) + '.h5')))

    print('CoGLEM1-v0 Experiment Start.\r')
    for i in range(episode_count):
        print('Episode : ' + str(i) + ' Replay Buffer ' + str(buff.count()) + '\r')
        if dagger_eps > .0:
            print(' >>> Using DAGGER!\r')
        if not train_indicator:
            print('Running in evaluation mode!\r')

        obs = env.reset()
        s_t = obs
     
        total_reward = 0.
        step = 0
        done = False
        while not done:
            loss = 0 
            epsilon -= (1. / EXPLORE)
            a_t = np.zeros([action_dim])
            noise_t = np.zeros([action_dim])
            
            # Use DAGGER or not
            if train_indicator and np.random.uniform() < dagger_eps:
                # Get expert action!
                a_t = calculate_expert_action(s_t)
                print('Using expert agent :    {}\r'.format(a_t))
            else:
                if (np.random.uniform() < noise_eps):
                    a_t_original = np.random.uniform(size=2) * 2. - 1
                    print('Doing random action!\r')
                else:
                    # Get agent action
                    a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))[0]
                    print('Original values:        {}\r'.format(a_t_original))
                    a_expert = calculate_expert_action(s_t)
                    a_t_original[0] = a_expert[0] # FIX HEIGHT
                    a_t_original[1] = a_expert[1] 
                    print('Fixed height to expert height.\r')
                print('Original values:        {}\r'.format(a_t_original))
                noise_t[0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0],  0.2 , 0.5, 0.2)
                noise_t[1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[1],  0.0 , 0.5, 0.2)
                
                a_t[0] = a_t_original[0] + noise_t[0] * .0
                a_t[1] = a_t_original[1] + noise_t[1] * .0 ###### REMOVED NOISE

                # Constrain actions to [-1, 1]
                for j in range(len(a_t)):
                    while a_t[j] < -1:
                        a_t[j] += 2
                    while a_t[j] > 1:
                        a_t[j] -= 2
                print('Noise augmented values: {}\r'.format(a_t))

            obs, r_t, done, info = env.step(a_t)
            if obs is None:
                done = True
                print('Failed to get agent data. Reseting...\r')
            else:
                s_t1 = obs
            
                buff.add(s_t, a_t, r_t, s_t1, done) # Add to replay buffer
                
                # Batch update
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[1] for e in batch])
                
                # print('y_t : {}'.format(y_t))
                
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
                
                # print('Target model critic: {}'.format(target_q_values))
                
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA*target_q_values[k]
                
                # print('new y_t : {}'.format(y_t))
                
                if (train_indicator): # and i > 2
                    loss += critic.model.train_on_batch([states,actions], y_t) 
                    a_for_grad = actor.model.predict(states)
                    grads = critic.gradients(states, a_for_grad)
                    actor.train(states, grads)
                    actor.target_train()
                    critic.target_train()

                total_reward += r_t
                s_t = s_t1
            
            print('Episode ' + str(i) + ' Step ' + str(step) + ' Action ' + str(a_t) + \
                  ' Reward ' + str(r_t) + ' Loss ' + str(loss) + '\r')
        
            step += 1

        if (train_indicator and i % 2 == 0):
            print('Saving models...\r')
            actor.model.save_weights(os.path.join(MODEL_DIR, 'actormodel_' + str(i) + '.h5'), overwrite=True)
            with open(os.path.join(MODEL_DIR, 'actormodel_' + str(i) + '.json'), 'w') as outfile:
                json.dump(actor.model.to_json(), outfile)

            critic.model.save_weights(os.path.join(MODEL_DIR, 'criticmodel_' + str(i) + '.h5'), overwrite=True)
            with open(os.path.join(MODEL_DIR, 'criticmodel_' + str(i) + '.json'), 'w') as outfile:
                json.dump(critic.model.to_json(), outfile)

        print('TOTAL REWARD @ ' + str(i) +'-th Episode - Reward ' + str(total_reward) + '\r')
        print('Total Step: ' + str(step) + '\r')
        print('\n\r')

    env.close()
    print('Finish.')

if __name__ == '__main__':
    load_model='expert'
    if len(sys.argv) > 1:
        load_model = sys.argv[1]
        print('Using id: {}'.format(load_model))
    runSimulation(train_indicator=True)
    # runSimulation(train_indicator=True, dagger_eps=0.2)
    # runSimulation(train_indicator=False, id=load_model)
