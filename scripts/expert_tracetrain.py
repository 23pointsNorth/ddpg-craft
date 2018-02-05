from gym_recording import playback
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD, Adam
import tensorflow as tf
from ActorNetwork import ActorNetwork
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


state_dim = 10
action_dim = 2
BATCH_SIZE = 32
TAU = 0.99
LEARNING_RATE = 0.001 
NUMBER_OF_EPOCHS = 100
MODEL_DIR = './models/'

all_obs = np.empty((0, state_dim)) 
all_a = np.empty((0, action_dim)) 

def handle_ep(observations, actions, rewards, infos):
    global all_a, all_obs
    print('An episode began!')
    skipped = True
    for obs, a, r, i in zip(observations, actions, rewards, infos):
        if not skipped: # Skip fist action when it is to send craft somewhere
            skipped = True
            continue
        if len(a) == 0 or len(obs) == 0:
            continue
        # print(': {} {} {} {}'.format(obs, a, r, i))
        # print(': {} {}'.format(obs, a))
        # print(all_a.shape)
        # print(a.shape)
        # print(a)

        all_obs = np.vstack((all_obs, obs)) 
        all_a = np.vstack((all_a, a))



if __name__ == '__main__':
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE)

    playback.scan_recorded_traces('./traces', handle_ep)
    print('Recorded {} pairs'.format(len(all_a)))
    print(all_a.shape)
    print(all_obs.shape)

    # Train on pairs
    # print(all_obs)
    # print(all_a)

    opt = Adam(lr=LEARNING_RATE)
    actor.model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['mse'])
    callback_list=[]
    actor.model.fit(all_obs, all_a,
                  batch_size=BATCH_SIZE,
                  epochs=NUMBER_OF_EPOCHS,
                  validation_split=0.1,
                  shuffle=True,
                  callbacks=callback_list,
                  class_weight = {0:1, 1:10})

    print('------')
    print(actor.model.predict(all_obs))
    print('------')
    print(all_a)


    print('Save model')
    actor.model.save_weights(os.path.join(MODEL_DIR, 'actormodel_expert.h5'), overwrite=True)
    with open(os.path.join(MODEL_DIR, 'actormodel_expert.json'), 'w') as outfile:
        json.dump(actor.model.to_json(), outfile)

    print('Done')