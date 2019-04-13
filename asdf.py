
# import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from joblib import dump
from statistics import median, mean
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense


from keras.models import Sequential
from keras.layers import Dense

LR = 1e-3
# env = gym.make("CartPole-v0")
# env.reset()
# goal_steps = 500
# score_requirement = 50
# initial_games = 10000

# def some_random_games_first():
#     # Each of these is its own game.
#     for episode in range(5):
#         env.reset()
#         # this is each frame, up to 200...but we wont make it that far.
#         for t in range(200):
#             # This will display the environment
#             # Only display if you really want to see it.
#             # Takes much longer to display it.
#             # env.render()
#
#             # This will just create a sample action in any environment.
#             # In this environment, the action can be 0 or 1, which is left or right
#             action = env.action_space.sample()
#
#             # this executes the environment with an action,
#             # and returns the observation of the environment,
#             # the reward, if the env is over, and other info.
#             observation, reward, done, info = env.step(action)
#             if done:
#                 break
#
#
# def initial_population():
#     # [OBS, MOVES]
#     training_data = []
#     # all scores:
#     scores = []
#     # just the scores that met our threshold:
#     accepted_scores = []
#     # iterate through however many games we want:
#     for _ in range(initial_games):
#         score = 0
#         # moves specifically from this environment:
#         game_memory = []
#         # previous observation that we saw
#         prev_observation = []
#         # for each frame in 200
#         for _ in range(goal_steps):
#             # choose random action (0 or 1)
#             action = random.randrange(0 ,2)
#             # do it!
#             observation, reward, done, info = env.step(action)
#
#             # notice that the observation is returned FROM the action
#             # so we'll store the previous observation here, pairing
#             # the prev observation to the action we'll take.
#             if len(prev_observation) > 0 :
#                 game_memory.append([prev_observation, action])
#             prev_observation = observation
#             score +=reward
#             if done: break
#
#         # IF our score is higher than our threshold, we'd like to save
#         # every move we made
#         # NOTE the reinforcement methodology here.
#         # all we're doing is reinforcing the score, we're not trying
#         # to influence the machine in any way as to HOW that score is
#         # reached.
#         if score >= score_requirement:
#             accepted_scores.append(score)
#             for data in game_memory:
#                 # convert to one-hot (this is the output layer for our neural network)
#                 if data[1] == 1:
#                     output = [0 ,1]
#                 elif data[1] == 0:
#                     output = [1 ,0]
#
#                 # saving our training data
#                 training_data.append([data[0], output])
#
#         # reset env to play again
#         env.reset()
#         # save overall scores
#         scores.append(score)
#
#     # just in case you wanted to reference later
#     training_data_save = np.array(training_data)
#     np.save('saved.npy' ,training_data_save)
#
#     # some stats here, to further illustrate the neural network magic!
#     print('Average accepted score:' ,mean(accepted_scores))
#     print('Median score for accepted scores:' ,median(accepted_scores))
#     print(Counter(accepted_scores))
#
#     return training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def create_sequential(input_size):

    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=input_size,activation='relu'))
    model.add(Dense(128, input_dim=64,activation='relu'))
    model.add(Dense(256, input_dim=128,activation='relu'))
    model.add(Dense(512, input_dim=256,activation='relu'))
    model.add(Dense(256, input_dim=512,activation='relu'))
    model.add(Dense(128, input_dim=256,activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(training_data, model=False):
    print(training_data)
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data])
    print(X.shape)
    print(y.shape)
    if not model:
        model = create_sequential(input_size=len(X[0]))

    model.fit(X, y, epochs=150, batch_size=10)
    # model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

# model = sequential(2)
# print("Done creating sequential")
training_data = []
with open('testfile.txt','r') as f:
    x = f.read().split()
    print(len(x))
    a = int(len(x) /2)
    for index in range(a):
        array = []
        array.append(([float(x[2*index])]))
        if (x[2*index + 1] == 'fold'):
            array.append([1,0,0])
        elif (x[2*index + 1] == 'call'):
            array.append([0,1,0])
        elif (x[2*index + 1] == 'raise'):
            array.append([0,0,1])
        training_data.append(array)
model = train_model(training_data)
# print("done")
model.save('mymodel.h5')
# dump(model, 'mymodel.joblib')
test = np.array([1.00])

test.shape = (1,1)
print(model.predict(test))