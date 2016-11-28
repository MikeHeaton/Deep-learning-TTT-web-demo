# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:19:31 2016

@author: Mike
"""

from ttt_params import Params
import ttt_NN
import tensorflow as tf
import numpy as np
import ttt_players
import matplotlib.pyplot as plt
import random
import ttt_game
from collections import Counter
AIPLAYER = 1

def getbatch(G, epsilon):
    counter = 0
    history = []
    scorehistory = []
    wintype_history = []
    params = G.params
    while counter < params.UNSUP_BATCHLENGTH:
        # Play new games until we've recorded as many states as requested.

        iterator = G.gameiterator(newgame=True, epsilon=epsilon)
        gamehistory = [] 

        for gso in iterator:
            if gso["curplayer"] == AIPLAYER:
                gamehistory.append(gso)
        
        # After the game finishes, record the score and win type.
        # Remember, score is -1 if P0 wins, 1 if P1 wins, 0 if a draw
        # Need to maximise score so reverse it if AIPLAYER==0
        scorehistory.append(gso["score"] * (2*AIPLAYER-1))
        wintype_history.append(gso["wintype"])
        
        for t in range(len(gamehistory)-1, -1, -1):
            # Loop backwards over the game history (backwards is needed to get 
            # rewards at time t+1)
            state = gamehistory[t]["inputvec"]
            action = gamehistory[t]["thisplay"]
            
            if t == len(gamehistory) - 1: 
                gamehistory[t]["reward"] = scorehistory[-1]
            else:
                gamehistory[t]["reward"]  = gamehistory[t]["score"]
            
            # Label formula is:
            # Q(s,a) = r_(t+1) + GAMMA * max Q(s',a') 
            # with s' the state at t+1
            # and a' ranging over possible actions.
            if t == len(gamehistory) - 1:
                gamehistory[t]["label"]  = gamehistory[t]["reward"] 
            else:
                gamehistory[t]["label"]  = ( params.GAMMA * 
                            np.max([gamehistory[t+1]["scoresestimate"][i] 
                            for i in range(len(gamehistory[t+1]["scoresestimate"])) 
                            if gamehistory[t+1]["legalmask"][i] == 1])
                           )
            label = gamehistory[t]["label"]
            history.append((state, action, label))
            counter += 1
            
    print(Counter(wintype_history))

    return history, scorehistory

def runepoch(m, sess, epsilon):
    # Run an epoch using m to play p0.

    print("Setting up model...")
    params = m.params
    if AIPLAYER == 1:
        p0 = ttt_players.ttt_perfectplayer()
        p1 = ttt_players.ttt_NNPlayer(m, sess)
    elif AIPLAYER == 0:
        p1 = ttt_players.ttt_perfectplayer()
        p0 = ttt_players.ttt_NNPlayer(m, sess)
    else:
        print("AIPLAYER param not 1 or 0, not recognised.")
    G = ttt_game.Game(params, p0=p0, p1=p1)

    print("Generating play...")
    history, scorehistory = getbatch(G, epsilon)

    print("Sampling history...")
    sample = random.sample(history, params.UNSUP_SAMPLESIZE)
    minibatches = batchsample(sample, params)

    print("Training...")
    trainingloss = 0
    for states, actions, labels in minibatches:
        feeddict = {m.input_placeholder: states,
                    m.label_placeholder: actions,
                    m.score_placeholder: labels                    
                    }
        _, loss = sess.run([m.reinforcement_train_op, m.reinforcementloss], feed_dict = feeddict)
        trainingloss += loss

    print("Testing...")
    testscore = G.playgame()
    return trainingloss, np.mean(scorehistory), testscore

def batchsample(sample, params):
    # Takes a sample of states, and batches them into a form acceptable to the NN.
    # Samples are (statehistory, action, label)

    states_chunked = [[z[0] for z in sample[0+t:params.BATCHSIZE+t]] for t in range(0, len(sample), params.BATCHSIZE)]
    actions_chunked = [[z[1] for z in sample[0+t:params.BATCHSIZE+t]] for t in range(0, len(sample), params.BATCHSIZE)]
    rewards_chunked = [[z[2] for z in sample[0+t:params.BATCHSIZE+t]] for t in range(0, len(sample), params.BATCHSIZE)]
    return zip(states_chunked, actions_chunked, rewards_chunked)

def runtraining(params, usesaved=True):
    tf.reset_default_graph()
    with tf.Session() as sess:

        m = ttt_NN.ttt_NN(params)
        saver = tf.train.Saver()
        if usesaved:
            saver.restore(sess, './lstm_weights')
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        genscorehistory = []
        trainlosshistory = []
        testscorehistory = []
        epsilon = m.params.EPSILON_INIT

        for t in range(m.params.TRAINTIME):
            if epsilon > m.params.EPSILON_FINAL:
                epsilon -= 1/m.params.TRAINTIME
            print("--EPOCH {:d}--".format(t))
            
            trainloss, genscore, testscore = runepoch(m, sess, epsilon)
            genscorehistory.append(genscore)
            trainlosshistory.append(trainloss)
            testscorehistory.append(testscore)

            print("Epsilon: ", epsilon) 
            print("Test average score: {:f}".format(genscore))
            print("Training loss: {:f}".format(trainloss))
            print("(Deterministic) game test score: {:d}".format(testscore))

            if t % params.SAVEEVERY == 0 and t > 0:
                saver.save(sess, './network_weights')
                print("SAVED")

        plt.rcParams["figure.figsize"] = (9,9)
        plt.figure(1)
        plt.subplot(311)
        plt.plot(trainlosshistory)
        plt.title("Train Loss History")
        plt.subplot(312)
        plt.plot(genscorehistory)
        plt.title("Generating (with Softmax) Scores History")
        plt.subplot(313)
        print(testscorehistory)
        plt.plot(testscorehistory)
        plt.title("Testing (best guess) Scores History")

    return trainlosshistory, genscorehistory, testscorehistory

if __name__ == '__main__':
    runtraining(Params(), usesaved=False)
