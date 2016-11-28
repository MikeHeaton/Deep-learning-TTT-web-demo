# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:16:21 2016

@author: Mike
"""

import ttt_game
import ttt_NN
import tensorflow as tf
import ttt_unsupervisedlearn as ttt_uslearn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import ttt_players
from ttt_params import Params

def testgame():
    tf.reset_default_graph()
    with tf.Session() as sess:
        params = Params()

        print("Setting up model...")
        p1 = ttt_players.ttt_Manualplayer()
        m = ttt_NN.ttt_NN(params)
        saver = tf.train.Saver()
        saver.restore(sess, './lstm_weights')
        p0 = ttt_players.ttt_NNPlayer(m, sess)
        G = ttt_game.Game(params, p0=p0, p1=p1)

        G.playgame()

def interestplot(interestvec, statevec):
    plot, axes = plt.subplots()
    ax = plot.gca()
    ax.set_autoscale_on(False)
    plt.axis([0.0,3.0,0.0,3])
    for i in range(3):
        for j in range(3):
            if statevec[3*i + j] == 1:
                ax.annotate('X',xy=(2-i+0.1,j+0.1 ),size=180)
            elif statevec[3*i + j + 9]== 1:
                ax.annotate('O',xy=(2-i+0.1,j+0.1 ),size=180)
            else:
                rect = mpatches.Rectangle((2-i,j),1,1,alpha=1,facecolor=cm.YlOrRd(0.5+0.5*interestvec[3*i+j]))
                ax.annotate("{:.4f}".format(interestvec[3*i+j]),xy=(2-i+0.3,j+0.4 ),size=20,color ='w')
                axes.add_patch(rect)
            
