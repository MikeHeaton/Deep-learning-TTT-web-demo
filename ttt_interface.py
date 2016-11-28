"""



"""

import ttt_NN
import ttt_players
import ttt_game
import tensorflow as tf
from ttt_params import Params

def setup_model(session):
    tf.reset_default_graph()
    m = ttt_NN.ttt_NN(Params())

    saver = tf.train.Saver()
    saver.restore(session, './lstm_weights')
    
    G = ttt_game.NetworkGame(m, session)
    
    return G.playmove
            



    """Want a function which takes in a state and a move, updates the state,
    asks the NN for its move, updates the state, and returns."""






