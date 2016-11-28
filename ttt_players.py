# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:23:54 2016

@author: Mike

Contains player classes for playing tic tac toe in various ways.
"""

import numpy as np
import ttt_game
import random

class Player():
    # Base class for other types of players
    def __init__(self):
        raise NotImplementedError

    def play(self, state, legalmask, epsilon=0):
        raise NotImplementedError
        
class ttt_Manualplayer(Player):
    # Asks the player for input at each step 
    def __init__(self):
        pass

    def play(self, state, epsilon=0):
        ttt_game.displaygamevec(state)
        vec = np.zeros([9])
        vec[input()] += 1
        return vec, vec

class ttt_NNPlayer(Player):
        # Implements an AI player which plays using a learned neural network.
    def __init__(self, model, sess):
        self.model = model
        self.sess = sess
        self.params = self.model.params

    def _runmodel(self, inputstate, legalmask):
        # Polls the NN model to get a good move and returns it.

        self.input = np.expand_dims(inputstate, axis=0)
        feeddict = {self.model.input_placeholder: self.input}
        scores = self.sess.run(self.model.logits, feed_dict=feeddict)[0]

        # Mask the scores to only look at _legal_ moves
        mask = (legalmask == 1)
        subset_idx = np.argmax(scores[mask])
        
        decision = np.zeros([self.model.params.LABELLEN])
        m = np.arange(scores.shape[0])[mask][subset_idx]
        decision[m] += 1
        return scores, decision

    def play(self, inputstate, legalmask, epsilon=0):
        # Poll self._runmodel to get the best move
        # Override it with probability epsilon
        # Note that training uses the overriden score, not the score of the
        # move returned by _runmodel.

        estscores, decision = self._runmodel(inputstate, legalmask)
        mask = (legalmask == 1)

        if np.random.rand() < epsilon: 
            decision *= 0
            m = np.random.choice(np.arange(decision.shape[0])[mask])
            decision[m] += 1

        return estscores, decision

class ttt_PerfectPlayer(Player):
    # Implements a deterministic AI player which plays a decent game of TTT.
    # "perfectplayer" is a misnomer, this is not one! I wanted an AI which the
    # neural network could learn to reliably beat.
    def __init__(self):
        pass

    def play(self, statevec, legalmask, epsilon=0):
        # Implements a simple set of rules to not totally suck at TTT, but also
        # not play optimally. The AI looks ahead by one step only. (as opposed
        # to the optimal minmax solution which looks N steps ahead!)
        
        self.mylayer = statevec[:9].reshape([3,3])
        self.oplayer = statevec[9:].reshape([3,3])
        self.count = 0

        allowables = [(i//3, i%3) for i in range(9) if legalmask[i] == 1]
        m = -1
        for (x,y) in allowables:
            # For each potential play, make it if it wins the game.
            newmylayer = self.mylayer.copy()
            newmylayer[x,y] = 1
            if self._haswon(newmylayer, self.oplayer) == 1:
                m = 3*x+y
                break

        if m == -1:
            # For each potential play, make it if the opponent would win the
            # game next turn by playing there.
            for (x,y) in allowables:
                newoplayer = self.oplayer.copy()
                newoplayer[x,y] = 1
                if self._haswon(self.mylayer, newoplayer) == -1:
                    m = 3*x+y
                    break

        if m == -1:
            # If neither of the above, just play randomly.
            x,y = random.choice(allowables)
            m = 3*x + y

        playvec = np.zeros([9])
        playvec[m] = 1
        return playvec, playvec

    def _haswon(self, mylayer, oplayer):
        for p, layer in ((1,mylayer), (-1, oplayer)):
            for i in range(3):
                if (np.sum(layer[i,:])  == 3 or
                    np.sum(layer[:,i]) == 3) :
                    return p
            if (    np.sum([layer[i,i] for i in range(3)]) == 3 or
                    np.sum([layer[i,2-i] for i in range(3)]) == 3 ):
                return p
        return 0

    def _hasdrawn(self, mylayer, oplayer):
        print(np.sum(mylayer + oplayer), end="")
        if np.sum(mylayer + oplayer) == 9:
                return True
        return False

class ttt_MinMaxPlayer(ttt_PerfectPlayer):
    def __init__(self, playernumber):
        self.playerno = playernumber
        self.playdict = {}

    def play(self, statevec, legalmask, epsilon=0):
        self.mylayer = statevec[:9].reshape([3,3])
        self.oplayer = statevec[9:].reshape([3,3])
        bestplay = self._minmax(self.playerno, self.mylayer, self.oplayer,-3,3)

        m = 3*bestplay[0] + bestplay[1]
        playvec = np.zeros([9])
        playvec[m] = 1
        return playvec, playvec

    def _allowableplays(self, playerlayer, oplayer):
        allowables = [(i,j) for i in [0,1,2] for j in [0,1,2] if
                      (playerlayer + oplayer)[i][j] == 0]
        return allowables

    def _apply(self, pos, curplayer, playerlayers):
        layers = [playerlayers[0].copy(), playerlayers[1].copy()]
        layers[curplayer][pos[0],pos[1]] = 1
        return layers

    def _minmax(self, curplayer, p0layer, p1layer, alpha, beta):

        statevector = (  "".join([str(p0layer[i][j]) for i in [0,1,2] for j in
                           [0,1,2]]) 
                               +
                        "".join([str(p1layer[i][j]) for i in [0,1,2] for j in
                               [0,1,2]])  )
        print(statevector)
        if statevector in self.playdict:
            return self.playdict[statevector]
        
        winner = self._haswon(p0layer, p1layer)  
        if winner != 0:
            self.playdict[statevector] = (0, winner)
            return (0, winner)
        elif self._hasdrawn(p0layer, p1layer):
            self.playdict[statevector] = (0, 0)
            return (0, 0)
        else:
            allowables = self._allowableplays(p0layer, p1layer)
            np.random.shuffle(allowables)
            layers = [p0layer, p1layer]

            if curplayer == 1:
                move = (3,3)
                v = (-3, (3,3))
                for (i,j) in allowables:
                        v = max(v, 
                                (self._minmax(0,
                                                *self._apply((i,j),curplayer,
                                                             [p0layer,p1layer]),
                                                                alpha, beta),
                                 (i,j)),
                                key = lambda x : x[0])
                        alpha = max(alpha, v[0])
                        if beta <= alpha:
                            break
            if curplayer == 0:
                move = (3,3)
                v = (3, (3,3))
                for (i,j) in allowables:
                    v = min(v, 
                            (self._minmax(0,
                                            *self._apply((i,j),curplayer,
                                                         [p0layer,p1layer]),
                                                            alpha, beta),
                             (i,j)),
                            key = lambda x : x[0])
                    beta = min(beta, v[0])
                    if beta <= alpha:
                        break
            return v 
            
if __name__ == "__main__":
    statevec = np.array([0]*18)
    test = ttt_MinMaxPlayer(1)
    test.play(statevec,'X')
    print("One down")
    test.play(statevec,'X')
