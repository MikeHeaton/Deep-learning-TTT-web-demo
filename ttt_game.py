#-*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:40:45 2016

@author: Mike
"""

import numpy as np
import ttt_test
import ttt_players

class Game():
    """------Managing states and moves------"""

    def __init__(self, params, p0=None, p1=None):
        # Cards are represented by a params.DECKSIZE x (params.DECKSIZE+1) arr:
        # Each row is a card, a 1hot with the position of the card in hands
        # or in the last column if its already been played.
        if not p0:
            p0 = ttt_players.Baselineplayer(params) 
        if not p1:
            p1 = ttt_players.Baselineplayer(params)
        self.params = params
        self.player = [p0, p1]
        self.initstate()
        self.wintype = 0

    def initstate(self):
        self.state = np.zeros([3, 3, 2])
        self.score = 0
        self.finished = False
        self.wintype = 0

    def getstate(self, player):
        """State format:
        00|00|01
        --------
        00|10|00
        --------
        00|00|10
        3x3x2 array, each 2-dim is a 1hot for P0/P1 respectively.
        The statevector is from the player's perspective: a 9-length vector
        with all the player's marks, concatenated with a 9-length
        vector with all the opponent's marks."""

        playerlayer = self.state[:, :, player]
        opplayer = self.state[:, :, 1 - player]
        self.statevector = np.concatenate((playerlayer.flatten(),
                                           opplayer.flatten()), axis=0)
        return self.statevector

    def getacceptabilitymask(self):
        """Returns a filter of moves which are acceptable.
        Allows the player to know which moves are genuinely available."""
        self.acceptabilitymask = np.zeros([9])
        for N in range(9):
            X = N // 3
            Y = N % 3
            empty = 1 - np.sum(self.state[X, Y, :])
            self.acceptabilitymask[N] = empty
        return self.acceptabilitymask

    def updatestate(self, player, decision):
        """Applies a decision vector to update the state.
        Decision vector format:
        A one-hot matrix, play at one of the 3x3 places.
        000
        000
        000
        
        [0,0,0,1,0,0,0,0,0]
        Received as a 1x9 vector 1hotted at N which is interpreted as
        X = N//3, Y = N%3."""
        
        self.finished = False
        val = np.argmax(decision)
        X = val // 3
        Y = val % 3
        #print(X,Y)
        if self.state[X, Y, 1 - player] == 1:
            # If already played in this space, the player loses.
            self.score = 1 - 2 * player
            self.wintype = "Goofed"
            print("ERROR - play in illegal spot")
            self.finished = True
        else:
            # Else put a mark into this space
            self.state[X, Y, player] = 1

            # and check for wins:
            for p in [0, 1]:
                for i in range(3):
                    if (np.sum(self.state[i, :, p]) == 3 or
                            np.sum(self.state[:, i, p]) == 3):
                        self.score = 2*p - 1
                        self.wintype = "Row or col"
                        self.finished = True
                if (np.sum([self.state[i, i, p]
                            for i in range(3)]) == 3 or
                    np.sum([self.state[i, 2 - i, p]
                            for i in range(3)]) == 3):
                    self.score = 2 * p - 1
                    self.wintype = "Diagonal"
                    self.finished = True
            if np.sum(self.state) == 9:
                self.finished = True
                self.wintype = "Draw"

        return self.finished

    """------Playing a game------"""

    def gameiterator(self, newgame=True, epsilon=0):
        if newgame:
            self.initstate()
        
        curplayer = 0
        count = 0

        while self.finished is False:
            state = self.getstate(curplayer)
            inputvec = state
            legalmask = self.getacceptabilitymask()

            scoresestimate, thisplay = self.player[curplayer].play(inputvec,
                                                             legalmask,
                                                              epsilon=epsilon)
            self.updatestate(curplayer, thisplay)

            # Assign all useful information to a dictionary, and return it
            # NOTE score == -1 <=> Player 0 has won
            # score == 1 <=> Player 1 has won
            gso = {}
            gso["inputvec"] = inputvec
            gso["thisplay"] = thisplay
            gso["score"] = self.score
            gso["curplayer"] = curplayer
            gso["scoresestimate"] = scoresestimate
            gso["wintype"] = self.wintype
            gso["legalmask"] = legalmask

            yield gso

            curplayer = 1 - curplayer
            count += 1
            if count > 100:
                print("GAME TIMED OUT")
                break

    def playgame(self, verbose=False):
        # Play a whole game, outputting states to the console as we go.
        gameiterator = self.gameiterator(epsilon=0)
        for s in gameiterator:
            if verbose:
                printtttstate(s["inputvec"])
                print(s["scoresestimate"])
            else:
                pass
        return self.score

class NetworkGame(Game):
    """Simpler class with a method for taking in a state and a move, updating
    the state, askingan NN for its move, updating the state, and returning."""
    def __init__(self, model, sess):
        self.aiPlayer = ttt_players.ttt_NNPlayer(model, sess)
        self.state = None
        self.session = sess

    def playmove(self, inputvector):
        # Interpret inputvector as the state
        self.ravelinputvector(inputvector)
  
        # Get AI player move
        legalmask = self.getacceptabilitymask()
        estscores, aimove = self.aiPlayer.play(inputvector, legalmask, epsilon=0)

        # Update state for move
        self.updatestate(1,aimove)

        # Return new state
        return self.getstate(0)   

    def ravelinputvector(self, inputvector):
        self.state = np.zeros([3,3,2])
        xvector = inputvector[:9]
        ovector = inputvector[9:]

        self.state[:,:,0] = np.reshape(xvector, [3,3])
        self.state[:,:,1] = np.reshape(ovector, [3,3])


def printtttstate(state):
    print("---")
    for Y in range(3):
        for X in range(3):
            if state[3*Y + X] == 1:
                print("X", end=""),
            elif state[3*Y + X + 9] == 1:
                print("0", end=""),
            else:
                print(".", end=""),
        # print("")
