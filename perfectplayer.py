import numpy as np
import random
QUERYSTATE = np.array([[ 2., -1.,  0.], [ 0.,  1.,  0.], [-1.,  0.,  0.]])

class ttt_MinMaxPlayer():
    def __init__(self, playerno):
        self.playerno = playerno
        self.statedict = {}

    def play(self, statevec, legalmask, epsilon=0):
        board = makeTTTBoard(statevec)
        print(board.state)
        bestplays = self._minmax(self.playerno, board, -1, 1, depth=False)
        return bestplays
        return random.choice(bestplays[1])

    def _minmax(self, curplayer, position, alpha, beta, depth=False):
        """ If we already have the state in our dictionary, use that"""
        # print(position.as_tuple())
        #if position.as_tuple() in self.statedict:
        #    return self.statedict[position.as_tuple()]

        """ If position is terminal, return score"""
        result = position.eval()
        if result[0]: #result is a pair (finished?, score)
            # print("Terminal")
            self.statedict[position.as_tuple()] = (result[1], None)
            return (result[1], None)
        justnow = False
        if np.array_equal(position.state, QUERYSTATE):
            depth = True
            justnow = True

        """ Else loop through possible moves and update alpha/beta """
        legalmoves = position.legalmoves()
        # print(position.state, alpha, beta)
        if curplayer == -1:
            v = 2
            bestmoves = []
            for (X, Y) in legalmoves:
                if depth == 1:
                    pass
                    # print(position.add(X, Y, 1).state)
                    # print(curplayer, v, bestmoves)
                #print("Try ", (X,Y))
                v_xy, move = self._minmax(1,
                                    position.add(X, Y, -1),
                                    alpha,
                                    beta, depth=depth)
                if v_xy == v:
                    if justnow:
                        print("appending -- ", v_xy, (X,Y))
                    bestmoves.append(("append", v_xy, (X, Y)))
                if v_xy < v:
                    v = v_xy
                    if justnow:
                        print("new -- ", v_xy, (X,Y))
                    bestmoves = [("new", v_xy, (X, Y))]
                if v < beta:
                    beta = v
                #if beta <= alpha:
                    #print("BREAKING")
                    #break
            self.statedict[position.as_tuple()] = (v, bestmoves)
            if depth:
                print("Returning ---")
                print(position.state, curplayer, v, bestmoves)
            if justnow:
                x = input()
            return (v, bestmoves)

        else:
            v = -2
            bestmoves = []
            for (X, Y) in legalmoves:
                #print("Try ", (X,Y))
                if depth == 1:
                    pass
                    # print(position.add(X, Y, 1).state)
                    # print(curplayer, v, bestmoves)
                v_xy, move = self._minmax(-1,
                                    position.add(X, Y, 1),
                                    alpha,
                                    beta, depth=depth)
                if v_xy == v:
                    if justnow:
                        print("appending -- ", v_xy, (X,Y))
                    bestmoves.append(("append", v_xy, (X, Y)))
                if v_xy > v:
                    v = v_xy
                    if justnow:
                        print("new -- ", v_xy, (X,Y))
                    bestmoves = [("new", v_xy, (X, Y))]
                if v > alpha:
                    alpha = v
                #if beta <= alpha:
                    #print("BREAKING")
                    #break
            self.statedict[position.as_tuple()] = (v, bestmoves)

            if depth:
                print("Returning ---")
                print(position.state, curplayer, v, bestmoves)
            if justnow:
                x = input()
            return (v, bestmoves)

def makeTTTBoard(statevec):
    state = np.zeros((3,3))
    for i in range(9):
        X = i // 3
        Y = i % 3
        if statevec[i] == "1":
            state[X, Y] = -1
        if statevec[i+9] == "1":
            state[X, Y] = 1
    return TTTBoard(state)

class TTTBoard():
    def __init__(self, state):
        """ state should be a 3x3 integer matrix
        0 for empty; 1 for O (p1); -1 for X (p0)"""
        self.state = state

    def as_tuple(self):
        return tuple(map(tuple, self.state))

    def eval(self):

        """Evaluate the state. Returns a tuple -
        result[0] is True if the game has terminated, with result[1] being the
        result. Otherwise result[1] is False if the game is to keep going."""
        has_won = self._haswon()
        if has_won != 0:
            return (True, has_won)

        if self._hasdrawn():
            return (True, 0)

        return (False, False)

    def add(self, X, Y, player):
        # Adds a spot at X,Y
        newstate = self.state.copy()
        newstate[X, Y] = player
        return TTTBoard(newstate)

    def legalmoves(self):
        # Returns a list of legal moves (X,Y).
        legalmvs = [(X,Y) for (X,Y) in [(0,0), (0,1), (0,2),
                                    (1,0), (1,1), (1,2),
                                    (2,0), (2,1), (2,2)]
                if self.state[X,Y] == 0]
        return legalmvs

    def _haswon(self):
        # Checks board for winners. Returns 1/-1 if there's a winner or 0 else.
        for p in [-1, 1]:
            # Check rows / columns
            for i in range(3):
                if (np.sum(self.state[i,:])  == p * 3 or
                    np.sum(self.state[:,i]) == p * 3) :
                    #print(self.state, p)
                    return p

            # Check diagonals
            if (    np.sum([self.state[i,i] for i in range(3)]) == p * 3 or
                    np.sum([self.state[i,2-i] for i in range(3)]) == p * 3 ):
                #print(self.state, p)
                return p
        return 0

    def _hasdrawn(self):
        # If there's an empty square on the board, we're not done yet.
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if self.state[i,j] == 0:
                    return False
        #print("---drawn")
        #print(self.state)
        #print("---")
        return True

if __name__ == '__main__':
    #test = makeTTTBoard("100010001000000000")
    #test.eval()
    test = ttt_MinMaxPlayer(1)
    #print(test.play("110000000001100000", None, epsilon=0))
    print(test.play("010000000000010000", None, epsilon=0))
    print(len(test.statedict))

"""
Returning ---
[[-1. -1.  1.]
 [ 1.  0.  0.]
 [ 0.  0.  0.]] -1 0 [(1, 1)]

Returning ---
[[ 1. -1.  0.]
 [ 0. -1.  0.]
 [ 0.  1.  0.]] -1 0 [(0, (0, 2)), (0, (1, 0)), (0, (1, 2)), (0, (2, 0)), (0, (2, 2))]

 Returning ---
[[ 1. -1.  0.]
 [ 0.  1.  0.]
 [-1.  0.  0.]] -1 0 [('new', 0, (1, 0)), ('append', 0, (1, 2)), ('append', 0, (2, 2))]

"""
