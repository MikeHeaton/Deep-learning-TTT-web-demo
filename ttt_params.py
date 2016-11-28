

class Params():
    def __init__(self):
        """NN params"""
        self.INPUTLEN = 18
        self.LABELLEN = 9
        self.LEARNING_RATE = 0.001
        self.HIDDENSIZE = 100
        self.SAVEEVERY = 100
        self.BATCHSIZE = 16

        """Reinforcement learning params"""
        self.GAMMA = 0.99
        self.UNSUP_SAMPLESIZE = 1024
        self.UNSUP_BATCHLENGTH = 1024

        self.TRAINTIME = 1000
        self.EPSILON_INIT = 1
        self.EPSILON_FINAL = 0.1
