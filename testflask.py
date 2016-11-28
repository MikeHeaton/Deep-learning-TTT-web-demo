import tensorflow as tf
from flask import Flask, send_from_directory, render_template, request
import flask
import json
import ttt_game
import ttt_NN
import ttt_interface
from ttt_params import Params
    
def statevalue(statevec, X, Y):
    place = X*3 + Y
    if statevec[place] == 1:
        return "X"
    elif statevec[place + 9] == 1:
        return "O"
    else:
        return "-"

app = Flask(__name__, static_url_path='')

tf.reset_default_graph()
sess = tf.Session()

m = ttt_NN.ttt_NN(Params())

saver = tf.train.Saver()
saver.restore(sess, './network_weights')

G = ttt_game.NetworkGame(m, sess)
emptyboard = "0"*18

@app.route('/')
def hello_world():
    statevec = emptyboard
    return render_template('index.html', board=statevec)


@app.route('/submit')
def makemove():    
    statevec_string = request.args['boardvec']
    statevec = [int(i) for i in statevec_string]
    print(len(statevec)) 
    print(statevec, "<--")
    statevec = G.playmove(statevec) 
    print(statevec)
    return flask.jsonify({"statevec" : ''.join([str(int(i)) for i in
                                                list(statevec)])})
    #render_template('index.html', board=statevec)

@app.route('/reset')
def reset():
    global statevec
    statevec = emptyboard

    boardvalues = [[statevalue(statevec, X, Y) for Y in range(3)] for X in range(3)]

    print(boardvalues)
    return render_template('index.html', board=boardvalues)


""" Console commands:
export FLASK_APP=testflask.py
flask run
"""

