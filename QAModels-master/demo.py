#!/usr/bin/env python3
# coding=utf-8
import argparse
import threading
import traceback
from time import sleep
from predictor import Predictor
import bottle
import socket
import time
import torch

'''
This file is taken and modified from R-Net by Minsangkim142
https://github.com/minsangkim142/R-net
'''

app = bottle.Bottle()
query = []
response = ""


@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html


@app.post('/answer')
def answer():
    passage = bottle.request.json['passage']
    question = bottle.request.json['question']
    print("received question: {}".format(question))
    # if not passage or not question:
    #     exit()
    global query, response
    query = (passage, question)
    while not response:
        sleep(0.1)
    print("received response: {}".format(response))
    response_ = {"answer": response}
    response = []
    return response_


class Demo(object):
    def __init__(self, model, config):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args=[model, config, run_event]).start()
        ip = socket.gethostbyname(socket.gethostname())
        app.run(port=config.port, host=config.ip or ip)
        try:
            while True:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    @staticmethod
    def demo_backend(model, config, run_event):
        global query, response
        while run_event.is_set():
            sleep(0.1)
            try:
                if query:
                    t0 = time.time()
                    document, question = query
                    predictions = model.predict(document, question, top_n=config.top_n)
                    results = []
                    for i, p in enumerate(predictions, 1):
                        result_str = '{}, {}, {:4.2f}<br>'.format(i, p[0], p[1])
                        results.append(result_str)
                    response = ''.join(results) + 'Prediction took {:.4f} s'.format(time.time() - t0)
                    print(response)
                    query = []
            except Exception as e:
                print(e)
                tb = traceback.format_exc()
                print(tb)
                response = 'ERROR! Stack trace: <br> {}'.format(tb)
                query = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,default="data/models/m_reader.mdl", help='data/models/m_reader.mdl')
    parser.add_argument('--ip', type=str, default=None, help='ip to serve.')
    parser.add_argument('--port', type=str, default=6061, help='port to serve')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Specify GPU device id to use')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Do not softmax normalize output scores.')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Specify how many predictions to return')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        print('CUDA enabled (GPU %d)' % args.gpu)
    else:
        print('Running on CPU only.')

    predictor = Predictor(args.model, normalize=not args.no_normalize)
    #predictor = Predictor("data/models/m_reader.mdl", normalize=not args.no_normalize)
    if args.cuda:
        predictor.cuda()
    Demo(predictor, args)
