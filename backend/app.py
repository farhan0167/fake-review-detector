from flask import Flask, request
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from fastai.text.all import *
import torch
import socket
from collections import defaultdict
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)
connection_table = defaultdict(int)


class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]


@app.route("/")
def welcome():
    
    return {
        "message": "Hello"
    }


@app.route("/get-connection-info")
def connecttion_info():
    server_name = socket.gethostname()
    server_ip = socket.gethostbyname(server_name)
    client_ip = request.remote_addr
    ip, port = request.environ.get('REMOTE_ADDR'), request.environ.get('REMOTE_PORT')


    if client_ip not in connection_table.keys():
        connection_table[client_ip]+=1
    else:
        if connection_table[client_ip]>10:
            #record an entry for the ip address in connection table
            #need to wait an hour before making more requests
            return {
                "message": "Too many requests"
            }
        connection_table[client_ip]+=1

    return {
        "server_name": server_name,
        "server_ip": server_ip,
        "client_ip": client_ip,
        "number_of_requests": connection_table[client_ip],
        'ip_of_client': ip,
        'port_of_client': port
    }

@app.route("/generate", methods=['GET', 'POST'])
def generate_review():
    if request.method == 'POST':
        req =  request.get_json()
        prompt = req['prompt']
    else:
        prompt = "The service was the best I've"

    tokenizer = torch.load('pre-trained/gpt2pretrained-tokenizer.pth')
    model = torch.load('pre-trained/gpt2pretrained-model.pth')
    learn = Learner(dls=None, model=model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
    learn.load("gpt2-Restaurants")

    NUM_OF_SAMPLES = 1
    prompt_ids = tokenizer.encode(prompt)
    inp = tensor(prompt_ids)[None]

    preds = learn.model.generate(inp, max_length=40, do_sample=True, top_k=0, top_p=0.92, num_return_sequences=NUM_OF_SAMPLES, temperature=0.7)

    return {
        "message": tokenizer.decode(preds[0].cpu().numpy())
    }

if __name__ == '__main__':
    app.run(host="localhost", port=8000)