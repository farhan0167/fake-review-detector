from flask import Flask, request
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from fastai.text.all import *
import torch
import socket
from collections import defaultdict
import json
import netifaces

app = Flask(__name__)
connection_table = defaultdict(int)


class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]

def get_mac_address(ip_address):
    interfaces = netifaces.interfaces()
    for i in interfaces:
        if i == 'lo':
            continue
        iface = netifaces.ifaddresses(i).get(netifaces.AF_INET)
        if iface != None:
            for j in iface:
                if j['addr'] == ip_address:
                    mac_address = netifaces.ifaddresses(i).get(netifaces.AF_LINK)[0]['addr']
                    return mac_address
    return None

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
    client_mac = get_mac_address(client_ip)

    if client_ip not in connection_table.keys():
        connection_table[client_ip]+=1
    else:
        connection_table[client_ip]+=1

    return {
        "server_name": server_name,
        "server_ip": server_ip,
        "client_ip": client_ip,
        "number_of_requests": connection_table[client_ip],
        'ip_of_client': ip,
        'port_of_client': port,
        'client_mac': client_mac
    }
@app.route("/generate")
def generate_review():
    tokenizer = torch.load('pre-trained/gpt2pretrained-tokenizer.pth')
    model = torch.load('pre-trained/gpt2pretrained-model.pth')
    learn = Learner(dls=None, model=model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
    learn.load("gpt2-Restaurants")

    NUM_OF_SAMPLES = 1
    prompt = "The service was the best I've"
    prompt_ids = tokenizer.encode(prompt)
    inp = tensor(prompt_ids)[None]

    preds = learn.model.generate(inp, max_length=40, do_sample=True, top_k=0, top_p=0.92, num_return_sequences=NUM_OF_SAMPLES, temperature=0.7)

    return {
        "prompt": prompt,
        "message": tokenizer.decode(preds[0].cpu().numpy())
    }

if __name__ == '__main__':
    app.run(host="localhost", port=8000)