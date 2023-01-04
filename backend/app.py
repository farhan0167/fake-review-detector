from flask import Flask
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from fastai.text.all import *
import torch

app = Flask(__name__)

class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]

@app.route("/")
def hello_world():

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