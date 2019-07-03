import sys
import os
import json
import nltk
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def main():
    
    data_path = os.path.join(sys.argv[1], 'valid.json')
    with open(data_path) as f:
        data = json.load(f)
    
    embedding_path = os.path.join(sys.argv[1], 'embedding.pkl')
    with open(embedding_path, 'rb') as f:
        embedding = pickle.load(f)
    
    
    contexts = data[0]['messages-so-far'][-1]['utterance']
    contexts = nltk.word_tokenize(contexts)
    options = data[0]['options-for-correct-answers'][-1]['utterance']
    options = nltk.word_tokenize(options)
    contexts_e = []
    options_e = []
    for context in contexts:
        contexts_e.append(embedding.to_index(context))
    for option in options:
        options_e.append(embedding.to_index(option))

    contexts_e = torch.tensor(contexts_e)
    options_e = torch.tensor(options_e)

    embed = torch.nn.Embedding(embedding.vectors.size(0),
                               embedding.vectors.size(1))
    contexts_e = embed(contexts_e)
    contexts_e = torch.unsqueeze(contexts_e, 0)
    options_e = embed(options_e)
    options_e = torch.unsqueeze(options_e, 0)

    lstm = torch.nn.LSTM(300, 128, batch_first=True, bidirectional=True)
    context_lstm, (h_n, c_n) = lstm(contexts_e)
    option_lstm, (h_n, c_n) = lstm(options_e)
    atten = torch.bmm(option_lstm, context_lstm.transpose(1, 2))
    atten_soft = F.softmax(atten, dim=2)
    atten_soft = torch.squeeze(atten_soft)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(atten_soft.detach().numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels(contexts, rotation=45)
    ax.set_yticklabels(options)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig("atten_visualize.png")


if __name__ == "__main__":
    nltk.download("punkt")
    main()
