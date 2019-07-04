import os
import sys
import argparse
import logging
import torch
import json
import pickle
from tqdm import tqdm
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss
from torch.utils.data import DataLoader
from collections import Counter
from dataset import CharacterDataset
from char_embedding import *
from ELMoNet import ELMoNet
from train import trainELMo

def break_sentence(sentence, max_sent_len):
    ret = []
    cur = 0
    length = len(sentence)
    while cur < length:
        if cur + max_sent_len + 5 >= length:
            ret.append(sentence[cur: length])
            break
        ret.append(sentence[cur: min(length, cur + max_sent_len)])
        cur += max_sent_len
    return ret

def read_corpus(data_path, max_chars=50, max_sent_len=20):
    logging.info("reading corpus from {}".format(data_path))
    data = []
    data_shift = []
    with open(data_path) as f:
        for (i, line) in enumerate(f):
            if i > 1000000:
                break
            data.append('<bos>')
            for token in line.strip().split():
                if max_chars is not None and len(token) + 2 > max_chars:
                    token = token[:max_chars - 2]
                data.append(token)
            data.append('<eos>')
        dataset = break_sentence(data, max_sent_len)
    return dataset[:-1]

#def read_corpus2(data_path, max_sent_len):
#    logging.info("reading corpus from {}".format(data_path))
#    data = []
#    with open(data_path) as f:
#        for (i, line) in enumerate(f):
#            data.append([])
#            if i > 10000:
#                break
#            for token in line.strip().split():
#                data[i].append(token)
#
#        break_sentence = []
#        for sent in data:
#            length = len(sent)
#            cur = 0
#            while cur < length:
#                if cur + max_sent_len >= length:
#                    break_sentence.append(sent[cur: length])
#                    break
#                break_sentence.append(sent[cur : min(length , cur + max_sent_len)])
#                cur += max_sent_len


def get_truncated_vocab(dataset, min_count):
    logging.info("get truncated vocab...")
    word_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)
    word_count = list(word_count.items())

    frequency = 0
    for word, count in word_count:
        if count < min_count:
            frequency += 1
    word_count.append(('<unk>', frequency))

    word_count.sort(key=lambda x: x[1], reverse=True)

    i = 0
    for word, count in word_count:
        if count < min_count:
            break
        i += 1

    return word_count[:i]

def get_truncated_char(dataset, min_count):
    logging.info("get truncated character...")
    char_count = Counter()
    for sentence in dataset:
        for word in sentence:
            char_count.update(word)
    char_count = list(char_count.items())
    char_count.sort(key=lambda x:x[1], reverse=True)

    i = 0
    for char, count in char_count:
        if count < min_count:
            break
        i += 1
    return char_count[:i]


def get_lexicon(vocab, character):
    """get the lexicon of words and characters"""
    
    logging.info("creating words and characters lexicon ...")
    word_lexicon = {}
    special_word = ['<unk>', '<bos>', '<eos>']

    for word in special_word:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)

    for word, _ in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)


    char_lexicon = {}
    special_char = ['<unk>', '<bos>', '<eos>', '<pad>']

    for char in special_char:
        char_lexicon[char] = len(char_lexicon)

    for char, _ in character:
        if char not in char_lexicon:
            char_lexicon[char] = len(char_lexicon)

    return word_lexicon, char_lexicon

def word2id(train_data, word_lexicon):
    train_data_id = []
    for i, sentence in enumerate(train_data):
        train_data_id.append([])
        for word in sentence:
            if word in word_lexicon:
                train_data_id[i].append(word_lexicon[word])
            else:
                train_data_id[i].append(word_lexicon['<unk>'])

#    train_data_id_reverse = []
#    for i, sentence in enumerate(train_data_id):
#        train_data_id_reverse.append([])
#        for word in reversed(sentence):
#            train_data_id_reverse[i].append(word)

    return train_data_id

def create_dataset(train_data, word_label, char_lexicon, max_sent_len, max_word_len):

    max_len = 0
    special_word = ['<unk>', '<bos>', '<eos>', '<pad>']

    char2id = []
    for i, sentence in enumerate(train_data):
        """ create a list for every sentence """
        char2id.append([])
        for word in sentence:
            tmp = []
            if len(word) > max_word_len:
                tmp.append(char_lexicon['<unk>'])
            elif word not in special_word:
                if len(word) > max_len:
                    max_len = len(word)
                for char in word:
                    if char not in char_lexicon:
                        tmp.append(char_lexicon['<unk>'])
                    else:
                        tmp.append(char_lexicon[char])
            else:
                tmp.append(char_lexicon[word])
            char2id[i].append(tmp)

    max_len = min(max_word_len, max_len)

    """ padding the character of each word """
    for i, sentence in enumerate(char2id):
        for j, word in enumerate(sentence):
            if max_len > len(word):
                for _ in range(max_len-len(word)):
                    char2id[i][j].append(char_lexicon['<pad>'])
    
    char2id_reverse = []
    for i, sentence in enumerate(char2id):
        char2id_reverse.append([])
        for rev in reversed(sentence):
            char2id_reverse[i].append(rev)

    """ create reverse word label """
    word_label_reverse = []
    for i, sentence in enumerate(word_label):
        word_label_reverse.append([])
        for word in reversed(sentence):
            word_label_reverse[i].append(word)

    """ create target """
    target = []
    for i, sentence in enumerate(word_label):
        tmp = []
        if i == (len(word_label)-1):
            tmp = sentence[1:] + [char_lexicon['<unk>']]
        else:
            tmp = sentence[1:] + [word_label[i+1][0]]
        target.append(tmp)
            
    """ create inverse target """
    target_reverse = []
    for i, sentence in enumerate(word_label_reverse):
        tmp = []
        if i == (len(word_label_reverse)-1):
            tmp = sentence[1:] + [char_lexicon['<unk>']]
        else:
            tmp = sentence[1:] + [word_label_reverse[i+1][0]]
        target_reverse.append(tmp)

    length_char = (len(char2id) // 10) * 9
    length_word = (len(word_label) // 10) * 9

    """ set data to Dataset of PyTorch """
    return CharacterDataset(char2id[:length_char], target[:length_word]), CharacterDataset(char2id_reverse[:length_char], target_reverse[:length_word]), CharacterDataset(char2id[length_char:], target[length_word:]), CharacterDataset(char2id_reverse[length_char:], target_reverse[length_word:])

def save_model(model, epoch, output_dir):

    output_model_path = os.path.join(output_dir, "model_{}".format(epoch))
    torch.save(model.state_dict(), output_model_path)

def save_log(epoch_log, output_dir):
    output_log_path = os.path.join(output_dir, "log.json")
    with open(output_log_path, 'w') as outfile:
        json.dump(epoch_log, outfile)

def train(train_dataset, train_dataset_reverse, config_path, char_lexicon, word_lexicon, batch_size, learning_rate, device, max_epoch, output_dir):

    with open(config_path, 'r') as f:
        config = json.load(f)

    forward_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    backward_loader = DataLoader(train_dataset_reverse, batch_size=batch_size, shuffle=False)
#    word_label_loader =
    num_embeddings = len(char_lexicon)
    padding_idx = char_lexicon['<pad>']
    model = ELMoNet(num_embeddings,
                    config['embedding_dim'],
                    padding_idx,
                    config['filters'],
                    config['n_highways'],
                    config['projection_size'])
    
    model.train(True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cutoffs = [100, 1000, 10000]
    word_lexicon_size = len(word_lexicon)

    adap_loss = AdaptiveLogSoftmaxWithLoss(config['projection_size'], word_lexicon_size, cutoffs)
    adap_loss = adap_loss.to(device)
    trange = tqdm(enumerate(zip(forward_loader, backward_loader)), total=len(forward_loader),
                  desc='training', ascii=True)
    for epoch in range(max_epoch):
        
        loss = 0
        epoch_log = {}
        
        for i, ((forward_batch, forward_label), (backward_batch, backward_label)) in trange:
                
            optimizer.zero_grad()
                
            forward_feature, backward_feature = \
                model.forward(forward_batch.to(device), backward_batch.to(device))
            

            forward_feature = forward_feature.view(-1,forward_feature.size()[2])
            forward_label = forward_label.view(forward_label.size()[0] * forward_label.size()[1])
            forward_output, forward_loss = \
                adap_loss(forward_feature, forward_label.to(device))

            backward_feature = backward_feature.view(-1, backward_feature.size()[2])
            backward_label = backward_label.view(backward_label.size()[0] * backward_label.size()[1])
            backward_output, backward_loss = \
                adap_loss(backward_feature, backward_label.to(device))
        
            
            forward_loss.backward()
            backward_loss.backward()

            optimizer.step()

            loss += (forward_loss.item() + backward_loss.item())/2
            trange.set_postfix(loss=loss / (i+1))

        loss /= len(forward_loader)
        epoch_log["epoch{}".format(epoch)] = loss
        print ("epoch=%f\n" % epoch)
        print ("loss=%f\n" % loss)

        save_model(model, epoch, output_dir)
    save_log(epoch_log, output_dir)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default = None,
                        type=str,
                        required=True)
    parser.add_argument("--config_path",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--max_sent_len",
                        default=64,
                        type=int)
    parser.add_argument("--max_word_len",
                        default=16,
                        type=int)
    parser.add_argument("--min_word_count",
                        default=3,
                        type=int)
    parser.add_argument("--min_char_count",
                        default=1000,
                        type=int)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=int)
    parser.add_argument("--device",
                        default='cuda:0',
                        type=str)
    parser.add_argument("--max_epoch",
                        default=3,
                        type=int)

    args = parser.parse_args()
    
    train_data = read_corpus(args.data_path, max_sent_len=args.max_sent_len)
    
    vocab = get_truncated_vocab(train_data, args.min_word_count)
    
    character = get_truncated_char(train_data, args.min_char_count)
    
    word_lexicon, char_lexicon = get_lexicon(vocab, character)

    word_label = word2id(train_data, word_lexicon)
    
    dataset, dataset_reverse, validset, validset_reverse = create_dataset(train_data, word_label, char_lexicon, args.max_sent_len, args.max_word_len)
    
#    train(dataset, dataset_reverse, args.config_path,
#          char_lexicon, word_lexicon, args.batch_size,
#          args.learning_rate, args.device, args.max_epoch, args.output_dir)

    word_lexicon_path = os.path.join(args.output_dir, "charLexicon.pkl")
    with open(word_lexicon_path, 'wb') as outfile:
        pickle.dump(char_lexicon, outfile)

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    train_elmo = trainELMo(dataset, dataset_reverse, config,
                           char_lexicon, word_lexicon, args.batch_size,
                           args.learning_rate, args.device, args.max_epoch, args.output_dir)
    train_elmo.do_train(dataset, dataset_reverse, validset, validset_reverse)





if __name__ =='__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S')
    main()
