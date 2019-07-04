from ELMo.ELMoNet import ELMoNet
import torch

class sent2elmo():
    def __init__(self, char_lexicon, config, device, model_path):
        
        self.char_lexicon = char_lexicon
        self.config = config
        self.device = device
        self.model_path = model_path
        
        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage.cuda(0))
        
        num_embeddings = len(self.char_lexicon)
        padding_idx = self.char_lexicon['<pad>']
        self.model = ELMoNet(num_embeddings,
                             self.config['embedding_dim'],
                             padding_idx,
                             self.config['filters'],
                             self.config['n_highways'],
                             self.config['projection_size'],
                             checkpoint['vocab_size'])
        
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def get_feature(self, sentences):
        
        test_data, test_data_reverse = self.create_dataset(sentences)
        
        test_data = torch.tensor(test_data)
        test_data = test_data[:,:-1,:]
        test_data_reverse = torch.tensor(test_data_reverse)
        test_data_reverse = test_data_reverse[:,:-1,:]
        
        with torch.no_grad():
            forward_feature, backward_feature = self.model.forward(test_data.to(self.device), test_data_reverse.to(self.device),None,None, False)
            forward_feature = forward_feature[:,:-1,:]
            backward_feature = backward_feature[:,:-1,:]
            feature = torch.cat((forward_feature, backward_feature), dim=2)
    
        return feature

    def create_dataset(self, sentences):
        
        max_len = 0
        max_word_len = 16
        special_word = ['<unk>', '<bos>', '<eos>', '<pad>']
        
        char2id = []
        for i, sentence in enumerate(sentences):
            """ create a list for every sentence """
            char2id.append([])
            for word in sentence:
                tmp = []
                if len(word) > max_word_len:
                    tmp.append(self.char_lexicon['<unk>'])
                elif word not in special_word:
                    if len(word) > max_len:
                        max_len = len(word)
                    for char in word:
                        if char not in self.char_lexicon:
                            tmp.append(self.char_lexicon['<unk>'])
                        else:
                            tmp.append(self.char_lexicon[char])
                else:
                    tmp.append(self.char_lexicon[word])

                char2id[i].append(tmp)
    
        max_len = min(max_word_len, max_len)

        """ padding the character of each word """
        for i, sentence in enumerate(char2id):
            for j, word in enumerate(sentence):
                if len(word) < max_len:
                    for _ in range(max_len-len(word)):
                        char2id[i][j].append(self.char_lexicon['<pad>'])

        char2id_reverse = []
        for i, sentence in enumerate(char2id):
            char2id_reverse.append([])
            for rev in reversed(sentence):
                char2id_reverse[i].append(rev)
                        
        return char2id, char2id_reverse

