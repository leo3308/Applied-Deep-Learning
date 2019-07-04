import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss
from .char_embedding import *

class ELMoNet(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx, conv_filters, n_highways, projection_size, vocab_size):
        super(ELMoNet, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.conv_filters = conv_filters
        self.n_highways = n_highways
        self.projection_size = projection_size

        self.char_embedding = CharEmbedding(self.num_embeddings,
                                            self.embedding_dim,
                                            self.padding_idx,
                                            self.conv_filters,
                                            self.n_highways,
                                            self.projection_size)
                                            
        self.hidden_size = 2048
        
        self.lstm1f = nn.LSTM(self.projection_size, self.hidden_size, 1, batch_first=True)
        self.lstm2f = nn.LSTM(self.projection_size, self.hidden_size, 1, batch_first=True)
        self.lstm1r = nn.LSTM(self.projection_size, self.hidden_size, 1, batch_first=True)
        self.lstm2r = nn.LSTM(self.projection_size, self.hidden_size, 1, batch_first=True)
        
        self.linear1f = nn.Linear(self.hidden_size, self.projection_size)
        self.linear1r = nn.Linear(self.hidden_size, self.projection_size)
        self.linear2f = nn.Linear(self.hidden_size, self.projection_size)
        self.linear2r = nn.Linear(self.hidden_size, self.projection_size)
    
        self.adap_loss = AdaptiveLogSoftmaxWithLoss(self.projection_size,
                                                    vocab_size,
                                                    [10,100,1000])

    def forward(self, train_data, train_data_back, target_x, target_y, training):
        
        if training:
#            train_data_x , target_x = train_data
#            train_data_back_x , target_y = train_data_back

            char_output = self.char_embedding(train_data)
            lstm1f_output , (h_n, c_n) = self.lstm1f(char_output)
            o1f = self.linear1f(lstm1f_output)
            lstm2f_output , (h_n, c_n) = self.lstm2f(o1f)
            o2f = self.linear2f(lstm2f_output)
            
            char_back_output = self.char_embedding(train_data_back)
            lstm1r_output , (h_n, c_n) = self.lstm1r(char_back_output)
            o1r = self.linear1r(lstm1r_output)
            lstm2r_output , (h_n, c_n) = self.lstm2r(o1r)
            o2r = self.linear2r(lstm2r_output)
            
            forward = o2f.view(-1, o2f.size()[2])
            target_x = target_x.view(target_x.size()[0] * target_x.size()[1])
            _ , forward_loss = self.adap_loss(forward, target_x)
            
            backward = o2r.view(-1, o2r.size()[2])
            target_y = target_y.view(target_y.size()[0] * target_y.size()[1])
            _ , backward_loss = self.adap_loss(backward, target_y)
            
            return forward_loss, backward_loss
                
        else:
            char_output = self.char_embedding(train_data)
            lstm1f_output , (h_n, c_n) = self.lstm1f(char_output)
            o1f = self.linear1f(lstm1f_output)
            lstm2f_output , (h_n, c_n) = self.lstm2f(o1f)
            o2f = self.linear2f(lstm2f_output)
            
            char_back_output = self.char_embedding(train_data_back)
            lstm1r_output , (h_n, c_n) = self.lstm1r(char_back_output)
            o1r = self.linear1r(lstm1r_output)
            lstm2r_output , (h_n, c_n) = self.lstm2r(o1r)
            o2r = self.linear2r(lstm2r_output)

            return o2f, o2r





