import torch
import torch.nn.functional as F
class AttenNet(torch.nn.Module):
    """
        
        Args:
        
        """
    
    def __init__(self, dim_embeddings,similarity='inner_product'):
        super(AttenNet, self).__init__()
        
        self.hidden_size = 256
        
        self.lstm1 = torch.nn.LSTM(dim_embeddings, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)
        self.Linear_Q = torch.nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.Linear_A = torch.nn.Linear(dim_embeddings, self.hidden_size*2)
    
    def forward(self, context, context_lens, options, option_lens):
        
        '''context feed in lstm and pass a linear'''
        context_lstm, (h_n, c_n) = self.lstm1(context)
        context_max = context_lstm.max(1)[0] #max pooling of context
        context_max_linear = self.Linear_Q(context_max)
        context_max_linear = torch.unsqueeze(context_max_linear, 1)#add a dimension of context_lstm
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_lstm, (h_n, c_n) = self.lstm1(option)
            atten = torch.bmm(option_lstm, context_lstm.transpose(1, 2))
            atten_soft = F.softmax(atten, dim=2)
            mix = torch.bmm(atten_soft, context_lstm)
            cat1 = option_lstm
            cat2 = mix
            cat3 = option_lstm * mix
            cat4 = option_lstm - mix
            concate = torch.cat((cat1, cat2, cat3, cat4), dim=2)

            option_lstm2, (h_n, c_n) = self.lstm2(concate)
            option_lstm2 = option_lstm2.max(1)[0]
            option_lstm2 = torch.unsqueeze(option_lstm2, 2)
            logit = torch.bmm(context_max_linear, option_lstm2)
            logit = torch.squeeze(logit)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

