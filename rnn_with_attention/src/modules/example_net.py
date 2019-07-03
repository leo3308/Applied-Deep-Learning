import torch

class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,similarity='inner_product'):
        super(ExampleNet, self).__init__()
        
        self.hidden_size = 256
    
        self.lstm = torch.nn.LSTM(dim_embeddings, self.hidden_size, batch_first=True, bidirectional=True)
        self.Linear = torch.nn.Linear(self.hidden_size*2, self.hidden_size*2)

    def forward(self, context, context_lens, options, option_lens):

        output1, (h_n, c_n) = self.lstm(context)
        context_lstm = output1.max(1)[0] #max pooling of context
        context_lstm = self.Linear(context_lstm)
        context_lstm = torch.unsqueeze(context_lstm, 1)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            output2, (h_n, c_n) = self.lstm(option)
            option_lstm = output2.max(1)[0]
            option_lstm = torch.unsqueeze(option_lstm, 2)
            logit = torch.bmm(context_lstm, option_lstm)
            logit = torch.squeeze(logit)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits
