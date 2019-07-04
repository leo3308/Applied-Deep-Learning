import torch
import os
import json
from ELMoNet import ELMoNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

class trainELMo():

    def __init__(self, train_dataset, train_dataset_reverse, config, char_lexicon, word_lexicon, batch_size, learning_rate, device, max_epoch, output_dir):

#        self.train_dataset = train_dataset
#        self.train_dataset_reverse = train_dataset_reverse
        self.config = config
        self.char_lexicon = char_lexicon
        self.word_lexicon = word_lexicon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.max_epoch = max_epoch
        self.output_dir = output_dir

        num_embeddings = len(self.char_lexicon)
        padding_idx = self.char_lexicon['<pad>']
        self.model = ELMoNet(num_embeddings,
                             self.config['embedding_dim'],
                             padding_idx,
                             self.config['filters'],
                             self.config['n_highways'],
                             self.config['projection_size'],
                             len(word_lexicon))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = self.learning_rate)
        
#        cutoffs = [10, 100, 1000]
#        self.adap_loss = AdaptiveLogSoftmaxWithLoss(self.config['projection_size'],
#                                                    len(self.word_lexicon),
#                                                    cutoffs)
#        self.adap_loss.to(self.device)

        self.epoch = 0
    def save_model(self):
    
        output_model_path = os.path.join(self.output_dir,
                                         "model_{}".format(self.epoch))
        torch.save({'model':self.model.state_dict(),
                   'vocab_size': len(self.word_lexicon)},output_model_path)

    def save_log(self, log):
        output_log_path = os.path.join(self.output_dir, "log.json")
        with open(output_log_path, 'w') as outfile:
            json.dump(log, outfile)

    def do_train(self, train_dataset, train_dataset_reverse, valid_dataset, valid_dataset_reverse):
        log = {}
        log['train'] = {}
        log['valid'] = {}
        while self.epoch < self.max_epoch:
            
            length = (len(train_dataset) // 10) * 9
            
            print ("training %i" % self.epoch)
            forward_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
            backward_loader = DataLoader(train_dataset_reverse, batch_size=self.batch_size, shuffle=False)
            
            log_train = self._run_epoch(forward_loader, backward_loader, True)
            
            print ("evaluating %i" % self.epoch)
            forward_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
            backward_loader = DataLoader(valid_dataset_reverse, batch_size=self.batch_size, shuffle=False)

            log_valid = self._run_epoch(forward_loader, backward_loader, False)

            log['train'].update(log_train)
            log['valid'].update(log_valid)

            self.epoch += 1
        self.save_log(log)
    
    
    def _run_epoch(self, forward_loader, backward_loader, training):
        
        self.model.train(training)
        
        if training:
            descript = 'training'
        else:
            descript = 'evaluating'
                    
        
        trange = tqdm(enumerate(zip(forward_loader, backward_loader)), total=len(forward_loader),desc=descript, ascii=True)
        
        loss = 0
        epoch_log = {}

        for i, (forward_batch, backward_batch) in trange:

            forward_batch_x, forward_batch_y = forward_batch
            backward_batch_x, backward_batch_y = backward_batch
            if training:
                
                self.optimizer.zero_grad()
                
#                forward_feature, backward_feature = \
#                    self.model.forward(forward_batch_x.to(self.device),
#                                       backward_batch_x.to(self.device))
#
#                forward_feature = forward_feature.view(-1,forward_feature.size()[2])
#                forward_batch_y = forward_batch_y.view(forward_batch_y.size()[0] * forward_batch_y.size()[1])
#                forward_output, forward_loss = \
#                    self.adap_loss(forward_feature, forward_batch_y.to(self.device))
#
#                backward_feature = backward_feature.view(-1, backward_feature.size()[2])
#                backward_batch_y = backward_batch_y.view(backward_batch_y.size()[0] * backward_batch_y.size()[1])
#                backward_output, backward_loss = \
#                    self.adap_loss(backward_feature, backward_batch_y.to(self.device))
#
#                batch_loss = (forward_loss + backward_loss)/2
#                batch_loss.backward()
#                forward_loss.backward()
#                backward_loss.backward()
                f_loss , b_loss = self.model(forward_batch_x.to(self.device),
                                             backward_batch_x.to(self.device),
                                             forward_batch_y.to(self.device),
                                             backward_batch_y.to(self.device),
                                             True)
                
                batch_loss = (f_loss + b_loss)/2
                
                batch_loss.backward()

                self.optimizer.step()
                
                loss += batch_loss.item()
            else:
                with torch.no_grad():
#                    forward_feature, backward_feature = \
#                        self.model.forward(forward_batch_x.to(self.device),
#                                      backward_batch_x.to(self.device))
#                    forward_feature = forward_feature.view(-1,forward_feature.size()[2])
#                    forward_batch_y = forward_batch_y.view(forward_batch_y.size()[0] * forward_batch_y.size()[1])
#                    forward_output, forward_loss = \
#                        self.adap_loss(forward_feature, forward_batch_y.to(self.device))
#
#                    backward_feature = backward_feature.view(-1, backward_feature.size()[2])
#                    backward_batch_y = backward_batch_y.view(backward_batch_y.size()[0] * backward_batch_y.size()[1])
#                    backward_output, backward_loss = \
#                        self.adap_loss(backward_feature, backward_batch_y.to(self.device))
                    f_loss , b_loss = self.model(forward_batch_x.to(self.device),
                                                 backward_batch_x.to(self.device),
                                                 forward_batch_y.to(self.device),
                                                 backward_batch_y.to(self.device),
                                                 True)
                    
                    batch_loss = (f_loss + b_loss)/2

                    loss += batch_loss.item()

            trange.set_postfix(loss=loss / (i+1))
                
        loss /= len(forward_loader)
        epoch_log["epoch{}".format(self.epoch)] = loss
        print ("epoch=%i" % self.epoch)
        print ("loss=%f\n" % loss)

        self.save_model()

        return epoch_log

