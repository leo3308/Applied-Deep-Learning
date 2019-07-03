#!/bin/bash
#Program:
#	This is a program for downloading some models of ADL_hw1
#2019-04-02

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
dirEmbedding=data/embedding.pkl
dirRnnModel=models/example/model.pkl.rnn
dirAttnModel=models/example/model.pkl.rnnAtten
wget -O ${dirEmbedding} https://www.dropbox.com/s/sly78ta4ok0rgc1/embedding.pkl?dl=0
wget -O ${dirRnnModel} https://www.dropbox.com/s/zetp18fq239kp84/model.pkl.9.96?dl=0
wget -O ${dirAttnModel} https://www.dropbox.com/s/z1kyvg1ap58halp/model.pkl.rnn_atten?dl=0
exit 0
