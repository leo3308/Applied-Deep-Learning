#!/bin/bash
#Program:
#	This is a program for downloading some models of ADL_hw1
#2019-04-02

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
export PATH
dirBase=predict_base_cased
dirLarge=predict_large_cased
mkdir $dirBase
wget -O $dirBase/config.json https://www.dropbox.com/s/8dmyn3nmk6zbr8q/config.json?dl=0
wget -O $dirBase/pytorch_model.bin https://www.dropbox.com/s/b3ne6crtmvm4we5/pytorch_model.bin?dl=0
wget -O $dirBase/vocab.txt https://www.dropbox.com/s/al7edepbauvvkxj/vocab.txt?dl=0
mkdir $dirLarge
wget -O $dirLarge/config.json https://www.dropbox.com/s/g37o59g0seng0wp/config.json?dl=0
wget -O $dirLarge/pytorch_model.bin https://www.dropbox.com/s/l2iuxnlaqkgywhl/pytorch_model.bin?dl=0
wget -O $dirLarge/vocab.txt https://www.dropbox.com/s/3sjqf8zndyb6aud/vocab.txt?dl=0
exit 0
