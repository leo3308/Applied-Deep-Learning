# Applied Deep Learning Homework 1
## How to train my model

1. prepare train.json, valid.json, crawl-300d-2M.vec in data folder

2. preprocessor the data

```
cd ../src
python3 make_dataset.py ../data

```

3. train data
```
python3 train.py ../models/example

```

4. predict test data
```
bash ./rnn.sh /path/to/test.json /path/to/predict.csv
or 
bash ./attention.sh /path/to/test.json /path/to/predict.csv
```

## How to plot my figure
1. make sure that embedding.pkl and valid.json in the data folder

2. run the following codes on the shell
```
cd ../src
python3 plot.py ../data

```
![](https://github.com/leo3308/Applied-Deep-Learning/blob/master/rnn_with_attention/img/atten_visualize.png)

### Task Slids

https://docs.google.com/presentation/d/15LCy7TkJXl2pdz394gSPKY-fkwA3wFusv7h01lWOMSw/edit#slide=id.p
