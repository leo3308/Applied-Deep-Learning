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

## How to plot my figure
1. make sure that embedding.pkl and valid.json in the data folder

2. run the following codes on the shell
```
cd ../src
python3 plot.py ../data

```
