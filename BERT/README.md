# Applied Deep Learning Homework 2
## Description
In this case, I fine tune the BERT pretrained model. 

I modified the run_classifier.py into my task.  

The training data is just like the following figure. 

![](https://github.com/leo3308/Applied-Deep-Learning/blob/master/BERT/img/train_pic.png)

This task is to predict the class of the given sentence.


### How to train my model

1. prepare train.csv and dev.csv in a folder

2. add the ``` --do_train and --do_eval ```  in shellcript

3. remove  ```--do_predict and --output_path```  in shellcript

4. run the shellscript

```
bash ./best.sh /your/data/folder/ 
```

### How to get predict by my model
1. prepare test.csv in folder

2. DO NOT MODIFIED the shellscript

2. run the following codes on the shell
```
bash ./best.sh /your/test.csv/ /your/path/to/output/file
```

