# Applied Deep Learning Homework 2
## part 1
### How to train my ELMo

1. prepare corpus.txt in ../data

2. 

```
change the "from .char_embedding import *" in ELMoNet
to "from char_embedding import *"
```

3. change the config in train.sh
```
--data_path /path/to/corpus.txt/
--output_dir /path/to/output/dir/
```

4. run the shellscript

```
bash ./train.sh 
```

