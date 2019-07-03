cd src/
python3 make_test_dataset.py ../data $1
python3 predict.py ../models/example $2 --network ExampleNet --model model.pkl.rnn
