python main.py --model_name LSTM \
--data_dir ../data/redocred \
--eval_mode micro \
--train_file train_revised.json \
--test_file train_distant.json \
--train_prefix train_revised \
--test_prefix  train_distant \
--test_batch_size 40 \
--load_path ../data/checkpoints/LSTM/pretrain_redocred \
--save_path ../data/checkpoints/LSTM/pretrain_redocred \