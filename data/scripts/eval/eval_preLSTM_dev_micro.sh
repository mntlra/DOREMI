python main.py --model_name LSTM \
--data_dir ../data/docred \
--eval_mode micro \
--train_file train_annotated.json \
--test_file dev.json \
--train_prefix dev_train \
--test_prefix  dev_dev \
--test_batch_size 40 \
--load_path ../data/checkpoints/LSTM/pretrain \
--save_path ../data/checkpoints/LSTM/pretrain \