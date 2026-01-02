python main.py --model_name BiLSTM \
--data_dir ../data/docred \
--eval_mode micro \
--train_file train_annotated.json \
--test_file train_distant.json \
--train_prefix dev_train \
--test_prefix  train_distant \
--test_batch_size 40 \
--load_path ../data/checkpoints/BiLSTM/pretrain \
--save_path ../data/checkpoints/BiLSTM/pretrain \