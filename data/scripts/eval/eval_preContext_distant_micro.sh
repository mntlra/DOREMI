python main.py --model_name ContextAware \
--data_dir ../data/docred \
--eval_mode micro \
--train_prefix dev_train \
--test_prefix  train_distant \
--train_file train_annotated.json \
--test_file train_distant.json \
--test_batch_size 40 \
--load_path ../data/checkpoints/ContextAware/pretrain \
--save_path ../data/checkpoints/ContextAware/pretrain \