python main.py --model_name ContextAware \
--data_dir ../data/redocred \
--eval_mode micro \
--train_prefix train_revised \
--test_prefix  train_distant \
--train_file train_revised.json \
--test_file train_distant.json \
--test_batch_size 40 \
--load_path ../data/checkpoints/ContextAware/pretrain_redocred \
--save_path ../data/checkpoints/ContextAware/pretrain_redocred \