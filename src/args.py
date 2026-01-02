import argparse

def add_args(parser):
    """
    Arguments for main.py
    """
    parser.add_argument("--tag", default="", type=str, help="Step to be performed.")
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--checkpoint_file", default="", type=str, help="Name of the file containing the checkpoint")
    parser.add_argument('--model_name', type=str, default='LSTM', help='name of the model',
                        choices=["CNN3", "LSTM", "BiLSTM", "ContextAware", "BERT"])

    parser.add_argument('--train_prefix', type=str, default='', help="Train file to use for training CNN, LSTM, BiLSTM, and ContextAware.")
    parser.add_argument('--test_prefix', type=str, default='', help="Test file to use for training CNN, LSTM, BiLSTM, and ContextAware")
    parser.add_argument('--input_theta', type=float, default=-1, help="input threshold to use when using the model in inference. Needed if the models are not trained with the pipeline.")

    parser.add_argument("--train_file", default="train_annotated.json", type=str, help="train file")
    parser.add_argument("--dev_file", default="dev.json", type=str, help="dev file")
    parser.add_argument("--test_file", default="", type=str, help="test file for evaluation")
    parser.add_argument("--pred_file", default="results.json", type=str)
    parser.add_argument("--save_path", default="", type=str, help="Path to save the checkpoints and the results.")
    parser.add_argument("--load_path", default="", type=str, help="Path to load the checkpoints.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--eval_mode", default="micro", type=str,
                        choices=["micro", "per-relation", "macro", "preds"],
                        help="Whether to perform micro-averaged evaluation (micro), macro-averaged evaluation (macro), or report the performance for each relation (per-relation).")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_sent_num", default=25, type=int,
                        help="Max number of sentences in each document.")
    parser.add_argument("--lr_transformer", default=5e-5, type=float,
                        help="The initial learning rate for transformer.")
    parser.add_argument("--lr_added", default=1e-4, type=float,
                        help="The initial learning rate for added modules.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--isDistant", action="store_true",
                        help="Whether we are considering the distant dataset for inference.")
    parser.add_argument("--distant_prefix", type=str, default="",
                        help="The name of the distant dataset for inference.")


    # for sampling/inference
    parser.add_argument('--sc', type=str, default="logsum_agreement",
                        choices=["entropy", "agreement", "test_agreement", "logsum", "mean", "logsum_agreement"],
                        help="Selection Criterion")
    parser.add_argument('--pred_dir', type=str, default="../data/checkpoints", help="Path to prediction files")
    parser.add_argument("--pred_mode", type=str, default="pretrain_half",
                        help="Directory containing the prediction files")
    parser.add_argument("--filename", type=str, default="dev_sample", help="Name of prediction files")

    parser.add_argument('--save_sc_matrix', action="store_true", help="Whether to save the sc matrix.")
    parser.add_argument('--num_chunks', type=int, default=100, help="Number of chunks for inference on the distant dataset.")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples")
    parser.add_argument('--new_sample_file', type=str, default="dev_sample_iter1.json", help="Name of the sample file to save.")
    parser.add_argument('--new_train_file', type=str, default="iteration1.json", help="New training dataset")
    parser.add_argument('--exclude_pairs', type=bool, default=True, help="Whether to exclude pairs not annotated.")
    parser.add_argument('--only_annotated', action="store_true",
                        help="Whether to select only pairs that are annotated in the filename")


    parser.add_argument('--long_300', action="store_true",
                        help="Whether to consider as long_tail relations with less than 300 examples")
    parser.add_argument("--rel2id_longtail", default="rel2id_longtail.json", type=str, help="rel2id_longtail file")
    parser.add_argument("--relations_frequency", default="relations_frequency.json", type=str)

    return parser