import argparse
import os
import gc
import datetime
import models
from config import Config
import numpy as np
import torch
import random
import ujson as json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from models import BERT
from Logger import Logger
from args import add_args
from utils.utils import collate_fn, create_directory, dump_to_file
from prepro import read_docred_for_BERT
from evaluation import to_official, official_evaluate
from tqdm import tqdm

import pandas as pd
import pickle
from train import train_bert, evaluate, evaluate_per_rel

from sample import compute_disagreement, sample2official


def main():
    
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    # Setting seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    args.n_gpu = torch.cuda.device_count()

    logger_name = str(datetime.datetime.now()).replace(' ', '_')

    if args.tag == "infer" and args.isDistant:

        my_logger = Logger(f"../data/output/logs/sampling/{logger_name}.log")
        args.logger = my_logger

        if args.device != "":
            my_logger.logger.info(f"Using argument device: {args.device}")
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        else:
            # my_logger.logger.info(f"Number of devices available: {torch.cuda.device_count()}")
            my_logger.logger.info(f"Cuda current device: {torch.cuda.current_device()}")
            device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        my_logger.logger.info(f"Model will be loaded to device: {device}")
        args.device = device

        args.results_path = f"../data/results/{args.pred_mode}/"

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        # Lists to compute mean agreement
        means, lens, prods = [], [], []
        # Inference Score for distant dataset -- proceed in chunks
        ms = ["BERT", "CNN3", "LSTM", "BiLSTM", "ContextAware"]
        for s in range(1, args.num_chunks+1):
            test_prefix = f"{args.distant_prefix}_sample{s}"
            my_logger.logger.info(f"+++++ Considering sample ({s}/{args.num_chunks}) {test_prefix} +++++")
            args.test_prefix = test_prefix
            args.test_file = f"{args.distant_prefix}_sample{s}.json"
            for m in ms:
                args.model_name = m
                args.load_path = f"{args.pred_dir}/{m}/{args.pred_mode}/"
                args.save_path = args.load_path
                my_logger.logger.info(f"*** Starting inference for model {args.model_name} ***")
                if args.model_name != "BERT":
                    args.test_batch_size = 40
                    model = {
                        'CNN3': models.CNN3,
                        'LSTM': models.LSTM,
                        'BiLSTM': models.BiLSTM,
                        'ContextAware': models.ContextAware,
                    }
                    # for s in range(1, 5):
                    con = Config(args)
                    my_logger.logger.info(
                        f"--- Starting execution of infer scores of {args.model_name} in candidates"
                        f" {test_prefix} ---")
                    con.load_test_data()
                    con.infer_scores(model[args.model_name])
                else:
                    args.test_batch_size = 8
                    args.transformer_type = "bert"
                    args.model_name_or_path = "bert-base-cased"
                    # if s == 1:
                    config_bert = AutoConfig.from_pretrained(
                        args.config_name if args.config_name else args.model_name_or_path,
                        num_labels=args.num_class,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    )
                    model_bert = AutoModel.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=config_bert,
                    )

                    config_bert.transformer_type = args.transformer_type

                    read_bert = read_docred_for_BERT
                    config_bert.cls_token_id = tokenizer.cls_token_id
                    config_bert.sep_token_id = tokenizer.sep_token_id

                    model_bert = BERT(my_logger, config_bert, model_bert, tokenizer, args.device)
                    model_bert.to(args.device)

                    # Loading best model
                    model_path = os.path.join(args.load_path, "best.ckpt")
                    my_logger.logger.info(f"Loading model from {model_path} ...")
                    model_bert.load_state_dict(torch.load(model_path, map_location=args.device))

                    my_logger.logger.info(
                        f"--- Starting execution of infer scores of {args.model_name} for candidates {args.test_file}")
                    # Inference
                    test_file = os.path.join(args.data_dir, args.test_file)

                    my_logger.logger.info(f"[BERT] Test batch size is {args.test_batch_size} ...")

                    test_features = read_bert(args, my_logger, args.data_dir, test_file,
                                         tokenizer,
                                         transformer_type=args.transformer_type,
                                         max_seq_length=args.max_seq_length)

                    official_results = evaluate(args, model_bert, test_features, tag="infer")

                    offi_path = os.path.join(args.load_path, f"{args.distant_prefix}_sample_results.json")
                    my_logger.logger.info(f"Saving candidates predictions into {offi_path} ...")
                    json.dump(official_results, open(offi_path, "w"))

                    # Deleting the model
                    del config_bert
                    del tokenizer
                    del model_bert
                    gc.collect()


            # Compute selection criteria score
            my_logger.logger.info(f"Computing disagreement between models based on {args.sc}")
            args.score_file = f"{args.distant_prefix}_sample_results.json"
            args.filename = test_prefix
            sample_df = compute_disagreement(args, my_logger, f"{test_prefix}_title2pairs.json")
            sample_df_path = os.path.join(args.results_path, f"{test_prefix}_{args.sc}.csv")
            my_logger.logger.info(f"Saving disagreement between models based on {sample_df_path}")
            sample_df.to_csv(sample_df_path, index=False)
            mean_sample = sample_df['logsum_agreement'].mean()
            my_logger.logger.info(f"[MEAN] Mean logsum agreement of {test_prefix}: {mean_sample} (size: {len(sample_df)})")
            lens.append(len(sample_df))
            means.append(mean_sample)
            prods.append(len(sample_df)*mean_sample)
            # break
        my_logger.logger.info(f"Combining disagreement of {args.num_chunks} samples")
        combined_df = pd.concat([pd.read_csv(os.path.join(args.results_path, f"{args.distant_prefix}_sample{i}_{args.sc}.csv")).head(10000) for i in range(1, args.num_chunks+1)], ignore_index=True)
        args.filename = f"{args.distant_prefix}"
        args.only_annotated = False
        # Sorting combined DataFrame
        combined_df = combined_df.sort_values(by='logsum_agreement', ascending=False)
        # Computing mean logsum-agreement
        mean_agreement = sum(prods)/sum(lens) if len(lens) > 0 else 0
        my_logger.logger.info(f"[MEAN] Mean logsum agreement: {mean_agreement}")
        if len(combined_df) > 10000:
            combined_df.head(10000).to_csv(os.path.join(args.results_path, f"{args.new_sample_file}_{args.sc}_COMBINED.csv"), index=False)
            _, sample = sample2official(combined_df.head(10000), args, my_logger)
            sample_df_path = os.path.join(args.results_path, f"{args.new_sample_file}_{args.sc}.csv")
            my_logger.logger.info(f"Saving disagreement between models based on {sample_df_path}")
            combined_df.head(10000).to_csv(sample_df_path, index=False)
        else:
            combined_df.to_csv(os.path.join(args.results_path, f"{args.new_sample_file}_{args.sc}_COMBINED.csv"),
                                           index=False)
            _, sample = sample2official(combined_df, args, my_logger)
            sample_df_path = os.path.join(args.results_path, f"{args.new_sample_file}_{args.sc}.csv")
            my_logger.logger.info(f"Saving disagreement between models based on {sample_df_path}")
            combined_df.to_csv(sample_df_path, index=False)
        sample_path = f"{args.results_path}/{args.new_sample_file}"
        my_logger.logger.info(f"Saving sample at {sample_path} ..")
        json.dump(sample, open(sample_path, "w"))
        if "test_revised" in args.distant_prefix or "dev_sample" in args.distant_prefix:
            my_logger.logger.info(f"Saving new training dataset at {args.data_dir}/{args.new_train_file} ..")
            json.dump(_, open(f"{args.data_dir}/{args.new_train_file}", "w"))
    else:

        my_logger = Logger(f"../data/output/logs/{args.model_name}/{logger_name}.log")
        args.logger = my_logger
        if args.save_path == "":
            args.save_path = f"../data/checkpoints/{args.model_name}/"

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if args.device != "":
            my_logger.logger.info(f"Using argument device: {args.device}")
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        else:
            # my_logger.logger.info(f"Number of devices available: {torch.cuda.device_count()}")
            my_logger.logger.info(f"Cuda current device: {torch.cuda.current_device()}")
            device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        my_logger.logger.info(f"Model will be loaded to device: {device}")
        args.device = device

        if args.model_name != "BERT":
            model = {
                'CNN3': models.CNN3,
                'LSTM': models.LSTM,
                'BiLSTM': models.BiLSTM,
                'ContextAware': models.ContextAware,
            }
            con = Config(args)
            if args.tag == "train":
                if args.load_path != "":
                    my_logger.logger.info(f"--- Starting execution of finetuning of {args.model_name} in dataset"
                                          f" {args.train_prefix} ---")
                else:
                    if args.checkpoint_file == "":
                        my_logger.logger.info(f"--- Starting execution of training of {args.model_name} in dataset"
                                              f" {args.train_prefix} ---")
                    else:
                        my_logger.logger.info(f"--- Resume training of {args.model_name} in dataset"
                                              f" {args.train_prefix} ---")
                con.load_train_data()
                con.load_test_data()
                con.train_config(model[args.model_name], args.model_name)
            elif args.tag == "infer":
                my_logger.logger.info(f"--- Starting execution of infer scores of {args.model_name} in candidates"
                                      f" {args.test_prefix} ---")
                con.load_test_data()
                con.infer_scores(model[args.model_name])
            else:
                # Evaluation
                basename = os.path.splitext(args.test_file)[0]
                my_logger.logger.info(f"--- Starting execution of {args.eval_mode} evaluation of {args.model_name}"
                                      f" for dataset {args.test_file} ---")
                # Load threshold
                thresh_file = os.path.join(args.load_path, "thresh.csv")
                thresh = pd.read_csv(thresh_file)
                threshold = thresh.loc[thresh["epoch"] == "best"]["threshold"].values[0]
                my_logger.logger.info(f"Using probability threshold: {threshold}")

                if args.eval_mode == "per-relation":
                    # per-relation Evaluation
                    con.load_test_data()
                    output_dict = con.testall(model[args.model_name], args.input_theta, tag="test", thresh=threshold)

                    score_path = os.path.join(args.load_path, f"{basename}_relationwise_scores.csv")
                    headers = ["precision", "recall", "F1", "prec_ign", "rec_ign", "F1_ign", "prec_evi", "rec_evi", "F1_evi"]
                    scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
                    my_logger.logger.info(f"Saving scores into {score_path} ...")
                    scores_pd.to_csv(score_path, sep='\t')

                elif args.eval_mode == "macro":
                    # macro-avg Evaluation
                    con.load_test_data()
                    output_dict = con.testall(model[args.model_name], args.input_theta, tag="test", thresh=threshold)

                    if args.long_300:
                        score_path = os.path.join(args.load_path, f"{basename}_macro_scores_300.csv")
                    else:
                        score_path = os.path.join(args.load_path, f"{basename}_macro_scores.csv")
                    headers = ["F1", "ignF1", "precision", "ignPrec", "recall"]
                    scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
                    my_logger.logger.info(f"Saving scores into {score_path} ...")
                    scores_pd.to_csv(score_path, sep='\t')
                else:
                    # micro-avg Evaluation
                    con.load_test_data()
                    """if args.test_prefix == "dev_dev_eval":
                        scores = con.testall(model[args.model_name], args.input_theta, save_thresh=True)
                    else:"""
                    scores = con.testall(model[args.model_name], args.input_theta, tag="test", thresh=threshold)
                    if args.eval_mode == "micro":
                        score_path = os.path.join(args.save_path, f"{basename}_micro_scores.csv")
                        headers = ["precision", "recall", "F1"]
                        scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns=headers)
                        my_logger.logger.info(scores_pd)
                        my_logger.logger.info(f"Saving scores into {score_path} ...")
                        scores_pd.to_csv(score_path, sep='\t')

        else:

            config_bert = AutoConfig.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                num_labels=args.num_class,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            )
            model = AutoModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config_bert,
            )

            config_bert.transformer_type = args.transformer_type

            read_bert = read_docred_for_BERT
            config_bert.cls_token_id = tokenizer.cls_token_id
            config_bert.sep_token_id = tokenizer.sep_token_id

            model = BERT(my_logger, config_bert, model, tokenizer, args.device)
            model.to(args.device)

            mode = "train"
            if args.load_path != "":  # load model from existing checkpoint
                if args.checkpoint_file == "":
                    model_path = os.path.join(args.load_path, "best.ckpt")
                    mode = "finetune"
                else:
                    model_path = os.path.join(args.load_path, args.checkpoint_file)
                    mode = "resume"
                my_logger.logger.info(f"Loading model from {model_path} ...")
                model.load_state_dict(torch.load(model_path, map_location=args.device))

            if args.tag == "train":
                if mode == "train":
                    my_logger.logger.info(f"--- Starting execution of training of {args.model_name} in dataset"
                                          f" {args.train_prefix} ---")
                elif mode == "finetune":
                    my_logger.logger.info(f"--- Starting execution of finetuning of {args.model_name} in dataset"
                                              f" {args.train_prefix} ---")
                else:
                    my_logger.logger.info(f"--- Resuming training of {args.model_name} in dataset"
                                          f" {args.train_prefix} ---")
                # Training
                train_file = os.path.join(args.data_dir, args.train_file)
                dev_file = os.path.join(args.data_dir, args.dev_file)

                train_features = read_bert(args, my_logger, args.data_dir, train_file, tokenizer, transformer_type=args.transformer_type)
                dev_features = read_bert(args, my_logger, args.data_dir, dev_file, tokenizer, transformer_type=args.transformer_type)

                train_bert(args, model, train_features, dev_features)
            elif args.tag == "infer":
                my_logger.logger.info(
                    f"--- Starting execution of infer scores of {args.model_name} for candidates {args.test_file}")
                # Inference
                basename = os.path.splitext(args.test_file)[0].split("/")[-1]
                test_file = os.path.join(args.data_dir, args.test_file)

                test_features = read_bert(args, my_logger, args.data_dir, test_file, tokenizer, transformer_type=args.transformer_type,
                                     max_seq_length=args.max_seq_length)

                official_results = evaluate(args, model, test_features, tag="infer")

                offi_path = os.path.join(args.load_path, f"{basename}_results.json")
                my_logger.logger.info(f"Saving candidates predictions into {offi_path} ...")
                json.dump(official_results, open(offi_path, "w"))

            else:
                my_logger.logger.info(
                    f"--- Starting execution of {args.eval_mode} evaluation of {args.model_name} for dataset {args.test_file}")
                # Evaluation
                basename = os.path.splitext(args.test_file)[0]
                test_file = os.path.join(args.data_dir, args.test_file)

                # Load threshold
                thresh_file = os.path.join(args.load_path, "thresh.csv")
                thresh = pd.read_csv(thresh_file)
                threshold = thresh.loc[thresh["epoch"] == "best"]["threshold"].values[0]
                my_logger.logger.info(f"Using probability threshold: {threshold}")
                test_features = read_bert(args, my_logger, args.data_dir, test_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length)

                if args.eval_mode == "per-relation":
                    # per-relation Evaluation
                    output_dict = evaluate_per_rel(args, model, test_features, tag="test", threshold=threshold)

                    score_path = os.path.join(args.load_path, f"{basename}_relationwise_scores.csv")
                    my_logger.logger.info(f"saving evaluations into {score_path} ...")
                    headers = ["precision", "recall", "F1", "prec_ign", "rec_ign", "F1_ign", "prec_evi", "rec_evi", "F1_evi"]
                    scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
                    scores_pd.to_csv(score_path, sep='\t')

                elif args.eval_mode == "macro":
                    # macro-avg Evaluation
                    output_dict = evaluate_per_rel(args, model, test_features, tag="test", threshold=threshold)

                    if args.long_300:
                        score_path = os.path.join(args.load_path, f"{basename}_macro_scores_300.csv")
                    else:
                        score_path = os.path.join(args.load_path, f"{basename}_macro_scores.csv")

                    my_logger.logger.info(f"saving evaluations into {score_path} ...")
                    headers = ["F1", "ignF1", "precision", "ignPrec", "recall"]
                    scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
                    scores_pd.to_csv(score_path, sep='\t')
                else:
                    # micro-avg Evaluation
                    test_scores, test_output, official_results, _ = evaluate(args, model, test_features, tag="test", threshold=threshold)

                    offi_path = os.path.join(args.load_path, f"{basename}_results.json")
                    if args.eval_mode == "preds":
                        my_logger.logger.info(f"Saving official predictions into {offi_path} ...")
                        json.dump(official_results, open(offi_path, "w"))
                    else:
                        score_path = os.path.join(args.load_path, f"{basename}_micro_scores.csv")

                        dump_to_file(my_logger, official_results, offi_path, test_output, score_path)


if __name__ == "__main__":
    main()
