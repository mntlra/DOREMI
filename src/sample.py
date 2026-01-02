import argparse
import os
import datetime
import numpy as np
import ujson as json
from Logger import Logger
from prepro import read_docred_for_BERT
from evaluation import to_official, official_evaluate
from tqdm import tqdm
from scipy.special import softmax

import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--sc', type=str, default="entropy", choices=["entropy", "agreement", "test_agreement", "logsum", "mean", "logsum_agreement"], help="Selection Criterion")
parser.add_argument('--save_sc_matrix', type=bool, default=False, help="Whether to save the sc matrix.")
parser.add_argument('--data_dir', type=str, default="../data/docred", help="Path to data files")
parser.add_argument('--pred_dir', type=str, default="../data/checkpoints", help="Path to prediction files")
parser.add_argument("--pred_mode", type=str, default="pretrain_half", help="Directory containing the prediction files")
parser.add_argument("--filename", type=str, default="dev_sample", help="Name of prediction files")
parser.add_argument("--save_name", type=str, default="candidates_iter1", help="Name of candidates file for saving")
parser.add_argument("--candidates_file", type=str, default="", help="Name of file containing the candidates.")
parser.add_argument("--score_file", type=str, default="", help="Name of file containing the candidates probabilities.")
parser.add_argument('--num_samples', type=int, default=100, help="Number of samples")
parser.add_argument('--train_file', type=str, default="train_annotated.json", help="Training dataset")
parser.add_argument('--new_train_file', type=str, default="iteration1.json", help="New training dataset")
parser.add_argument('--new_sample_file', type=str, default="dev_sample_iter1.json", help="New training dataset")
parser.add_argument('--gt_path', type=str, default="", help="Path to ground truth labels")
parser.add_argument('--only_annotated', type=bool, default=False, help="Whether to select only pairs that are annotated in the filename")
parser.add_argument("--rel2id_longtail", default="rel2id_longtail.json", type=str, help="rel2id_longtail file")

def select_candidates(args, logger):
    """
    Select documents where at least one model predicted a long-tail relations in a triple.

    :param args: arguments.
    :param logger: Logger.

    :returns: list of candidates.
    """
    # Setting up global variables
    rel2id_longtail = json.load(open(f"{args.data_dir}/meta/{args.rel2id_longtail}", "r"))
    logger.logger.info(f"Selecting candidates with long-tail triples from {args.data_dir}/meta/{args.rel2id_longtail}")
    models = ["CNN3", "LSTM", "BiLSTM", "ContextAware", "BERT"]
    num_documents = 0
    title2candidates_longtail, title2pairs_longtail, title2pairs, title2candidates = {}, {}, {}, {}
    total_preds = 0
    title2modelpreds = {}
    for model in models:
        preds = json.load(open(f"{args.pred_dir}/{model}/{args.pred_mode}/{args.filename}_results.json"))
        logger.logger.info(f"Parsing {model} {args.filename} prediction file ({len(preds)} predictions)")
        total_preds += len(preds)
        for p in preds:
            if p["r"] in rel2id_longtail.keys():
                if p["title"] not in title2candidates.keys():
                    num_documents += 1
                    title2candidates[p["title"]] = 1
                    title2pairs[p["title"]] = [(p["h_idx"], p["t_idx"])]
                else:
                    # document already selected
                    if (p["h_idx"], p["t_idx"]) in title2pairs[p["title"]]:
                        # triple already selected
                        continue
                    else:
                        title2pairs[p["title"]].append((p["h_idx"], p["t_idx"]))

    data_path = f"{args.data_dir}/{args.filename}.json"
    logger.logger.info(f"Loading dataset information from {data_path} --")
    data_file = json.load(open(data_path))
    candidates = []

    for doc in data_file:
        if doc["title"] in title2candidates.keys():
            candidates.append(doc)

    logger.logger.info(f"Selected {num_documents} documents over {total_preds} predictions")
    return candidates, title2pairs

def add_triple_to_train(logger, filename, title2train, title2sample, h_sid, t_sid, document_title, h_idx, t_idx, annotated=False):

    label = {}
    # if "dev_sample" in filename:
    for l in title2sample[document_title]["labels"]:
        if l["h"] == h_idx and l["t"] == t_idx:
            if l["h"] == h_idx and l["t"] == t_idx:
                label = {"h": h_idx, "t": t_idx, "r": l["r"], "evidence": []}
                break
    if len(label) == 0 and annotated:
        # Skip negative example
        return title2train, 0

    new_hid, new_tid = 999, 999
    for ix, v in enumerate(title2train[document_title]["vertexSet"]):
        if v[0]["ent_id"] == h_sid:
            new_hid = ix
        if v[0]["ent_id"] == t_sid:
            new_tid = ix
    if new_hid != 999 and new_tid != 999:
        # Entities already included in the training dataset, no need to re-insert them
        logger.logger.info(f"[ALREADY CONSIDERED] Entities {h_idx} and {t_idx} of document {document_title} already in training")
        logger.logger.info(f"Checking whether pair ({h_idx}, {t_idx}) has already been considered")
        if "include_pairs" in title2train[document_title].keys():
            if [new_hid, new_tid] in title2train[document_title]["include_pairs"]:
                logger.logger.info(f"[OLD PAIR] ({h_idx}, {t_idx}) pair of document {document_title} already considered in the sample")
                return title2train, 0
            else:
                logger.logger.info(f"[NEW PAIR] Adding ({h_idx}, {t_idx}) pair of document {document_title} to consider in the training")
                title2train[document_title]["include_pairs"].append((new_hid, new_tid))
                # Adding label, if any
                if len(label) > 0:
                    # adding only if it is not a negative example
                    label["h"] = new_hid
                    label["t"] = new_tid
                    if title2train[document_title]["labels"] == []:
                        # previous example was a negative example
                        title2train[document_title]["labels"] = [label]
                    else:
                        title2train[document_title]["labels"].append(label)
                return title2train, 1
        else:
            logger.logger.info(
                f"[include_pairs] include_pairs not in document {document_title}")
            # include_pairs is not in the training dataset, meaning no pair has been considered yet
            title2train[document_title]["include_pairs"] = [(new_hid, new_tid)]
            # Adding label, if any
            if len(label) > 0:
                # adding only if it is not a negative example
                label["h"] = new_hid
                label["t"] = new_tid
                if title2train[document_title]["labels"] == []:
                    # previous example was a negative example
                    title2train[document_title]["labels"] = [label]
                else:
                    title2train[document_title]["labels"].append(label)
            return title2train, 1
    if new_hid == 999:
        new_hid = len(title2train[document_title]["vertexSet"])
        title2train[document_title]["vertexSet"].append(title2sample[document_title]["vertexSet"][h_idx])
        """if new_tid == 999:
            new_tid = len(title2train[document_title]["vertexSet"]) + 1
            title2train[document_title]["vertexSet"].append(
                title2dev[document_title]["vertexSet"][t_idx])"""
    if new_tid == 999:
        new_tid = len(title2train[document_title]["vertexSet"])
        title2train[document_title]["vertexSet"].append(title2sample[document_title]["vertexSet"][t_idx])
    title2train[document_title]["old2new"][h_idx] = new_hid
    title2train[document_title]["old2new"][t_idx] = new_tid

    # Adding pairs to be considered
    logger.logger.info(f"[include_pairs] Adding pair ({h_idx}, {t_idx}) of document {document_title} "
                       f"to be considered during training")
    if "include_pairs" in title2train[document_title].keys():
        title2train[document_title]["include_pairs"].append((new_hid, new_tid))
    else:
        title2train[document_title]["include_pairs"] = [(new_hid, new_tid)]

    if len(label) > 0:
        # adding only if it is not a negative example
        label["h"] = new_hid
        label["t"] = new_tid
        if title2train[document_title]["labels"] == []:
            # previous example was a negative example
            title2train[document_title]["labels"] = [label]
        else:
            title2train[document_title]["labels"].append(label)

    return title2train, 1


def add_document_to_train(logger, title2train, title2sample, document_title, h_idx, t_idx, annotated=False):
    label = {}
    for l in title2sample[document_title]["labels"]:
        if l["h"] == h_idx and l["t"] == t_idx:
            label = {"h": 0, "t": 1, "r": l["r"], "evidence": []}
            break
    if len(label) == 0 and annotated:
        # no labels in the sample, while sample should be annotated (annotated=True)
        logger.logger.info(f"No labels in the sample for the pair, while sample should be annotated "
                           f"(annotated=True) -- returning title2train, 0")
        return title2train, 0
    title2train[document_title] = {"title": document_title, "sents": title2sample[document_title]["sents"]}
    vertexSet = [title2sample[document_title]["vertexSet"][h_idx], title2sample[document_title]["vertexSet"][t_idx]]
    title2train[document_title]["vertexSet"] = vertexSet
    title2train[document_title]["old2new"] = {h_idx: 0, t_idx: 1}
    title2train[document_title]["include_pairs"] = [(0, 1)]
    if len(label) > 0:
        logger.logger.info(f"Label found in the sample, treating the pair as a positive example")
        title2train[document_title]["labels"] = [label]
    elif not annotated:
        logger.logger.info(f"Annotated is set to False, treating the pair as a negative example")
        # negative example
        title2train[document_title]["labels"] = []
    else:
        raise ValueError(f"only_annotated set to True but trying to add a negative example")

    return title2train, 1

def add_triple_to_official_sample(logger, title2sample, title2dev, document_title, h_sid, t_sid, h_idx, t_idx, annotated=False):
    label = {}
    for l in title2dev[document_title]["labels"]:
        if l["h"] == h_idx and l["t"] == t_idx:
            if l["h"] == h_idx and l["t"] == t_idx:
                label = {"h": h_idx, "t": t_idx, "r": l["r"], "evidence": []}
                break
    if len(label) == 0 and annotated:
        return title2sample
    if document_title in title2sample.keys():
        logger.logger.info(f"Adding triple for document {document_title} already in title2sample")
        # Adding triple to the already sampled document
        new_hid, new_tid = 999, 999
        for ix, v in enumerate(title2sample[document_title]["vertexSet"]):
            if v[0]["ent_id"] == h_sid:
                new_hid = ix
            if v[0]["ent_id"] == t_sid:
                new_tid = ix
        if new_hid != 999 and new_tid != 999:
            logger.logger.info(
                f"[ALREADY CONSIDERED] Entities {h_idx} and {t_idx} of document {document_title} already in sample")
            logger.logger.info(f"Checking whether pair ({h_idx}, {t_idx}) has already been considered")
            # Entities are already in the document, check if the pair has not been considered yet
            if "include_pairs" in title2sample[document_title].keys():
                if [new_hid, new_tid] in title2sample[document_title]["include_pairs"]:
                    logger.logger.info(
                        f"[OLD PAIR] ({h_idx}, {t_idx}) pair of document {document_title} already considered in the sample")
                    return title2sample
                else:
                    logger.logger.info(
                        f"[NEW PAIR] Adding ({h_idx}, {t_idx}) pair of document {document_title} to consider in the training")
                    title2sample[document_title]["include_pairs"].append((new_hid, new_tid))
                    # Adding labels, if any
                    if len(label) > 0:
                        label["h"] = new_hid
                        label["t"] = new_tid
                        if title2sample[document_title]["labels"] == []:
                            # previous example was a negative example
                            title2sample[document_title]["labels"] = [label]
                        else:
                            title2sample[document_title]["labels"].append(label)
                    return title2sample
            else:
                logger.logger.info(
                    f"[include_pairs] include_pairs not in document {document_title}")
                title2sample[document_title]["include_pairs"] = [(new_hid, new_tid)]
                # Adding labels, if any
                if len(label) > 0:
                    label["h"] = new_hid
                    label["t"] = new_tid
                    if title2sample[document_title]["labels"] == []:
                        # previous example was a negative example
                        title2sample[document_title]["labels"] = [label]
                    else:
                        title2sample[document_title]["labels"].append(label)
                return title2sample
        if new_hid == 999:
            new_hid = len(title2sample[document_title]["vertexSet"])
            title2sample[document_title]["vertexSet"].append(title2dev[document_title]["vertexSet"][h_idx])
            """if new_tid == 999:
                new_tid = len(title2sample[document_title]["vertexSet"]) + 1
                title2sample[document_title]["vertexSet"].append(
                    title2dev[document_title]["vertexSet"][t_idx])"""
        if new_tid == 999:
            new_tid = len(title2sample[document_title]["vertexSet"])
            title2sample[document_title]["vertexSet"].append(title2dev[document_title]["vertexSet"][t_idx])
        title2sample[document_title]["old2new"][h_idx] = new_hid
        title2sample[document_title]["old2new"][t_idx] = new_tid

        if "include_pairs" in title2sample[document_title].keys():
            title2sample[document_title]["include_pairs"].append((new_hid, new_tid))
        else:
            title2sample[document_title]["include_pairs"] = [(new_hid, new_tid)]

        if len(label) > 0:
            label["h"] = new_hid
            label["t"] = new_tid
            if title2sample[document_title]["labels"]==[]:
                # previous example was a negative example
                title2sample[document_title]["labels"] = [label]
            else:
                title2sample[document_title]["labels"].append(label)
        # if it is a negative example, do not add a empty list
    else:
        logger.logger.info(f"Adding {document_title} to title2sample")
        # Add document to sample
        title2sample[document_title] = {"title": document_title, "sents": title2dev[document_title]["sents"]}
        vertexSet = [title2dev[document_title]["vertexSet"][h_idx],
                     title2dev[document_title]["vertexSet"][t_idx]]
        title2sample[document_title]["vertexSet"] = vertexSet
        title2sample[document_title]["old2new"] = {h_idx: 0, t_idx: 1}
        title2sample[document_title]["include_pairs"] = [(0, 1)]
        if len(label) > 0:
            label["h"] = 0
            label["t"] = 1
            title2sample[document_title]["labels"] = [label]
        elif not annotated:
            # negative example
            title2sample[document_title]["labels"] = []
        else:
            raise ValueError(f"only_annotated set to True but trying to add a negative example")
    return title2sample


def sample2official(df, args, logger):
    """
    Format the sample triples like DocRED datasets.

    :param df: dataframe of triples, ordered by disagreement.
    :param args: arguments.
    :param logger: Logger.

    :return: List of candidates.
    """
    num_sampled = 0
    train, sample, dev = [], [], []
    logger.logger.info(f"Preparing train file {args.train_file} and data file {args.filename}.json files..")
    data = json.load(open(f"{args.data_dir}/{args.train_file}", "r"))
    title2train = {}
    for d in data:
        # Add entity identifier
        for i in range(len(d["vertexSet"])):
            if "ent_id" not in d["vertexSet"][i][0]:
                for e in d["vertexSet"][i]:
                    e["ent_id"] = i
        title2train[d["title"]] = d

    data = json.load(open(f"{args.data_dir}/{args.filename}.json", "r"))
    title2data = {}
    for d in data:
        # Add entity identifier
        for i in range(len(d["vertexSet"])):
            if "ent_id" not in d["vertexSet"][i][0]:
                for e in d["vertexSet"][i]:
                    e["ent_id"] = i

        title2data[d["title"]] = d

    logger.logger.info(f"Sampling triples..")
    # Transform sample_df into a dictionary
    title2sample = {}
    for ix, col in df.iterrows():
        if num_sampled == args.num_samples:
            logger.logger.info(f"Early stop: Sampled {num_sampled} triples (max n. of triples {args.num_samples})")
            return [title2train[title] for title in title2train.keys()], [title2sample[title] for title in title2sample.keys()]
        # Original entities identifier
        h_sid = title2data[col["title"]]["vertexSet"][col["h_idx"]][0]["ent_id"]
        t_sid = title2data[col["title"]]["vertexSet"][col["t_idx"]][0]["ent_id"]
        doc_title = col["title"]
        logger.logger.info(f"Considering document {doc_title}, pair: ({h_sid}, {t_sid})")
        if col["title"] in title2train.keys():
            logger.logger.info(f"Document {doc_title} already in the training dataset")
            # Document already in train, check if pair has already been sampled
            already_sampled = False
            for l in title2train[col["title"]]["labels"]:
                l_h, l_t = l["h"], l["t"]
                # Current entity identifier
                h_lid = title2train[col["title"]]["vertexSet"][l["h"]][0]["ent_id"]
                t_lid = title2train[col["title"]]["vertexSet"][l["t"]][0]["ent_id"]
                if (h_lid, t_lid) == (h_sid, t_sid):
                    already_sampled = True
            if already_sampled:
                logger.logger.info(f"--- Pair already in training dataset, skipping it ---")
                # Triple already sampled in previous iteration, keep document as-is
                train.append(title2train[col["title"]])
            else:
                logger.logger.info(f"Adding pair ({h_sid}, {t_sid}) to the training dataset")
                title2train, sampled = add_triple_to_train(logger, args.filename, title2train, title2data, h_sid, t_sid, doc_title, col["h_idx"],
                                                  col["t_idx"], annotated=args.only_annotated)
                if sampled > 0:
                    logger.logger.info(f"[SAMPLED] Sampled is greater than 0, adding the triple to the sample")
                    title2sample = add_triple_to_official_sample(logger, title2sample, title2data, doc_title, h_sid, t_sid, col["h_idx"],
                                                        col["t_idx"], annotated=args.only_annotated)
                else:
                    logger.logger.info(f"[SAMPLED] Sampled is 0, not adding the triple to the sample")
                logger.logger.info(f"[ADD TRIPLE TO SAMPLE] Added {sampled} triples, updating num_sampled (currently: {num_sampled})")
                num_sampled += sampled
                logger.logger.info(f"[ADD TRIPLE TO SAMPLE] updated num_sampled: {num_sampled}")
        else:
            logger.logger.info(f"Document {doc_title} not in train, add the document")
            # Document not in train, Add the document
            title2train, sampled = add_document_to_train(logger, title2train, title2data, doc_title, col["h_idx"], col["t_idx"],
                                                annotated=args.only_annotated)
            if sampled > 0:
                logger.logger.info(f"[SAMPLED] Sampled is greater than 0, adding the triple to the sample")
                title2sample = add_triple_to_official_sample(logger, title2sample, title2data, doc_title, h_sid, t_sid, col["h_idx"],
                                                col["t_idx"], annotated=args.only_annotated)
            else:
                logger.logger.info(f"[SAMPLED] Sampled is 0, not adding the triple to the sample")
            logger.logger.info(f"[ADD TRIPLE TO SAMPLE] Added {sampled} triples, updating num_sampled (currently: {num_sampled})")
            num_sampled += sampled
            logger.logger.info(f"[ADD TRIPLE TO SAMPLE] updated num_sampled: {num_sampled}")

    # End loop over DataFrame -- sampled fewer triples than args.num_samples
    logger.logger.info(f"Sampled {num_sampled} triples (max n. of triples {args.num_samples})")
    return [title2train[title] for title in title2train.keys()], [title2sample[title] for title in title2sample.keys()]

def compute_row_dis_one_logsum(row):
    """
    Compute the disagreement between models.
    """
    models = ["CNN3", "LSTM", "BiLSTM", "ContextAware", "BERT"]
    one_probs = [-np.log(row[model]) for model in models]

    return np.sum(one_probs) 

def compute_row_dis(row):
    """
    Compute the disagreement between models.
    """
    models = ["CNN3", "LSTM", "BiLSTM", "ContextAware", "BERT"]
    one_probs = [row[model] for model in models]
    zero_probs = [1-row[model] for model in models]

    return np.prod(one_probs)+np.prod(zero_probs)

def compute_row_dis_logsum(row):
    """
    Compute the disagreement between models.
    """
    models = ["CNN3", "LSTM", "BiLSTM", "ContextAware", "BERT"]
    one_probs = [row[model] for model in models]
    zero_probs = [1-row[model] for model in models]

    return -np.log(np.prod(one_probs)+np.prod(zero_probs))

def compute_row_dis_one(row):
    """
    Compute the disagreement between models.
    """
    models = ["CNN3", "LSTM", "BiLSTM", "ContextAware", "BERT"]
    one_probs = [row[model] for model in models]

    return np.prod(one_probs)

def applysoftmax(row):
    models = ["BERT", "CNN3", "LSTM", "BiLSTM", "ContextAware"]

    tmp = 1
    return softmax(np.array(list(row[models].values)))

def compute_disagreement(args, logger, candidates_file):
    """
    Select new training examples where at least one model predicted a long-tail relations.

    :param args: arguments.
    :param logger: Logger.

    :returns: list of candidates.
    """

    logger.logger.info(f"Loading titles2pairs from {args.data_dir}/{candidates_file} ..")
    titles2pairs = json.load(open(f"{args.data_dir}/{candidates_file}"))
    titles2pairs = {t: [(x[0], x[1]) for x in titles2pairs[t]] for t in titles2pairs.keys()}
    models = ["CNN3", "BERT", "LSTM", "BiLSTM", "ContextAware"]
    tmp_dict = {}
    consider_pair = {}
    for model in models:
        logger.logger.info(f"Loading preds from {args.pred_dir}/{model}/{args.pred_mode}/{args.score_file}")
        preds = json.load(open(f"{args.pred_dir}/{model}/{args.pred_mode}/{args.score_file}"))
        logger.logger.info(f"Parsing {model} {args.filename} prediction file ({len(preds)} predictions)")
        for p in preds:
            # Initialize consider_pair for the document
            if p["title"] not in consider_pair.keys():
                consider_pair[p["title"]] = {}
            # If pair has not been considered yet
            if (p["h_idx"], p["t_idx"]) not in consider_pair[p["title"]].keys():
                p_title, p_h, p_t = p["title"], p["h_idx"], p["t_idx"]
                if (p["h_idx"], p["t_idx"]) in titles2pairs[p["title"]]:
                    logger.logger.info(f"*** PAIR TO BE CONSIDERED from document {p_title}, pair: ({p_h}, {p_t}) ***")
                    consider_pair[p["title"]][(p["h_idx"], p["t_idx"])] = True
                else:
                    consider_pair[p["title"]][(p["h_idx"], p["t_idx"])] = False

            if consider_pair[p["title"]][(p["h_idx"], p["t_idx"])] is True:
                if (p["h_idx"], p["t_idx"], p["r"], p["title"]) in tmp_dict.keys():
                    tmp_dict[(p["h_idx"], p["t_idx"], p["r"], p["title"])][model] = p["score"]
                else:
                    tmp_dict[(p["h_idx"], p["t_idx"], p["r"], p["title"])] = {
                        "h_idx": p["h_idx"],
                        "t_idx": p["t_idx"],
                        "r": p["r"],
                        "title": p["title"],
                        model: p["score"]
                    }

    df = pd.DataFrame(list(tmp_dict.values()))
    logger.logger.info(f"DataFrame information: {df.shape}")
    if args.sc == "entropy":
        tqdm.pandas()
        tqdm.pandas(desc="Normalizing the scores")
        # Variables normalization
        modified_subsets = df[models].progress_apply(lambda x: softmax(np.array(list(x.values))), axis=1)
        df[models] = pd.DataFrame(modified_subsets.tolist())
        logger.logger.info(f"Scores are normalized, computing the entropy..")
        df["entropy"] = -np.sum([df[model]*np.log2(df[model]) for model in models])
        logger.logger.info(f"Entropy computed!")
        df = df.groupby(['h_idx', 't_idx', "title"])['entropy'].mean().reset_index()
        df = df.sort_values(by='entropy', ascending=False)
        return df
    elif args.sc == "test_agreement":
        tqdm.pandas()
        tqdm.pandas(desc="Computing Disagreement")
        df["agreement_new"] = df.progress_apply(compute_row_dis_one, axis=1)
        df["agreement_prev"] = df.progress_apply(compute_row_dis, axis=1)
        df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}.csv"), index=False)
        return df
    elif args.sc == "agreement":
        tqdm.pandas()
        tqdm.pandas(desc="Computing Disagreement")
        df["agreement"] = df.progress_apply(compute_row_dis, axis=1)
        if args.save_sc_matrix:
            logger.logger.info(f"Saving matrix at {args.results_path}/{args.pred_mode}_{args.sc}.csv")
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}.csv"), index=False)
        df = df.groupby(['h_idx', 't_idx', "title"])['agreement'].prod().reset_index()
        df = df.sort_values(by='agreement', ascending=True)
        if args.save_sc_matrix:
            logger.logger.info(f"Saving matrix at {args.results_path}/{args.pred_mode}_{args.sc}_sorted.csv")
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}_sorted.csv"), index=False)
        return df
    elif args.sc == "logsum_agreement":
        tqdm.pandas()
        tqdm.pandas(desc="Computing Disagreement")
        df["logsum_agreement"] = df.progress_apply(compute_row_dis_logsum, axis=1)
        if args.save_sc_matrix:
            logger.logger.info(f"Saving matrix at {args.results_path}/{args.pred_mode}_{args.sc}.csv")
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.test_prefix}_{args.sc}.csv"), index=False)
        df = df.groupby(['h_idx', 't_idx', "title"])['logsum_agreement'].sum().reset_index()
        df = df.sort_values(by='logsum_agreement', ascending=False)
        if args.save_sc_matrix:
            logger.logger.info(f"Saving matrix at {args.results_path}/{args.pred_mode}_{args.test_prefix}_{args.sc}_sorted.csv")
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}_sorted.csv"), index=False)
        return df
    elif args.sc == "logsum":
        tqdm.pandas()
        tqdm.pandas(desc="Computing Disagreement (logsum)")
        df["PPC_logsum"] = df.progress_apply(compute_row_dis_one_logsum, axis=1)
        if args.save_sc_matrix:
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}.csv"), index=False)
        df = df.groupby(['h_idx', 't_idx', "title"])['PPC_logsum'].sum().reset_index()
        df = df.sort_values(by='PPC_logsum', ascending=False)
        if args.save_sc_matrix:
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}_sorted.csv"), index=False)
        return df
    else:
        tqdm.pandas()
        tqdm.pandas(desc="Computing Disagreement (mean)")
        df["PPC_mean"] = df.progress_apply(compute_row_dis_one, axis=1)
        if args.save_sc_matrix:
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}.csv"), index=False)
        df = df.groupby(['h_idx', 't_idx', "title"])['PPC_mean'].mean().reset_index()
        df = df.sort_values(by='PPC_mean', ascending=True)
        if args.save_sc_matrix:
            df.to_csv(os.path.join(args.results_path, f"{args.pred_mode}_{args.sc}_sorted.csv"), index=False)
        return df

def main():
    args = parser.parse_args()
    logger_name = str(datetime.datetime.now()).replace(' ', '_')
    my_logger = Logger(f"../data/output/logs/sampling/{logger_name}.log")
    if args.score_file != "":
        args.results_path = f"../data/results/{args.pred_mode}/"
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        my_logger.logger.info(f"Results path is: {args.results_path}")
        # Candidate already selected
        my_logger.logger.info(f"Computing disagreement between models based on {args.sc}")
        sample_df = compute_disagreement(args, my_logger, args.candidates_file)
        if args.sc != "test_agreement":
            train, sample = sample2official(sample_df, args, my_logger)
            my_logger.logger.info(f"Saving new training dataset at {args.data_dir}/{args.new_train_file} ..")
            json.dump(train, open(f"{args.data_dir}/{args.new_train_file}", "w"))
            sample_path = f"{args.results_path}/{args.new_sample_file}"
            my_logger.logger.info(f"Saving sample at {sample_path} ..")
            json.dump(sample, open(sample_path, "w"))
    else:
        # Candidate Selection
        my_logger.logger.info(f"Selecting candidates from {args.filename}..")
        candidates, titles2pairs = select_candidates(args, my_logger)
        candidates_path = f"{args.data_dir}/{args.filename}_{args.save_name}.json"
        my_logger.logger.info(f"Saving candidates at {candidates_path} ..")
        json.dump(candidates, open(candidates_path, "w"))
        pairs_path = f"{args.data_dir}/{args.filename}_{args.save_name}_title2pairs.json"
        my_logger.logger.info(f"Saving titles2pairs at {candidates_path} ..")
        json.dump(titles2pairs, open(pairs_path, "w"))

if __name__ == "__main__":
    main()