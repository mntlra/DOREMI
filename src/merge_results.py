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
parser.add_argument('--data_dir', type=str, default="../data/docred", help="Path to data files")
parser.add_argument('--pred_dir', type=str, default="../data/checkpoints", help="Path to prediction files")
parser.add_argument("--pred_mode", type=str, default="pretrain_half", help="Directory containing the prediction files")
parser.add_argument("--best_model_file", type=str, default="", help="File containing the checkpoint to consider for each model.")
parser.add_argument("--filename", type=str, default="train_distant_results.json", help="Name of prediction files")
parser.add_argument("--save_name", type=str, default="train_distant_pretrain.json", help="Name of file to save")
parser.add_argument("--train_file", type=str, default="train_distant.json", help="Name of train file")
parser.add_argument("--distant_data", type=str, default="prediction", help="", choices=["prediction", "UGDRE"])
parser.add_argument("--annotated_file", type=str, default="train_annotated.json", help="Training file with manual annotations.")
parser.add_argument("--aggregation", type=str, default="vanilla", help="Label aggregation criteria.",
                    choices=["vanilla", "majority", "score", "score70", "vanilla70", "majority70"])

def main():
    args = parser.parse_args()
    logger_name = str(datetime.datetime.now()).replace(' ', '_')
    my_logger = Logger(f"../data/output/logs/merging/{logger_name}.log")
    my_logger.logger.info(f"Starting label aggregation (mode: {args.aggregation}) for filename {args.filename}")
    models = ["BERT", "CNN3", "LSTM", "BiLSTM", "ContextAware"]
    # checkpoint to consider
    model2best = {}
    if args.best_model_file != "":
        model2best = json.load(open(f"{args.data_dir}/meta/{args.best_model_file}"))
    else:
        for m in models:
            model2best[m] = args.pred_mode

    # manual annotations
    title2manual = {}
    for d in json.load(open(f"{args.data_dir}/{args.annotated_file}", "r")):
        title2manual[d["title"]] = d
        title2manual[d["title"]]["id2pos"] = {}
        title2manual[d["title"]]["pos2id"] = {}
        for ix, e in enumerate(d["vertexSet"]):
            if "ent_id" in e[0]:
                title2manual[d["title"]]["id2pos"][e[0]["ent_id"]] = ix
                title2manual[d["title"]]["pos2id"][ix] = e[0]["ent_id"]

    if args.distant_data == "UGDRE":
        my_logger.logger.info(f"Using UGDRE Distant Data ..")
        ugdre_json = json.load(open(os.path.join(args.data_dir, "O-DDS.json")))
        ugdre = {}
        for dataset in ugdre_json:
            ugdre[dataset["title"]] = {}
            for lbl in dataset["labels"]:
                if (lbl["h"], lbl["t"]) in ugdre[dataset["title"]].keys():
                    ugdre[dataset["title"]][(lbl["h"], lbl["t"])].append(lbl["r"])
                else:
                    ugdre[dataset["title"]][(lbl["h"], lbl["t"])] = [lbl["r"]]
        rels_freq = json.load(open(os.path.join(args.data_dir, "meta/relations_frequency.json")))
    merged_dict = {}
    for model in models:
        preds = json.load(open(f"{args.pred_dir}/{model}/{model2best[model]}/{args.filename}"))
        my_logger.logger.info(f"Parsing {model} {args.filename} prediction file ({len(preds)} predictions)")
        """f1_file = pd.read_csv(f"{args.pred_dir}/{model}/{args.pred_mode}/dev_eval_micro_scores.csv",
                              sep="\t")
        my_logger.logger.info(f1_file["Unnamed: 0"].values)
        my_logger.logger.info("test_rel" in f1_file["Unnamed: 0"].values)
        if "test_rel" in f1_file["Unnamed: 0"].values:
            f1 = (f1_file.loc[f1_file["Unnamed: 0"] == "test_rel"]["F1"].values[0])/100
        else:
            f1 = (f1_file.loc[f1_file["Unnamed: 0"] == "dev_rel"]["F1"].values[0])/100"""
        my_logger.logger.info(f"Using aggregation score: {args.aggregation}")
        for p in preds:
            score = 1 if args.aggregation in ["vanilla", "vanilla70", "majority"] else p["score"]
            if args.aggregation not in ["score70", "vanilla70", "majority70"] or p["score"] > 0.7:
                # my_logger.logger.info(f"Aggregation: {args.aggregation}; Score: {score}")
                if p["title"] in merged_dict.keys():
                    if (p["h_idx"], p["t_idx"]) in merged_dict[p["title"]].keys():
                        try:
                            merged_dict[p["title"]][(p["h_idx"], p["t_idx"])][p["r"]] += score
                        except KeyError:
                            merged_dict[p["title"]][(p["h_idx"], p["t_idx"])][p["r"]] = score
                    else:
                        merged_dict[p["title"]][(p["h_idx"], p["t_idx"])] = {p["r"]: score}
                else:
                    merged_dict[p["title"]] = {(p["h_idx"], p["t_idx"]): {p["r"]: score}}
    train_file = json.load(open(f"{args.data_dir}/{args.train_file}"))
    my_logger.logger.info(f"Aggregating labels for {args.train_file} documents (n. documents {len(train_file)})")
    new_data = []
    for t in train_file:
        tmp = {"vertexSet": t["vertexSet"], "title": t["title"], "sents": t["sents"], "labels": []}
        for ix, ent in enumerate(t["vertexSet"]):
            if "ent_id" not in ent[0]:
                t["vertexSet"][ix][0]["ent_id"] = ix

        if t["title"] in title2manual.keys():
            id2pos = title2manual[t["title"]]["id2pos"]
        else:
            # my_logger.logger.info(f"{t['title']}  not in title2manual")
            id2pos = {}
        if t["title"] in merged_dict.keys():
            if args.distant_data == "UGDRE":
                negative_pairs = []
                for lbl in t["labels"]:
                    if (lbl["h"], lbl["t"]) not in merged_dict[t["title"]].keys():
                        negative_pairs.append((lbl["h"], lbl["t"]))
                # Positive example in models predictions
                for ht_pair in merged_dict[t["title"]].keys():
                    h_id = tmp["vertexSet"][ht_pair[0]][0]["ent_id"]
                    t_id = tmp["vertexSet"][ht_pair[1]][0]["ent_id"]
                    if h_id in id2pos.keys() and t_id in id2pos.keys() and [id2pos[h_id], id2pos[t_id]] in \
                            title2manual[t["title"]]["include_pairs"]:
                        # Adding manual annotations
                        selected_rels = []
                        for lbl in title2manual[t["title"]]["labels"]:
                            if title2manual[t["title"]]["pos2id"][lbl["h"]] == h_id and title2manual[t["title"]]["pos2id"][lbl["t"]] == t_id:
                                selected_rels.append(lbl["r"])
                        for r in selected_rels:
                            tmp["labels"].append({"h": ht_pair[0], "t": ht_pair[1], "r": r, "evidence": [], "source": "manual"})
                    else:
                        # best_score = 0.0
                        selected_rels = []
                        for r in merged_dict[t["title"]][ht_pair].keys():
                            if args.aggregation in ["majority", "majority70"]:
                                if merged_dict[t["title"]][ht_pair][r] > 3:
                                    selected_rels.append(r)
                            else:
                                selected_rels.append(r)
                        if any(r_sel in rels_freq["100"] for r_sel in selected_rels):
                            for r_sel in selected_rels:
                                tmp["labels"].append({"h": ht_pair[0], "t": ht_pair[1], "r": r_sel, "evidence": [], "source": "distant"})
                        else:
                            try:
                                rels = ugdre[t["title"]][(ht_pair[0], ht_pair[1])]
                                my_logger.logger.info(f"Predicted relation is not a long tail, using UGDRE relation: {rels}")
                                for rel_ugdre in rels:
                                    if rel_ugdre not in rels_freq["100"]:
                                        tmp["labels"].append({"h": ht_pair[0], "t": ht_pair[1], "r": rel_ugdre, "evidence": [], "source": "UGDRE"})
                            except KeyError:
                                title = t["title"]
                                my_logger.logger.info(f"Negative Example in UGDRE for pair: {(ht_pair[0], ht_pair[1])},"
                                                      f" Title: {title}")
                # Negative examples
                for ht_pair in negative_pairs:
                    h_id = tmp["vertexSet"][ht_pair[0]][0]["ent_id"]
                    t_id = tmp["vertexSet"][ht_pair[1]][0]["ent_id"]
                    if h_id in id2pos.keys() and t_id in id2pos.keys() and [id2pos[h_id], id2pos[t_id]] in \
                            title2manual[t["title"]]["include_pairs"]:
                        # Inserting manual annotations
                        selected_rels = []
                        for lbl in title2manual[t["title"]]["labels"]:
                            if title2manual[t["title"]]["pos2id"][lbl["h"]] == h_id and title2manual[t["title"]]["pos2id"][lbl["t"]] == t_id:
                                selected_rels.append(lbl["r"])
                        for r in selected_rels:
                            tmp["labels"].append(
                                {"h": ht_pair[0], "t": ht_pair[1], "r": r, "evidence": [], "source": "manual"})
                    else:
                        try:
                            rels = ugdre[t["title"]][(ht_pair[0], ht_pair[1])]
                            my_logger.logger.info(f"Predicted relation is not a long tail, using UGDRE relation: {rels}")
                            for rel_ugdre in rels:
                                if rel_ugdre not in rels_freq["100"]:
                                    tmp["labels"].append({"h": ht_pair[0], "t": ht_pair[1], "r": rel_ugdre, "evidence": [], "source": "UGDRE"})
                        except KeyError:
                            title = t["title"]
                            my_logger.logger.info(f"Negative Example in UGDRE for pair: {(ht_pair[0], ht_pair[1])},"
                                                  f" Title: {title}")
            else:
                if t["title"] in merged_dict.keys():
                    negative_pairs = []
                    for lbl in t["labels"]:
                        if (lbl["h"], lbl["t"]) not in merged_dict[t["title"]].keys():
                            negative_pairs.append((lbl["h"], lbl["t"]))
                    for ht_pair in merged_dict[t["title"]].keys():
                        """my_logger.logger.info(f"Considering pair: {ht_pair}")
                        my_logger.logger.info(f"Parsing: {t['title']}")
                        my_logger.logger.info(f"merged dict: {merged_dict[t['title']]}")
                        my_logger.logger.info(f"dict: {merged_dict[t['title']][ht_pair]}")"""
                        h_id = tmp["vertexSet"][ht_pair[0]][0]["ent_id"]
                        t_id = tmp["vertexSet"][ht_pair[1]][0]["ent_id"]
                        if h_id in id2pos.keys() and t_id in id2pos.keys() and [id2pos[h_id], id2pos[t_id]] in \
                                title2manual[t["title"]]["include_pairs"]:
                            my_logger.logger.info(
                                f"[MANUAL] Adding manual annotation for pair {ht_pair} of document {t['title']}")
                            # Inserting manual annotations
                            selected_rels = []
                            for lbl in title2manual[t["title"]]["labels"]:
                                if title2manual[t["title"]]["pos2id"][lbl["h"]] == h_id and title2manual[t["title"]]["pos2id"][lbl["t"]] == t_id:
                                    selected_rels.append(lbl["r"])
                            if len(selected_rels) == 0:
                                my_logger.logger.info(
                                    f"[MANUAL RELS] No relation for pair {ht_pair} of document {t['title']}")
                            for r in selected_rels:
                                tmp["labels"].append(
                                    {"h": ht_pair[0], "t": ht_pair[1], "r": r, "evidence": [], "source": "manual"})
                        else:
                            # best_score = 0.0
                            selected_rels = []
                            """# my_logger.logger.info(merged_dict)
                            my_logger.logger.info(t["title"])
                            my_logger.logger.info(ht_pair)
                            my_logger.logger.info(merged_dict[t["title"]])
                            my_logger.logger.info(merged_dict[t["title"]].keys())
                            my_logger.logger.info(f"{ht_pair}: {type(ht_pair)}")
                            my_logger.logger.info(merged_dict[t["title"]][ht_pair])"""
                            for r in merged_dict[t["title"]][ht_pair].keys():
                                if args.aggregation in ["majority", "majority70"]:
                                    if merged_dict[t["title"]][ht_pair][r] > 3:
                                        selected_rels.append(r)
                                else:
                                    selected_rels.append(r)
                            for r_sel in selected_rels:
                                tmp["labels"].append({"h": ht_pair[0], "t": ht_pair[1], "r": r_sel, "evidence": [], "source": "distant"})

                    # Negative examples
                    for ht_pair in negative_pairs:
                        h_id = tmp["vertexSet"][ht_pair[0]][0]["ent_id"]
                        t_id = tmp["vertexSet"][ht_pair[1]][0]["ent_id"]
                        if h_id in id2pos.keys() and t_id in id2pos.keys() and [id2pos[h_id], id2pos[t_id]] in \
                                title2manual[t["title"]]["include_pairs"]:
                            my_logger.logger.info(
                                f"[MANUAL] Adding manual annotation for pair {ht_pair} of document {t['title']}")
                            # Inserting manual annotations
                            selected_rels = []
                            for lbl in title2manual[t["title"]]["labels"]:
                                if title2manual[t["title"]]["pos2id"][lbl["h"]] == h_id and title2manual[t["title"]]["pos2id"][lbl["t"]] == t_id:
                                    selected_rels.append(lbl["r"])
                            if len(selected_rels) == 0:
                                my_logger.logger.info(f"[MANUAL RELS] No relations for pair {ht_pair} of document {t['title']}")
                            for r in selected_rels:
                                tmp["labels"].append(
                                    {"h": ht_pair[0], "t": ht_pair[1], "r": r, "evidence": [],
                                     "source": "manual"})
        else:
            if args.distant_data == "UGDRE":
                my_logger.logger.info(f"Title {t['title']} not in merged_dict, Checking manual annotations and UGDRE for labels")
            else:
                my_logger.logger.info(f"Title {t['title']} not in merged_dict, Checking manual annotations for labels")
            for ht_pair in [(a, b) for a, _ in enumerate(t["vertexSet"]) for b, _ in enumerate(t["vertexSet"]) if a != b]:
                h_id = tmp["vertexSet"][ht_pair[0]][0]["ent_id"]
                t_id = tmp["vertexSet"][ht_pair[1]][0]["ent_id"]
                if h_id in id2pos.keys() and t_id in id2pos.keys() and [id2pos[h_id], id2pos[t_id]] in \
                        title2manual[t["title"]]["include_pairs"]:
                    my_logger.logger.info(
                        f"[MANUAL] Adding manual annotation for pair {ht_pair} of document {t['title']}")
                    # Inserting manual annotations
                    selected_rels = []
                    for lbl in title2manual[t["title"]]["labels"]:
                        if title2manual[t["title"]]["pos2id"][lbl["h"]] == h_id and title2manual[t["title"]]["pos2id"][lbl["t"]] == t_id:
                            selected_rels.append(lbl["r"])
                    if len(selected_rels) == 0:
                        my_logger.logger.info(f"[MANUAL RELS] No relations for pait {ht_pair} of document {t['title']}")
                    for r in selected_rels:
                        tmp["labels"].append(
                            {"h": ht_pair[0], "t": ht_pair[1], "r": r, "evidence": [], "source": "manual"})
                elif args.distant_data == "UGDRE":
                    try:
                        rels = ugdre[t["title"]][(ht_pair[0], ht_pair[1])]
                        my_logger.logger.info(
                            f"Predicted relation is not a long tail, using UGDRE relation: {rels}")
                        for rel_ugdre in rels:
                            if rel_ugdre not in rels_freq["100"]:
                                tmp["labels"].append(
                                    {"h": ht_pair[0], "t": ht_pair[1], "r": rel_ugdre, "evidence": [],
                                     "source": "UGDRE"})
                    except KeyError:
                        title = t["title"]
                        my_logger.logger.info(f"Negative Example in UGDRE for pair: {(ht_pair[0], ht_pair[1])},"
                                              f" Title: {title}")

        new_data.append(tmp)
    my_logger.logger.info(f"Saving new data (n. document {len(new_data)}) in {args.data_dir}/{args.save_name}")
    json.dump(new_data, open(f"{args.data_dir}/{args.save_name}", "w"))


if __name__ == "__main__":
    main()