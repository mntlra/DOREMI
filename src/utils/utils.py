import os
import random
import torch
import numpy as np
import json
import pandas as pd
import csv

"""
Code from DREEAM GitHub repository. (https://github.com/YoumiMa/dreeam)
"""

def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d


def collate_fn(batch):

    max_len = max([len(f["input_ids"]) for f in batch])

    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    relation_mask = [f["relation_mask"] for f in batch if "relation_mask" in f]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    labels = [torch.tensor(label) for label in labels] 
    labels = torch.cat(labels, dim=0)

    relation_mask = [torch.tensor(rel_mask) for rel_mask in relation_mask]
    relation_mask = torch.cat(relation_mask, dim=0)

    output = (input_ids, input_mask, labels, entity_pos, hts, relation_mask)

    return output


def dump_to_file(logger, offi: list, offi_path: str, scores: list, score_path: str, thresh: list = [], thresh_filename: str=None):
    '''
    dump scores and (top-k) predictions to file.

    '''
    logger.logger.info(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))

    logger.logger.info(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns=headers)
    logger.logger.info(scores_pd)
    scores_pd.to_csv(score_path, sep='\t')

    """if len(thresh) > 0:
        thresh_path = os.path.join(os.path.dirname(offi_path), thresh_filename)
        with open(thresh_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "threshold"])
            for row in thresh:
                writer.writerow(row)"""

    return