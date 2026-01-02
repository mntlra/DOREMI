import os
import os.path
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, classification_report
import pandas as pd

def get_title2pred(pred: list) -> dict:
    '''
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']  
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    '''
    
    title2pred = {}

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])
        
        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(p["score"], title2pred[p["title"]][curr])
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            title2pred[p["title"]] = {curr: p["score"]}
    return title2pred

def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    '''
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand. 
    '''
    
    sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
    precs, recalls = [], []
    
    for pred in sorted_pred:     
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred)  # Precision
        recalls.append(correct / num_gt)  # Recall                             

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print('Best thresh', thresh, '\tbest F1', f1)
    return thresh, sorted_pred[:f1_pos + 1]


def extract_relative_score(scores: list, topks: list) -> list:
    '''
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    '''
    
    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks==0)].item()     
    
    scores -= na_score

    return scores


def gen_train_facts(data_file_name, truth_dir):
    
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate_per_rel(logger, tmp, path, train_file = "train_annotated.json", dev_file = "dev.json"):
    '''
        Evaluation relation-wise. Compute precision, accuracy, and F1 score for each relation label.
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))
        
    std = {}
    parsed_relations_truth = []
    tot_evidences = {}
    tot_relations = {}
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        if 'labels' not in x:   # official test set from DocRED
            continue
        
        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            if r in parsed_relations_truth:
                tot_evidences[r] += len(label['evidence'])
                tot_relations[r] += 1
            else:
                parsed_relations_truth.append(r)
                tot_evidences[r] = len(label['evidence'])
                tot_relations[r] = 1

            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            # tot_evidences += len(label['evidence'])


    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    num_predicted_relations = {tmp[0]["r"]: 1}

    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
            try:
                num_predicted_relations[tmp[i]["r"]] += 1
            except KeyError:
                num_predicted_relations[tmp[i]["r"]] = 1

    correct_re = {}
    correct_evidence = {}
    pred_evi = {}

    correct_in_train_annotated = {}
    correct_in_train_distant = {}

    parsed_relations = []

    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']

        if r not in parsed_relations:
            parsed_relations.append(r)
            correct_re[r] = 0
            correct_evidence[r] = 0
            pred_evi[r] = 0
            correct_in_train_annotated[r] = 0
            correct_in_train_distant[r] = 0

        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x : #and (title, h_idx, t_idx) in std:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi[r] += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re[r] += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence[r] += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated[r] += 1
            if in_train_distant:
                correct_in_train_distant[r] += 1

    output_dict = {}
    for r in parsed_relations_truth:
        if r not in parsed_relations:
            correct_re[r] = 0
            correct_evidence[r] = 0
            pred_evi[r] = 0
            correct_in_train_annotated[r] = 0
            correct_in_train_distant[r] = 0
            num_predicted_relations[r] = 0

        re_p = 1.0 * correct_re[r] / num_predicted_relations[r] if num_predicted_relations[r] != 0 else 0
        re_r = 1.0 * correct_re[r] / tot_relations[r] if tot_relations[r] != 0 else 0
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence[r] / pred_evi[r] if pred_evi[r] > 0 else 0
        evi_r = 1.0 * correct_evidence[r] / tot_evidences[r] if tot_evidences[r] > 0 else 0

        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re[r] - correct_in_train_annotated[r]) / (num_predicted_relations[r] - correct_in_train_annotated[r] + 1e-5)
        re_p_ignore_train = 1.0 * (correct_re[r] - correct_in_train_distant[r]) / (num_predicted_relations[r] - correct_in_train_distant[r] + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        output_dict[r] = [re_p, re_r, re_f1, re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated, evi_p, evi_r, evi_f1]

    return output_dict
    # return [re_p, re_r, re_f1], [evi_p, evi_r, evi_f1], [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], [re_p_ignore_train, re_r, re_f1_ignore_train]


def official_evaluate_long_tail(args, logger, tmp, path, dev_file="dev.json"):
    '''
        Evaluation of long-tailed relations (macro@K, with K=100; 200; 500; all)
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, args.train_file), truth_dir)
    # fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    parsed_relations_truth = []
    if args.long_300:
        tot_rel = {"100": 0, "300": 0, "500": 0, "1000": 0}
    else:
        tot_rel = {"100": 0, "200": 0, "500": 0}
    tot_relations = {}
    titleset = set([])

    title2vectexSet = {}

    rels_freq = json.load(open(os.path.join(path, f"meta/{args.relations_frequency}")))

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        if 'labels' not in x:  # official test set from DocRED
            continue

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            if r in parsed_relations_truth:
                # tot_evidences[r] += len(label['evidence'])
                tot_relations[r] += 1
            else:
                parsed_relations_truth.append(r)
                # tot_evidences[r] = len(label['evidence'])
                tot_relations[r] = 1
            if args.long_300:
                if r in rels_freq["100"]:
                    tot_rel["100"] += 1
                if r in rels_freq["300"]:
                    tot_rel["300"] += 1
                if r in rels_freq["500"]:
                    tot_rel["500"] += 1
                if r in rels_freq["1000"]:
                    tot_rel["1000"] += 1
            else:
                if r in rels_freq["100"]:
                    tot_rel["100"] += 1
                if r in rels_freq["200"]:
                    tot_rel["200"] += 1
                if r in rels_freq["500"]:
                    tot_rel["500"] += 1

            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            # tot_evidences += len(label['evidence'])

    tot_rel["ALL"] = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    num_predicted_relations = {tmp[0]["r"]: 1}
    if args.long_300:
        num_predicted = {"100": 0, "300": 0, "500": 0, "1000": 0}
    else:
        num_predicted = {"100": 0, "200": 0, "500": 0}

    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
            try:
                num_predicted_relations[tmp[i]["r"]] += 1
            except KeyError:
                num_predicted_relations[tmp[i]["r"]] = 1
            if args.long_300:
                if tmp[i]["r"] in rels_freq["100"]:
                    num_predicted["100"] += 1
                if tmp[i]["r"] in rels_freq["300"]:
                    num_predicted["300"] += 1
                if tmp[i]["r"] in rels_freq["500"]:
                    num_predicted["500"] += 1
                if tmp[i]["r"] in rels_freq["1000"]:
                    num_predicted["1000"] += 1
            else:
                if tmp[i]["r"] in rels_freq["100"]:
                    num_predicted["100"] += 1
                if tmp[i]["r"] in rels_freq["200"]:
                    num_predicted["200"] += 1
                if tmp[i]["r"] in rels_freq["500"]:
                    num_predicted["500"] += 1

    num_predicted["ALL"] = len(submission_answer)
    correct_re = {}
    correct_re_in_train_annotated = {}
    if args.long_300:
        corrected = {"ALL": 0, "100": 0, "300": 0, "500": 0, "1000": 0}
        correct_in_train_annotated = {"ALL": 0, "100": 0, "300": 0, "500": 0, "1000": 0}
    else:
        corrected = {"ALL": 0, "100": 0, "200": 0, "500": 0}
        correct_in_train_annotated = {"ALL": 0, "100": 0, "200": 0, "500": 0}

    parsed_relations = []

    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']

        if r not in parsed_relations:
            parsed_relations.append(r)
            correct_re[r] = 0
            correct_re_in_train_annotated[r] = 0

        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re[r] += 1
            corrected["ALL"] += 1
            in_train_annotated = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True

            if in_train_annotated:
                correct_in_train_annotated["ALL"] += 1
                correct_re_in_train_annotated[r] += 1

            if args.long_300:
                if r in rels_freq["100"]:
                    corrected["100"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["100"] += 1
                if r in rels_freq["300"]:
                    corrected["300"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["300"] += 1
                if r in rels_freq["500"]:
                    corrected["500"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["500"] += 1
                if r in rels_freq["1000"]:
                    corrected["1000"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["1000"] += 1
            else:
                if r in rels_freq["100"]:
                    corrected["100"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["100"] += 1
                if r in rels_freq["200"]:
                    corrected["200"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["200"] += 1
                if r in rels_freq["500"]:
                    corrected["500"] += 1
                    if in_train_annotated:
                        correct_in_train_annotated["500"] += 1

    f1_scores, f1_ign_scores, prec_scores, prec_ign_scores, rec_scores = {}, {}, {}, {}, {}
    macro_f1, macro_f1_ign, prec, prec_ign, rec = 0, 0, 0, 0, 0
    # logger.logger.info(f"parsed_relations_truth (n. {len(parsed_relations_truth)}): {parsed_relations_truth}")
    for r in parsed_relations_truth:
        if r not in parsed_relations:
            correct_re[r] = 0
            correct_re_in_train_annotated[r] = 0
            num_predicted_relations[r] = 0

        re_p = 1.0 * correct_re[r] / num_predicted_relations[r] if num_predicted_relations[r] != 0 else 0
        re_p_ign = 1.0 * (correct_re[r] - correct_re_in_train_annotated[r]) / (
                    num_predicted_relations[r] - correct_re_in_train_annotated[r]) if (num_predicted_relations[r] -
                                                                                       correct_re_in_train_annotated[
                                                                                           r]) != 0 else 0
        re_r = 1.0 * correct_re[r] / tot_relations[r] if tot_relations[r] != 0 else 0
        prec += re_p
        prec_ign += re_p_ign
        prec_scores[r] = re_p
        prec_ign_scores[r] = re_p_ign
        rec += re_r
        rec_scores[r] = re_r
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        if re_p_ign + re_r == 0:
            re_f1_ign = 0
        else:
            re_f1_ign = 2.0 * re_p_ign * re_r / (re_p_ign + re_r)

        f1_scores[r] = re_f1
        macro_f1 += re_f1

        f1_ign_scores[r] = re_f1_ign
        macro_f1_ign += re_f1_ign

    output_dict = {}

    # micro-averaged

    for k in corrected.keys():
        micro_prec = 1.0 * corrected[k] / num_predicted[k] if num_predicted[k] != 0 else 0
        micro_prec_ign = 1.0 * (corrected[k] - correct_in_train_annotated[k]) / (
                    num_predicted[k] - correct_in_train_annotated[k]) if (num_predicted[k] - correct_in_train_annotated[
            k]) != 0 else 0
        micro_rec = 1.0 * corrected[k] / tot_rel[k] if tot_rel[k] != 0 else 0
        if micro_prec + micro_rec == 0:
            micro_f1 = 0
        else:
            micro_f1 = 2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if micro_prec_ign + micro_rec == 0:
            micro_f1_ign = 0
        else:
            micro_f1_ign = 2.0 * micro_prec_ign * micro_rec / (micro_prec_ign + micro_rec)

        output_dict[f"micro@{k}"] = [micro_f1, micro_f1_ign, micro_prec, micro_prec_ign, micro_rec]

    # macro-averaged
    macro_f1 = macro_f1 / len(parsed_relations)
    macro_prec = prec / len(parsed_relations)
    macro_f1_ign = macro_f1_ign / len(parsed_relations)
    macro_prec_ign = prec_ign / len(parsed_relations)
    macro_rec = rec / len(parsed_relations)
    output_dict["macro@ALL"] = [macro_f1, macro_f1_ign, macro_prec, macro_prec_ign, macro_rec]

    if args.long_300:

        f1_100, f1_300, f1_200, f1_500 = 0, 0, 0, 0
        f1_ign_100, f1_ign_300, f1_ign_200, f1_ign_500 = 0, 0, 0, 0
        prec_100, prec_300, prec_200, prec_500 = 0, 0, 0, 0
        prec_ign_100, prec_ign_300, prec_ign_200, prec_ign_500 = 0, 0, 0, 0
        rec_100, rec_300, rec_200, rec_500 = 0, 0, 0, 0

        logger.logger.info(f"*** NUMBER OF CONSIDERED RELATIONS: {len(f1_scores.keys())} ***")

        for rel in rels_freq["100"]:
            if rel in f1_scores.keys():
                f1_100 += f1_scores[rel]
                rec_100 += rec_scores[rel]
                prec_100 += prec_scores[rel]
                f1_ign_100 += f1_ign_scores[rel]
                prec_ign_100 += prec_ign_scores[rel]
        output_dict["macro@100"] = [f1_100 / len(rels_freq["100"]), f1_ign_100 / len(rels_freq["100"]),
                                    prec_100 / len(rels_freq["100"]), prec_ign_100 / len(rels_freq["100"]),
                                    rec_100 / len(rels_freq["100"])]

        for rel in rels_freq["300"]:
            if rel in f1_scores.keys():
                f1_300 += f1_scores[rel]
                rec_300 += rec_scores[rel]
                prec_300 += prec_scores[rel]
                f1_ign_300 += f1_ign_scores[rel]
                prec_ign_300 += prec_ign_scores[rel]
        output_dict["macro@300"] = [f1_300 / len(rels_freq["300"]), f1_ign_300 / len(rels_freq["300"]),
                                    prec_300 / len(rels_freq["300"]), prec_ign_300 / len(rels_freq["300"]),
                                    rec_300 / len(rels_freq["300"])]

        for rel in rels_freq["500"]:
            if rel in f1_scores.keys():
                f1_200 += f1_scores[rel]
                rec_200 += rec_scores[rel]
                prec_200 += prec_scores[rel]
                f1_ign_200 += f1_ign_scores[rel]
                prec_ign_200 += prec_ign_scores[rel]
        output_dict["macro@500"] = [f1_200 / len(rels_freq["500"]), f1_ign_200 / len(rels_freq["500"]),
                                    prec_200 / len(rels_freq["500"]), prec_ign_200 / len(rels_freq["500"]),
                                    rec_200 / len(rels_freq["500"])]

        for rel in rels_freq["1000"]:
            if rel in f1_scores.keys():
                f1_500 += f1_scores[rel]
                rec_500 += rec_scores[rel]
                prec_500 += prec_scores[rel]
                f1_ign_500 += f1_ign_scores[rel]
                prec_ign_500 += prec_ign_scores[rel]
        output_dict["macro@1000"] = [f1_500 / len(rels_freq["1000"]), f1_ign_500 / len(rels_freq["1000"]),
                                     prec_500 / len(rels_freq["1000"]), prec_ign_500 / len(rels_freq["1000"]),
                                     rec_500 / len(rels_freq["1000"])]

        return output_dict

    else:
        f1_100, f1_200, f1_500 = 0, 0, 0
        prec_100, prec_200, prec_500 = 0, 0, 0
        rec_100, rec_200, rec_500 = 0, 0, 0
        f1_ign_100, f1_ign_200, f1_ign_500 = 0, 0, 0
        prec_ign_100, prec_ign_200, prec_ign_500 = 0, 0, 0

        logger.logger.info(f"*** NUMBER OF CONSIDERED RELATIONS: {len(f1_scores.keys())} ***")

        for rel in rels_freq["100"]:
            if rel in f1_scores.keys():
                f1_100 += f1_scores[rel]
                rec_100 += rec_scores[rel]
                prec_100 += prec_scores[rel]
                f1_ign_100 += f1_ign_scores[rel]
                prec_ign_100 += prec_ign_scores[rel]
        output_dict["macro@100"] = [f1_100 / len(rels_freq["100"]), f1_ign_100 / len(rels_freq["100"]),
                                    prec_100 / len(rels_freq["100"]), prec_ign_100 / len(rels_freq["100"]),
                                    rec_100 / len(rels_freq["100"])]

        for rel in rels_freq["200"]:
            if rel in f1_scores.keys():
                f1_200 += f1_scores[rel]
                rec_200 += rec_scores[rel]
                prec_200 += prec_scores[rel]
                f1_ign_200 += f1_ign_scores[rel]
                prec_ign_200 += prec_ign_scores[rel]
        output_dict["macro@200"] = [f1_200 / len(rels_freq["200"]), f1_ign_200 / len(rels_freq["200"]),
                                    prec_200 / len(rels_freq["200"]), prec_ign_200 / len(rels_freq["200"]),
                                    rec_200 / len(rels_freq["200"])]

        for rel in rels_freq["500"]:
            if rel in f1_scores.keys():
                f1_500 += f1_scores[rel]
                rec_500 += rec_scores[rel]
                prec_500 += prec_scores[rel]
                f1_ign_500 += f1_ign_scores[rel]
                prec_ign_500 += prec_ign_scores[rel]
        output_dict["macro@500"] = [f1_500 / len(rels_freq["500"]), f1_ign_500 / len(rels_freq["500"]),
                                    prec_500 / len(rels_freq["500"]), prec_ign_500 / len(rels_freq["500"]),
                                    rec_500 / len(rels_freq["500"])]

    return output_dict

