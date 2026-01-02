from tqdm import tqdm
import ujson as json
import numpy as np
import pickle
import os
import torch
import random

"""
Prepro code from DREEAM GitHub repository. (https://github.com/YoumiMa/dreeam)
Data processing for training and testing BERT.
"""
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}

def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
    # add * marks to the beginning and end of entities
        new_map = {}
        
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        
        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end
        
        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos

def read_docred_for_BERT(args, logger, data_dir, file_in, tokenizer, transformer_type="bert", max_seq_length=1024):
                #single_results=None):

    docred_rel2id = json.load(open(data_dir + '/meta/rel2id.json', 'r'))
    logger.logger.info(f"Reading rel2id from {data_dir}/meta/rel2id.json ..")

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []

    if file_in == "":
        return None

    with open(file_in, "r") as fh:
        logger.logger.info(f"Reading {file_in} ..")
        data = json.load(fh)

    for doc_id in tqdm(range(len(data)), desc="Loading examples"):

        sample = data[doc_id]
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        # add entity markers
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}

        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])

                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        # entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([])
            assert len(e) != 0
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end,))

        relations, hts, sent_labels, relation_mask = [], [], [], []

        for h, t in train_triple.keys():  # for every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)

            if "include_pairs" in sample.keys():
                doc_title = sample["title"]
                # logger.logger.info(f"[POS] Document {doc_title} has include_pairs")
                # logger.logger.info(f"[POS] Pairs to include: {sample['include_pairs']}")
                # logger.logger.info(f"[POS] [h,t] ({[h,t]}) in pairs to include? {[h, t] in sample['include_pairs']}")
                if [h, t] in sample["include_pairs"]:
                    logger.logger.info(f"[POS] Including pair ({h}, {t}) for document {doc_title}")
                    rel_mask = [1] * len(docred_rel2id)
                else:
                    logger.logger.info(f"[POS] Excluding pair ({h}, {t}) for document {doc_title}")
                    rel_mask = [0] * len(docred_rel2id)
            else:
                doc_title = sample["title"]
                # logger.logger.info(f"[POS] Document {doc_title} has not include_pairs")
                rel_mask = [1] * len(docred_rel2id)

            for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
                relation[mention["relation"]] = 1
                for i in mention["evidence"]:
                    sent_evi[i] += 1

            relations.append(relation)
            relation_mask.append(rel_mask)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                # all entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts:  # and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)

                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1

                    if "include_pairs" in sample.keys():
                        doc_title = sample["title"]
                        # logger.logger.info(f"[N/A pairs] Document {doc_title} has include_pairs")
                        # logger.logger.info(f"[N/A pairs] Pairs to include: {sample['include_pairs']}")
                        # logger.logger.info(f"[N/A pairs] [h,t] ({[h, t]}) in pairs to include? {[h, t] in sample['include_pairs']}")
                        if [h, t] in sample["include_pairs"]:
                            logger.logger.info(f"[N/A pairs] Including pair ({h}, {t}) for document {doc_title}")
                            rel_mask = [1] * len(docred_rel2id)
                        else:
                            logger.logger.info(f"[N/A pairs] Excluding pair ({h}, {t}) for document {doc_title}")
                            rel_mask = [0] * len(docred_rel2id)
                    else:
                        doc_title = sample["title"]
                        # logger.logger.info(f"[N/A pairs] Document {doc_title} has not  include_pairs")
                        rel_mask = [1] * len(docred_rel2id)
                    relation_mask.append(rel_mask)

        assert len(relations) == len(entities) * (len(entities) - 1)
        assert len(sents) < max_seq_length
        sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = [{'input_ids': input_ids,
                    'entity_pos': entity_pos,
                    'labels': relations,
                    'hts': hts,
                    'sent_pos': sent_pos,
                    'sent_labels': sent_labels,
                    'title': sample['title'],
                    'relation_mask': relation_mask
                    }]

        i_line += len(feature)
        features.extend(feature)

    logger.logger.info("# of documents {}.".format(i_line))

    # else:
    logger.logger.info("# of positive examples {}.".format(pos_samples))
    logger.logger.info("# of negative examples {}.".format(neg_samples))

    return features

