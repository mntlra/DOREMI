import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM
import random
from opt_einsum import contract
from utils.losses import ATLoss
from utils.bert_utils import process_long_input
import numpy as np

"""
    Adapted from DREEAM GitHub repository. (https://github.com/YoumiMa/dreeam)
"""

class BERT(nn.Module):
    def __init__(self, my_logger, config, model, tokenizer, device, emb_size=768, block_size=64):
        '''
            Initialize the model.
            :model: Pretrained langage model encoder;
            :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
            :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
            :block_size: Number of blocks for grouped bilinear classification;
            :num_labels: Maximum number of relation labels for each entity pair;
            :max_sent_num: Maximum number of sentences for each document.
        '''
        super().__init__()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_fnt = ATLoss()
        self.device = device
        self.logger = my_logger

        dis_size = 21
        self.head_extractor = nn.Linear(self.hidden_size + dis_size, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size + dis_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.dis2idx = np.zeros((1024), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis2idx[512:] = 10
        
        self.embedding_layer = nn.Embedding(dis_size+1, dis_size, padding_idx=11)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens, self.logger)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :dss: (num_ent_pairs_all_batches, emb_size)
        '''

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, dss, dht = [], [], [], []
        for i in range(len(entity_pos)):  # for each document
            entity_embs = []
            entity_first_pos = []
            first_mention = c+offset
            for e in entity_pos[i]:  # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:  # for every mention
                        if start < first_mention:
                            first_mention = start
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.stack(e_emb, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                else:
                    start, end = e[0]
                    first_mention = start
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)

                entity_embs.append(e_emb)
                entity_first_pos.append(first_mention)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            # embedding for distance between the two entities
            # Computing the distance idx
            d_ht = [self.dis2idx[entity_first_pos[h] - entity_first_pos[t]]+11 if (entity_first_pos[h] - entity_first_pos[t]) > 0 else 11-(self.dis2idx[abs(entity_first_pos[h] - entity_first_pos[t])]) for t, h in hts[i]]
            ds = self.embedding_layer(torch.tensor(d_ht).to(self.device))

            hss.append(hs)
            tss.append(ts)
            dss.append(ds)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        dss = torch.cat(dss, dim=0)
        return hss, dss, tss 

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                relation_mask=None,
                tag="dev",
                num_embeddings=False):

        outputs = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, ds, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        # put tensors in the same device
        hs = hs.to(self.device)
        ds = ds.to(self.device)
        ts = ts.to(self.device)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, ds], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, ds], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        if tag in ["test", "dev", "infer"]:
            outputs["probs"] = torch.sigmoid(logits)
        else:
            # training
            loss = self.loss(logits.float(), labels.float())
            if relation_mask is not None:
                masked_loss = self.loss(logits.float(), labels.float())*relation_mask
                outputs["loss"] = (masked_loss.sum(1)).mean()
            else:
                outputs["loss"] = (self.loss(logits.float(), labels.float()).sum(1)).mean()
            outputs["probs"] = torch.sigmoid(logits)

        return outputs

