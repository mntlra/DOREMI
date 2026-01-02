# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import random
from collections import defaultdict
import torch.nn.functional as F
from evaluation import official_evaluate
from evaluation_relationwise import official_evaluate_per_rel, official_evaluate_long_tail
import csv

"""
Code from DocRED GitHub repository. (https://github.com/thunlp/DocRED/tree/master)
"""

IGNORE_INDEX = -100
is_transformer = False

class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self, args, always_evaluate=False):
		self.args = args
		self.device = args.device
		self.logger = args.logger
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.prepro_data_path = args.data_dir + '/prepro_data/'
		self.meta_path = args.data_dir + '/meta/'
		self.data_path = args.data_dir
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.max_length = 512
		self.pos_num = 2 * self.max_length
		self.entity_num = self.max_length
		self.relation_num = args.num_class
		self.always_evaluate = always_evaluate

		self.coref_size = 20
		self.entity_type_size = 20
		self.max_epoch = args.num_train_epochs
		self.opt_method = 'Adam'
		self.optimizer = None

		self.load_path = args.load_path
		self.save_path = args.save_path
		self.test_epoch = 5
		self.pretrain_model = None


		self.word_size = 100
		self.epoch_range = None
		self.cnn_drop_prob = 0.5  # for cnn
		self.keep_prob = 0.8  # for lstm

		self.period = 50

		self.batch_size = args.train_batch_size
		self.h_t_limit = 1800

		self.test_batch_size = args.test_batch_size
		self.test_relation_limit = 1800
		self.char_limit = 16
		self.sent_limit = 25
		self.dis2idx = np.zeros((512), dtype='int64')
		self.dis2idx[1] = 1
		self.dis2idx[2:] = 2
		self.dis2idx[4:] = 3
		self.dis2idx[8:] = 4
		self.dis2idx[16:] = 5
		self.dis2idx[32:] = 6
		self.dis2idx[64:] = 7
		self.dis2idx[128:] = 8
		self.dis2idx[256:] = 9
		self.dis_size = 20

		self.train_prefix = args.train_prefix
		self.test_prefix = args.test_prefix

		self.pref2name = {"dev_train": "train_annotated", "dev_dev": "dev", "train_distant": "train_distant",
						  "dev_dev_eval": "dev_eval", "dev_dev_sample": "dev_sample"}

	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range
	
	def load_train_data(self):
		self.logger.logger.info("Reading training data...")
		prefix = self.train_prefix

		self.logger.logger.info(f'train file: {prefix}')
		self.data_train_word = np.load(os.path.join(self.prepro_data_path, prefix+'_word.npy'))
		self.data_train_pos = np.load(os.path.join(self.prepro_data_path, prefix+'_pos.npy'))
		self.data_train_ner = np.load(os.path.join(self.prepro_data_path, prefix+'_ner.npy'))
		self.data_train_char = np.load(os.path.join(self.prepro_data_path, prefix+'_char.npy'))
		self.train_file = json.load(open(os.path.join(self.prepro_data_path, prefix+'.json')))

		self.logger.logger.info(f"Finish reading train data. Size of {self.train_prefix} test file: {len(self.train_file)}")

		self.train_len = ins_num = self.data_train_word.shape[0]
		assert(self.train_len==len(self.train_file))

		self.train_order = list(range(ins_num))
		self.train_batches = ins_num // self.batch_size
		if ins_num % self.batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		self.logger.logger.info("Reading testing data...")
		prefix = self.test_prefix
		self.logger.logger.info(f"Test prefix: {prefix}")
		"""if test_prefix != "":
			self.logger.logger.info(f"Setting test prefix to {test_prefix}")
			self.test_prefix = test_prefix
			prefix = test_prefix
		else:
			prefix = self.test_prefix"""
		self.data_word_vec = np.load(os.path.join(self.meta_path, 'vec.npy'))
		self.data_char_vec = np.load(os.path.join(self.meta_path, 'char_vec.npy'))
		self.rel2id = json.load(open(os.path.join(self.meta_path, 'rel2id.json')))
		self.id2rel = {v: k for k,v in self.rel2id.items()}

		# prefix = self.test_prefix
		self.logger.logger.info(f"Test file: {prefix}")
		self.is_test = ('dev_test' == prefix)
		self.data_test_word = np.load(os.path.join(self.prepro_data_path, prefix+'_word.npy'))
		self.data_test_pos = np.load(os.path.join(self.prepro_data_path, prefix+'_pos.npy'))
		self.data_test_ner = np.load(os.path.join(self.prepro_data_path, prefix+'_ner.npy'))
		self.data_test_char = np.load(os.path.join(self.prepro_data_path, prefix+'_char.npy'))
		self.test_file = json.load(open(os.path.join(self.prepro_data_path, prefix+'.json')))


		self.test_len = self.data_test_word.shape[0]
		assert(self.test_len==len(self.test_file))


		self.logger.logger.info(f"Finish reading test file. Size of {self.test_prefix} test file: {len(self.test_file)}")

		self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
		if self.data_test_word.shape[0] % self.test_batch_size != 0:
			self.test_batches += 1

		self.test_order = list(range(self.test_len))
		self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)


	def get_train_batch(self):
		random.shuffle(self.train_order)

		context_idxs = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
		context_pos = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
		h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).to(self.device)
		t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).to(self.device)
		relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).to(self.device)
		ori_relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).to(self.device)
		relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).to(self.device)

		pos_idx = torch.LongTensor(self.batch_size, self.max_length).to(self.device)

		context_ner = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
		context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).to(self.device)

		relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)


		ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)

		docred_rel2id = json.load(open(self.args.data_dir + '/meta/rel2id.json', 'r'))

		for b in range(self.train_batches):
			start_id = b * self.batch_size
			cur_bsz = min(self.batch_size, self.train_len - start_id)
			cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
			cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0), reverse=True)

			for mapping in [h_mapping, t_mapping]:
				mapping.zero_()

			for mapping in [relation_multi_label, relation_mask, pos_idx, ori_relation_mask]:
				mapping.zero_()

			ht_pair_pos.zero_()


			relation_label.fill_(IGNORE_INDEX)

			max_h_t_cnt = 1

			relations_batch = []
			# pairs_excluded = []

			for i, index in enumerate(cur_batch):
				context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
				context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))
				context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))

				for j in range(self.max_length):
					if self.data_train_word[index, j]==0:
						break
					pos_idx[i, j] = j+1

				ins = self.train_file[index]
				labels = ins['labels']
				idx2label = defaultdict(list)
				"""if "pairs_to_exclude" in ins.keys():
					pairs_excluded.append(ins["pairs_to_exclude"])"""

				# training triples with positive examples (entity pairs with labels)
				train_triple = {}

				for label in labels:
					idx2label[(label['h'], label['t'])].append(label['r'])

					# update training triples
					if (label['h'], label['t']) not in train_triple:
						train_triple[(label['h'], label['t'])] = [
							{'relation': label['r']}]
					else:
						train_triple[(label['h'], label['t'])].append(
							{'relation': label['r']})

				relations = []
				for h, t in train_triple.keys():  # for every entity pair with gold relation
					relation = [0] * len(docred_rel2id)
					# sent_evi = [0] * len(sent_pos)

					for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
						relation[mention["relation"]] = 1

					relations.append(relation)

				train_tripe = list(idx2label.keys())
				for j, (h_idx, t_idx) in enumerate(train_tripe):
					hlist = ins['vertexSet'][h_idx]
					tlist = ins['vertexSet'][t_idx]

					for h in hlist:
						h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

					for t in tlist:
						t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

					label = idx2label[(h_idx, t_idx)]

					delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
					if delta_dis < 0:
						ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
					else:
						ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])


					for r in label:
						relation_multi_label[i, j, r] = 1

					ori_relation_mask[i, j] = 1
					# self.logger.logger.info(f"pairs_to_exclude in ins.keys of {ins['title']}? {'pairs_to_exclude' in ins.keys()}")
					# self.logger.logger.info(f"include_pairs in ins.keys of {ins['title']}? {'include_pairs' in ins.keys()}")
					# if self.args.exclude_pairs:
					"""
					if "pairs_to_exclude" in ins.keys():
						self.logger.logger.info(f"ins[pairs_to_exclude] {ins['pairs_to_exclude']}")
						self.logger.logger.info(f"[h,t] ({h_idx}, {t_idx}) in pairs to exclude? {[h_idx, t_idx] in ins['pairs_to_exclude']}")
						if [h_idx, t_idx] in ins["pairs_to_exclude"]:
							doc_title = ins["title"]
							self.logger.logger.info(f"[POS] Excluding pair ({h_idx}, {t_idx}) for document {doc_title}")
							relation_mask[i, j] = 0
						else:
							relation_mask[i, j] = 1
					else:
						relation_mask[i, j] = 1
					"""

					if "include_pairs" in ins.keys():
						# self.logger.logger.info(f"ins[include_pairs] {ins['include_pairs']}")
						# self.logger.logger.info(f"[h,t] ({h_idx}, {t_idx}) in pairs to include? {[h_idx, t_idx] in ins['include_pairs']}")
						doc_title = ins["title"]
						if [h_idx, t_idx] in ins["include_pairs"]:
							# self.logger.logger.info(f"[POS] Including pair ({h_idx}, {t_idx}) for document {doc_title}")
							relation_mask[i, j] = 1
						else:
							# self.logger.logger.info(f"[POS] Excluding pair ({h_idx}, {t_idx}) for document {doc_title}")
							relation_mask[i, j] = 0
					else:
						doc_title = ins["title"]
						# self.logger.logger.info(f"[POS] Document {doc_title} does not have include_pairs")
						relation_mask[i, j] = 1

					rt = np.random.randint(len(label))
					relation_label[i, j] = label[rt]



				lower_bound = len(ins['na_triple'])
				# random.shuffle(ins['na_triple'])
				# lower_bound = max(20, len(train_tripe)*3)


				for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_tripe)):
					hlist = ins['vertexSet'][h_idx]
					tlist = ins['vertexSet'][t_idx]

					for h in hlist:
						h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

					for t in tlist:
						t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

					relation_multi_label[i, j, 0] = 1
					relation_label[i, j] = 0
					ori_relation_mask[i, j] = 1
					# if self.args.exclude_pairs:
					"""if "pairs_to_exclude" in ins.keys():
						if [h_idx, t_idx] in ins["pairs_to_exclude"]:
							doc_title = ins["title"]
							self.logger.logger.info(f"[N/A triples] ins[pairs_to_exclude] {ins['pairs_to_exclude']}")
							self.logger.logger.info(f"[N/A triples] [h,t] ({h_idx}, {t_idx}) in pairs to exclude? {[h_idx, t_idx] in ins['pairs_to_exclude']}")
							self.logger.logger.info(f"[N/A triples] Excluding pair ({h_idx}, {t_idx}) for document {doc_title}")
							relation_mask[i, j] = 0
						else:
							relation_mask[i, j] = 1
					else:
						relation_mask[i, j] = 1"""

					if "include_pairs" in ins.keys():
						# self.logger.logger.info(f"[N/A triples] ins[include_pairs] {ins['include_pairs']}")
						# self.logger.logger.info(f"[N/A triples] [h,t] ({h_idx}, {t_idx}) in pairs to include? {[h_idx, t_idx] in ins['include_pairs']}")
						doc_title = ins["title"]
						if [h_idx, t_idx] in ins["include_pairs"]:
							# self.logger.logger.info(f"[N/A triples] Including pair ({h_idx}, {t_idx}) for document {doc_title}")
							relation_mask[i, j] = 1
						else:
							# self.logger.logger.info(f"[N/A triples] Including pair ({h_idx}, {t_idx}) for document {doc_title}")
							relation_mask[i, j] = 0
					else:
						doc_title = ins["title"]
						# self.logger.logger.info(f"[N/A triples] Document {doc_title} does not have include_pairs")
						relation_mask[i, j] = 1
					# else:
						# relation_mask[i, j] = 1
					delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
					if delta_dis < 0:
						ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
					else:
						ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

				max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

				relations_batch.append(relations)

			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())

			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
				   'input_lengths' : input_lengths,
				   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
				   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
				   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt] if self.args.exclude_pairs else ori_relation_mask[:cur_bsz, :max_h_t_cnt],
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
				   'label_matrix': relations_batch,
				   'ori_relation_mask': ori_relation_mask[:cur_bsz, :max_h_t_cnt]
				   }

	def get_test_batch(self):
		context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
		context_pos = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
		h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).to(self.device)
		t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).to(self.device)
		context_ner = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
		context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).to(self.device)
		relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).to(self.device)
		ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)

		docred_rel2id = json.load(open(self.args.data_dir + '/meta/rel2id.json', 'r'))

		relations_batch = []

		for b in range(self.test_batches):
			start_id = b * self.test_batch_size
			cur_bsz = min(self.test_batch_size, self.test_len - start_id)
			cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

			for mapping in [h_mapping, t_mapping, relation_mask]:
				mapping.zero_()


			ht_pair_pos.zero_()

			max_h_t_cnt = 1

			cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x]>0) , reverse = True)

			labels = []

			L_vertex = []
			titles = []
			indexes = []
			for i, index in enumerate(cur_batch):
				context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
				context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
				context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))



				idx2label = defaultdict(list)
				ins = self.test_file[index]

				# training triples with positive examples (entity pairs with labels)
				train_triple = {}

				for label in ins['labels']:
					idx2label[(label['h'], label['t'])].append(label['r'])

					# update training triples
					if (label['h'], label['t']) not in train_triple:
						train_triple[(label['h'], label['t'])] = [
							{'relation': label['r']}]
					else:
						train_triple[(label['h'], label['t'])].append(
							{'relation': label['r']})

				relations = []
				for h, t in train_triple.keys():  # for every entity pair with gold relation
					relation = [0] * len(docred_rel2id)
					# sent_evi = [0] * len(sent_pos)

					for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
						relation[mention["relation"]] = 1

					relations.append(relation)

				L = len(ins['vertexSet'])
				titles.append(ins['title'])

				j = 0
				for h_idx in range(L):
					for t_idx in range(L):
						if h_idx != t_idx:
							hlist = ins['vertexSet'][h_idx]
							tlist = ins['vertexSet'][t_idx]

							for h in hlist:
								h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
							for t in tlist:
								t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

							relation_mask[i, j] = 1

							delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
							# self.logger.logger.info(f"ht_pair_pos for delta_dis: {delta_dis}")
							if delta_dis < 0:
								ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
							else:
								ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
							# self.logger.logger.info(f"ht_pair_pos: {ht_pair_pos[i,j]}")
							j += 1


				max_h_t_cnt = max(max_h_t_cnt, j)
				label_set = {}
				for label in ins['labels']:
					if "train_distant" in self.test_prefix:
						label_set[(label['h'], label['t'], label['r'])] = False
					else:
						if self.train_prefix == "dev_train":
							label_set[(label['h'], label['t'], label['r'])] = label['in'+self.train_prefix]
						else:
							label_set[(label['h'], label['t'], label['r'])] = label['indev_train']


				labels.append(label_set)


				L_vertex.append(L)
				indexes.append(index)
				relations_batch.append(relations)



			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())

			# self.logger.logger.info(f"self.dis2idx: {self.dis2idx}")

			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   'labels': labels,
				   'L_vertex': L_vertex,
				   'input_lengths': input_lengths,
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
				   'titles': titles,
				   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
				   'indexes': indexes,
				   'label_matrix': relations_batch
				   }

	def train_config(self, model_pattern, model_name, device_ids=[]):

		# self.logger.logger.info(f"Creating the {model_name} instance..")
		model = model_pattern(config=self, my_logger=self.logger)
		# self.logger.logger.info(f"Instance Created!")
		resume = False
		if self.load_path != "":
			# load model from existing checkpoint
			if self.args.checkpoint_file == "":
				model_path = os.path.join(self.load_path, "best.ckpt")
			else:
				model_path = os.path.join(self.load_path, self.args.checkpoint_file)
				resume = True
			self.logger.logger.info(f"Loading model from {model_path}")
			model.load_state_dict(torch.load(model_path, map_location=self.device))

		"""ori_model.to(self.device)
		self.logger.logger.info(f"Model is loaded in device {self.device}")
		if len(device_ids) > 0:
			self.logger.logger.info(f"Using data parallel in {len(device_ids)} devices")
			model = nn.DataParallel(ori_model, device_ids=device_ids)
		else:
			self.logger.logger.info(f"Using data parallel in a single devices")
			model = nn.DataParallel(ori_model)
		self.logger.logger.info(f"Model is loaded in device {self.device}")"""
		model.to(self.device)

		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
		# nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
		BCE = nn.BCEWithLogitsLoss(reduction='none')

		if not os.path.exists(self.save_path):
			os.mkdir(self.save_path)

		best_auc = 0.0
		best_f1 = 0.0
		best_epoch = 0

		model.train()

		global_step = 0
		total_loss = 0
		start_time = time.time()
		best_threshold = 0.0
		thresh_path = os.path.join(self.save_path, f"thresh.csv")
		self.logger.logger.info(f"Threshold path: {thresh_path}")
		if resume is False:
			with open(thresh_path, "w") as file:
				writer = csv.writer(file)
				writer.writerow(["epoch", "threshold"])

		self.logger.logger.info(f"*** Training for {self.max_epoch} epochs "
								f"for {self.train_batches} batches of size {self.batch_size} ***")
		for epoch in tqdm(range(int(self.max_epoch)), desc="Training Epoch: "):

			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()

			batch = 1
			for data in self.get_train_batch():

				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				h_mapping = data['h_mapping']
				t_mapping = data['t_mapping']
				relation_label = data['relation_label']
				input_lengths = data['input_lengths']
				relation_multi_label = data['relation_multi_label']
				relation_mask = data['relation_mask']
				context_ner = data['context_ner']
				context_char_idxs = data['context_char_idxs']
				ht_pair_pos = data['ht_pair_pos']
				ori_relation_mask = data['ori_relation_mask']


				dis_h_2_t = ht_pair_pos+10
				dis_t_2_h = -ht_pair_pos+10

				"""if self.args.exclude_pairs:
					r_mask = relation_mask
				else:
					r_mask = ori_relation_mask"""
				r_mask = relation_mask
				predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths, h_mapping, t_mapping, r_mask, dis_h_2_t, dis_t_2_h)
				loss = torch.sum(BCE(predict_re, relation_multi_label)*r_mask.unsqueeze(2)) / (self.relation_num * torch.sum(r_mask))
				# self.logger.logger.info(f"Number of ones in relation_mask: {torch.sum(relation_mask)}")
				# self.logger.logger.info(f"Number of ones in ori_relation_mask: {torch.sum(ori_relation_mask)}")
				if torch.sum(ori_relation_mask) < torch.sum(relation_mask):
					raise ValueError(f"relation_mask must have fewer ones than ori_relation_mask, but relation_mask has {torch.sum(relation_mask)} ones, while ori_relation_mask has {torch.sum(ori_relation_mask)}")

				output = torch.argmax(predict_re, dim=-1)
				output = output.data.cpu().numpy()

				"""
				LEGACY DEBUG PRINTS
				self.logger.logger.info(f"[SIZE] Size of predict_re: {predict_re.size()}")
				self.logger.logger.info(f"[SIZE] Size of predict_re[0]: {predict_re[0].size()}")
				self.logger.logger.info(f"[SIZE] Size of predict_re[0][0]: {predict_re[0][0].size()}")
				self.logger.logger.info(f"[SIZE] Size of relation_multi_label: {relation_multi_label.size()}")
				self.logger.logger.info(f"[SIZE] Size of relation_mask: {relation_mask.size()}")
				self.logger.logger.info(f"relation_mask: {relation_mask}")
				self.logger.logger.info(f"predict_re: {predict_re}")
				self.logger.logger.info(f"relation_multi_label: {relation_multi_label}")
				self.logger.logger.info(
					f"[SIZE] BCE(predict_re, relation_multi_label): {BCE(predict_re, relation_multi_label).size()}")
				self.logger.logger.info(
					f"BCE(predict_re, relation_multi_label): {BCE(predict_re, relation_multi_label)}")
				self.logger.logger.info(f"relation_mask.unsqueeze(2): {relation_mask.unsqueeze(2)}")
				self.logger.logger.info(f"[SIZE] relation_mask.unsqueeze(2): {relation_mask.unsqueeze(2).size()}")
				self.logger.logger.info(
					f"[SIZE] masked BCE*relation_mask.unsqueeze(2): {(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)).size()}")
				self.logger.logger.info(
					f"masked BCE*relation_mask.unsqueeze(2): {BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)}")
				self.logger.logger.info(
					f"torch.sum(masked BCE*relation_mask.unsqueeze(2)): {torch.sum(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2))}")
				self.logger.logger.info(f"[SIZE] torch.sum(relation_mask): {torch.sum(relation_mask).size()}")
				self.logger.logger.info(f"torch.sum(relation_mask): {torch.sum(relation_mask)}")
				self.logger.logger.info(
					f"[SIZE] self.relation_num * torch.sum(relation_mask): {self.relation_num * torch.sum(relation_mask)}")
				self.logger.logger.info(f"loss: {loss}")
				"""

				# self.logger.logger.info(f"Size of predictions for batch {batch}: {output.shape}")

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				relation_label = relation_label.data.cpu().numpy()

				#for i in range(output.size()[0]):
					# for j in range(output.size()[1]):
				# BiLSTM, ContextAware, LSTM, CNN3 with cpu computation
				for i in range(output.shape[0]):
					for j in range(output.shape[1]):
						label = relation_label[i][j]
						if label < 0:
							break

						if label == 0:
							self.acc_NA.add(output[i][j] == label)
						else:
							self.acc_not_NA.add(output[i][j] == label)

						self.acc_total.add(output[i][j] == label)

				global_step += 1
				total_loss += loss.item()

				if global_step % self.period == 0:
					cur_loss = total_loss / self.period
					elapsed = time.time() - start_time
					self.logger.logger.info('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
					total_loss = 0
					start_time = time.time()

				batch += 1
					
			if self.always_evaluate or self.max_epoch > self.test_epoch and (epoch+1) % self.test_epoch == 0:
				self.logger.logger.info('-' * 89)
				# Saving checkpoints to resume training if something goes wrong
				ckpt_file = os.path.join(self.save_path, "checkpoint.ckpt")
				self.logger.logger.info(f"[CHECKPOINT] Saving model checkpoint at epoch {epoch} into {ckpt_file} ...")
				torch.save(model.state_dict(), ckpt_file)

				eval_start_time = time.time()
				self.logger.logger.info(f"Entering test mode..")
				model.eval()
				self.logger.logger.info(f"TAG: {self.args.tag}")
				f1, auc, pr_x, pr_y, threshold = self.test(model, model_name, tag=self.args.tag)
				self.logger.logger.info(f"Saving threshold ..")
				with open(thresh_path, "a") as file:
					writer = csv.writer(file)
					writer.writerow([epoch, threshold])
				model.train()
				self.logger.logger.info('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
				self.logger.logger.info('-' * 89)


				if f1 > best_f1:
					best_f1 = f1
					best_auc = auc
					best_epoch = epoch
					self.logger.logger.info(f"[BEST] Saving best model (epoch {epoch}) at {self.save_path}/best.ckpt ...")
					path = os.path.join(self.save_path, "best.ckpt")
					torch.save(model.state_dict(), path)
					best_threshold = threshold
			elif self.max_epoch <= self.test_epoch:
				self.logger.logger.info('-' * 89)
				# Saving checkpoints to resume training if something goes wrong
				ckpt_file = os.path.join(self.save_path, "checkpoint.ckpt")
				self.logger.logger.info(f"[CHECKPOINT] Saving model checkpoint at epoch {epoch} into {ckpt_file} ...")
				torch.save(model.state_dict(), ckpt_file)

				eval_start_time = time.time()
				self.logger.logger.info(f"Entering test mode..")
				model.eval()
				self.logger.logger.info(f"TAG: {self.args.tag}")
				f1, auc, pr_x, pr_y, threshold = self.test(model, model_name, tag=self.args.tag)
				self.logger.logger.info(f"Saving threshold ..")
				with open(thresh_path, "a") as file:
					writer = csv.writer(file)
					writer.writerow([epoch, threshold])
				model.train()
				self.logger.logger.info('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
				self.logger.logger.info('-' * 89)

				if f1 > best_f1:
					best_f1 = f1
					best_auc = auc
					best_epoch = epoch
					self.logger.logger.info(
						f"[BEST] Saving best model (epoch {epoch}) at {self.save_path}/best.ckpt ...")
					path = os.path.join(self.save_path, "best.ckpt")
					torch.save(model.state_dict(), path)
					best_threshold = threshold

		# Save last epoch
		self.logger.logger.info(f"Saving last epoch model at {self.save_path}/last.ckpt ...")
		path = os.path.join(self.save_path, "last.ckpt")
		torch.save(model.state_dict(), path)
		# thresholds = [[k, threshs[k]] for k in threshs.keys()]

		# thresh_path = os.path.join(os.path.dirname(self.save_path), f"thresh_{self.args.train_prefix}.csv")
		self.logger.logger.info(f"Append best threshold at {thresh_path}")
		with open(thresh_path, "a") as file:
			writer = csv.writer(file)
			writer.writerow(["best", best_threshold])

		self.logger.logger.info("Finish training")
		self.logger.logger.info(f"Best epoch = {best_epoch} | auc = {best_auc}")
		return None
		# self.logger.logger.info("Storing best result...")
		# self.logger.logger.info("Finish storing")

	def test(self, model, output=False, input_theta=-1, tag="dev", thresh=-1):# save_thresh=False):
		data_idx = 0
		eval_start_time = time.time()
		# test_result_ignore = []
		# total_recall_ignore = 0

		test_result = []
		# parsed_items = {}
		total_recall = 0
		# top1_acc = have_label = 0
		self.logger.logger.info(f"*** Evaluating the model on {self.test_batches} batches of size {self.test_batch_size}")
		batch = 1
		# for batch in tqdm(range(self.test_batches), desc="Evaluating Batches"):
		for data in self.get_test_batch():
			self.logger.logger.info(f"Processing batch {batch} of {self.test_batches} ({'{0:.2f}'.format((batch/self.test_batches)*100)} %)")
			with torch.no_grad():
				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				h_mapping = data['h_mapping']
				t_mapping = data['t_mapping']
				labels = data['labels']
				L_vertex = data['L_vertex']
				input_lengths = data['input_lengths']
				context_ner = data['context_ner']
				context_char_idxs = data['context_char_idxs']
				relation_mask = data['relation_mask']
				ht_pair_pos = data['ht_pair_pos']

				titles = data['titles']
				indexes = data['indexes']

				dis_h_2_t = ht_pair_pos+10
				dis_t_2_h = -ht_pair_pos+10

				predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,
								   h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)

				predict_re = torch.sigmoid(predict_re)

			predict_re = predict_re.data.cpu().numpy()

			for i in range(len(labels)):
				label = labels[i]
				index = indexes[i]


				total_recall += len(label)
				# for l in label.values():
					# if not l:
						# total_recall_ignore += 1

				L = L_vertex[i]
				j = 0

				for h_idx in range(L):
					for t_idx in range(L):
						if h_idx != t_idx:
							# TOP-1 Accuracy
							# r = np.argmax(predict_re[i, j])
							# r = output[1][i][j]
							# if (h_idx, t_idx, r) in label:
								# top1_acc += 1

							# flag = False

							for r in range(1, self.relation_num):
								intrain = False

								if (h_idx, t_idx, r) in label:
									# flag = True
									if label[(h_idx, t_idx, r)]==True:
										intrain = True

								# if float(output[0][i, j]) > input_theta:
									# w = i
								#try:
									# tmp = parsed_items[(titles[i], h_idx, t_idx, r)]
									#parsed_items[(titles[i], h_idx, t_idx, r)] += 1
								#except KeyError:
									# if not intrain:
									# 	test_result_ignore.append( ((h_idx, t_idx, r) in label, float(output[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )
								if tag in ["train", "infer"]:
									test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i, j, r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r))
								elif float(predict_re[i, j, r]) >= thresh:
									test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i, j, r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r))
							# if flag:
								# have_label += 1

							j += 1


			data_idx += 1
			batch += 1

			if data_idx % self.period == 0:
				self.logger.logger.info('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
				eval_start_time = time.time()

		self.logger.logger.info(f"Number of parsed batches: {batch-1}")
		self.logger.logger.info(f"Length of test result: {len(test_result)}")
		if tag == "infer":
			self.logger.logger.info(f"Returning candidates for inference ..")
			cand_output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6],
					   'score': x[-8]} for x in test_result]
			return cand_output
		elif tag == "train":
			self.logger.logger.info(f"Evaluating epoch for training ..")
			# test_result_ignore.sort(key=lambda x: x[1], reverse=True)
			test_result.sort(key=lambda x: x[1], reverse=True)

			self.logger.logger.info(f'total_recall: {total_recall}')
			# plt.xlabel('Recall')
			# plt.ylabel('Precision')
			# plt.ylim(0.2, 1.0)
			# plt.xlim(0.0, 0.6)
			# plt.title('Precision-Recall')
			# plt.grid(True)

			pr_x = []
			pr_y = []
			correct = 0
			w = 0

			if total_recall == 0:
				total_recall = 1  # for test

			for i, item in enumerate(test_result):
				correct += item[0]
				pr_y.append(float(correct) / (i + 1))
				pr_x.append(float(correct) / total_recall)
				if item[1] > input_theta:
					w = i


			pr_x = np.asarray(pr_x, dtype='float32')
			pr_y = np.asarray(pr_y, dtype='float32')
			f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
			f1 = f1_arr.max()
			f1_pos = f1_arr.argmax()
			theta = test_result[f1_pos][1]

			"""if save_thresh:
				with open(f"{self.save_path}/thresh.csv", "w") as file:
					writer = csv.writer(file)
					writer.writerow(["epoch", "threshold"])
					writer.writerow(["best", theta])"""

			if input_theta==-1:
				w = f1_pos
				input_theta = theta

			auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
			if not self.is_test:
				self.logger.logger.info('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
			else:
				self.logger.logger.info('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

			if output and self.args.eval_mode == "micro":
				# output = [x[-4:] for x in test_result[:w+1]]
				output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6], 'score': x[-8]} for x in test_result[:w+1]]
				prefix = self.test_prefix if self.test_prefix not in self.pref2name.keys() else self.pref2name[
					self.test_prefix]
				self.logger.logger.info(f"Saving output predictions, number of predictions: {len(output)}")
				self.logger.logger.info(f"Saving output predictions at {self.save_path}/{prefix}_results.json ...")
				json.dump(output, open(f"{self.save_path}/{prefix}_results.json", "w"))
				self.logger.logger.info(f"End of saving file")

			pr_x = []
			pr_y = []
			correct = correct_in_train = 0
			w = 0
			for i, item in enumerate(test_result):
				correct += item[0]
				if item[0] & item[2]:
					correct_in_train += 1
				if correct_in_train == correct:
					p = 0
				else:
					p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
				pr_y.append(p)
				pr_x.append(float(correct) / total_recall)
				if item[1] > input_theta:
					w = i

			pr_x = np.asarray(pr_x, dtype='float32')
			pr_y = np.asarray(pr_y, dtype='float32')
			f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
			f1 = f1_arr.max()

			auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

			self.logger.logger.info('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))
			return f1, auc, pr_x, pr_y, input_theta
		else:
			self.logger.logger.info(f"Performing evaluation only ..")
			test_result.sort(key=lambda x: x[1], reverse=True)
			self.logger.logger.info(f"Computing official results..")
			official_results = [{'h_idx': x[-3], 't_idx': x[-2], 'r': x[-5], 'title': x[-6], 'score': x[-8]} for x in test_result]
			if output and self.args.eval_mode in ["micro", "preds"]:
				prefix =  self.test_prefix if self.test_prefix not in self.pref2name.keys() else self.pref2name[self.test_prefix]
				self.logger.logger.info(f"Saving output predictions, number of predictions: {len(official_results)}")
				self.logger.logger.info(f"Saving output predictions at {self.save_path}/{prefix}_results.json ...")
				json.dump(official_results, open(f"{self.save_path}/{prefix}_results.json", "w"))
				self.logger.logger.info(f"End of saving file")

			if self.args.eval_mode == "preds":
				return official_results

			if len(official_results) > 0:
				if self.args.eval_mode == "micro":
					self.logger.logger.info(f"Perform official MICRO evaluation for file: {self.args.test_file}")
					best_re, best_evi, best_re_ign, _ = official_evaluate(self.args.logger, official_results, self.args.data_dir,
																		  self.args.train_file,
																		  self.args.test_file)
				elif self.args.eval_mode == "macro":
					self.logger.logger.info(f"Perform official MACRO evaluation for file: {self.args.test_file}")
					return official_evaluate_long_tail(self.args, self.logger, official_results, self.args.data_dir, self.args.test_file)
				else:
					self.logger.logger.info(f"Perform official per-relation evaluation for file: {self.args.test_file}")
					return official_evaluate_per_rel(self.logger, official_results, self.args.data_dir, self.args.train_file,
													   self.args.test_file)
			else:
				best_re = best_evi = best_re_ign = [-1, -1, -1]
			scores = {
				tag + "_rel": [i * 100 for i in best_re],
				tag + "_rel_ign": [i * 100 for i in best_re_ign],
				tag + "_evi": [i * 100 for i in best_evi],
			}
			return scores

	def testall(self, model_pattern, input_theta, tag="dev", thresh=-1): # save_thresh=False):#, ignore_input_theta):
		model = model_pattern(config=self, my_logger=self.logger)
		# self.logger.logger.info(f"Model will be loaded to {self.device}")
		self.logger.logger.info(f"Loading best model from {self.save_path}/best.ckpt ..")
		model.load_state_dict(torch.load(os.path.join(self.save_path, "best.ckpt"), map_location=self.device))
		self.logger.logger.info(f"Model will be loaded to {self.device}")
		model.to(self.device)
		model.eval()
		scores = self.test(model, True, input_theta, tag, thresh) # save_thresh=save_thresh)
		return scores

	def infer_scores(self, model_pattern):#, ignore_input_theta):
		model = model_pattern(config=self, my_logger=self.logger)
		self.logger.logger.info(f"Loading best model from {self.save_path}/best.ckpt ..")
		model.load_state_dict(torch.load(os.path.join(self.save_path, "best.ckpt"), map_location=self.device))
		self.logger.logger.info(f"Model will be loaded to {self.device}")
		model.to(self.device)
		model.eval()
		output = self.test(model, tag="infer")
		if self.args.isDistant:
			prefix = f"{self.args.distant_prefix}_sample"
		else:
			prefix = self.test_prefix if self.test_prefix not in self.pref2name.keys() else self.pref2name[self.test_prefix]
		self.logger.logger.info(f"Saving candidates predictions, number of predictions: {len(output)}")
		self.logger.logger.info(f"Saving candidates predictions at {self.save_path}/{prefix}_results.json ...")
		json.dump(output, open(f"{self.save_path}/{prefix}_results.json", "w"))
		return
