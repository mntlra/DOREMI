import numpy as np
import os
import json
from nltk.tokenize import WordPunctTokenizer
import argparse
from Logger import Logger
import datetime
import math

"""
Prepro code from DocRED GitHub repository. (https://github.com/thunlp/DocRED/tree/master)
Data pre-processing for training and testing CNN3, ContextAware, LSTM, and BiLSTM.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="../data/docred/")
parser.add_argument('--out_path', type=str, default="../data/docred/prepro_data")
parser.add_argument("--filename", type=str, default="")
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--train_suffix", type=str, default="")
parser.add_argument("--isTraining", action="store_true")
parser.add_argument("--isDistant", action="store_true")
parser.add_argument('--num_chunks', type=int, default=100, help="Number of chunks for inference on the distant dataset.")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

dev_eval_file = os.path.join(in_path, 'dev_eval.json')
dev_sample_file = os.path.join(in_path, 'dev_sample.json')

rel2id = json.load(open(os.path.join(in_path, 'meta/rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])

def init(data_file_name, rel2id, logger, max_length = 512, is_training = True, suffix='', train_suffix = ""):

	ori_data = json.load(open(data_file_name))

	logger.logger.info(f"is_training: {is_training}")

	Ma = 0
	Ma_e = 0
	data = []
	intrain = notintrain = notindevtrain = indevtrain = 0
	for i in range(len(ori_data)):
		Ls = [0]
		L = 0
		for x in ori_data[i]['sents']:
			L += len(x)
			Ls.append(L)

		vertexSet =  ori_data[i]['vertexSet']
		# point position added with sent start position
		for j in range(len(vertexSet)):
			for k in range(len(vertexSet[j])):
				vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

				sent_id = vertexSet[j][k]['sent_id']
				dl = Ls[sent_id]
				pos1 = vertexSet[j][k]['pos'][0]
				pos2 = vertexSet[j][k]['pos'][1]
				vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

		ori_data[i]['vertexSet'] = vertexSet

		item = {}
		item['vertexSet'] = vertexSet

		labels = ori_data[i].get('labels', [])

		# logger.logger.info(f"Labels: {labels}")

		train_triple = set([])
		new_labels = []
		for label in labels:
			rel = label['r']
			assert(rel in rel2id)
			label['r'] = rel2id[label['r']]

			train_triple.add((label['h'], label['t']))


			if suffix=='_train':
				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						fact_in_dev_train.add((n1['name'], n2['name'], rel))


			if is_training:
				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						fact_in_train.add((n1['name'], n2['name'], rel))
			else:
				logger.logger.info("Not training")
				# fix a bug here
				label['intrain'] = False
				label['indev_train'] = False
				# logger.logger.info(f"vertexSet: {vertexSet}")
				label_h = label['h']
				label_t = label['t']
				# logger.logger.info(f"label[h]: {label_h}")
				# logger.logger.info(f"label[t]: {label_t}")

				for n1 in vertexSet[label['h']]:
					for n2 in vertexSet[label['t']]:
						if (n1['name'], n2['name'], rel) in fact_in_train:
							label['intrain'] = True

						if suffix == '_dev' or suffix == '_test':
							if (n1['name'], n2['name'], rel) in fact_in_dev_train:
								label['indev_train'] = True

						if train_suffix != "":
							if (n1['name'], n2['name'], rel) in fact_in_train:
								label[f'in{train_suffix}'] = True

			new_labels.append(label)

		"""logger.logger.info(f"Length new labels: {len(new_labels)}")
		if len(new_labels) > 0:
			logger.logger.info(f"Labels have keys: {new_labels[0].keys()}")"""
		item['labels'] = new_labels
		item['title'] = ori_data[i]['title']

		na_triple = []
		for j in range(len(vertexSet)):
			for k in range(len(vertexSet)):
				if (j != k):
					if (j, k) not in train_triple:
						na_triple.append((j, k))

		item['na_triple'] = na_triple
		item['Ls'] = Ls
		item['sents'] = ori_data[i]['sents']
		if "include_pairs" in ori_data[i].keys():
			item["include_pairs"] = ori_data[i]["include_pairs"]
		data.append(item)

		Ma = max(Ma, len(vertexSet))
		Ma_e = max(Ma_e, len(item['labels']))


	logger.logger.info(f'{data_file_name} data_len: {len(ori_data)}')


	# saving
	logger.logger.info(f"Saving files for {data_file_name}")
	"""if is_training:
		name_prefix = "train"
	else:
		name_prefix = "dev"""

	json.dump(data, open(os.path.join(out_path, suffix + '.json'), "w"))

	char2id = json.load(open(os.path.join(in_path, "meta/char2id.json")))
	# id2char= {v:k for k,v in char2id.items()}
	# json.dump(id2char, open("data/id2char.json", "w"))

	word2id = json.load(open(os.path.join(in_path, "meta/word2id.json")))
	ner2id = json.load(open(os.path.join(in_path, "meta/ner2id.json")))

	sen_tot = len(ori_data)
	sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
	sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
	sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
	sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)

	for i in range(len(ori_data)):
		item = ori_data[i]
		words = []
		for sent in item['sents']:
			words += sent

		for j, word in enumerate(words):
			word = word.lower()

			if j < max_length:
				if word in word2id:
					sen_word[i][j] = word2id[word]
				else:
					sen_word[i][j] = word2id['UNK']

				for c_idx, k in enumerate(list(word)):
					if c_idx>=char_limit:
						break
					sen_char[i,j,c_idx] = char2id.get(k, char2id['UNK'])

		for j in range(j + 1, max_length):
			sen_word[i][j] = word2id['BLANK']

		vertexSet = item['vertexSet']

		for idx, vertex in enumerate(vertexSet, 1):
			for v in vertex:
				sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
				sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

	logger.logger.info(f"Finishing processing {data_file_name}")
	np.save(os.path.join(out_path, suffix + '_word.npy'), sen_word)
	np.save(os.path.join(out_path, suffix + '_pos.npy'), sen_pos)
	np.save(os.path.join(out_path, suffix + '_ner.npy'), sen_ner)
	np.save(os.path.join(out_path, suffix + '_char.npy'), sen_char)
	logger.logger.info(f"Finish saving {data_file_name}")

def main():
	logger_name = str(datetime.datetime.now()).replace(' ', '_')
	my_logger = Logger(f"../data/output/logs/prepro/{logger_name}.log")
	if args.isDistant:
		data = json.load(open(os.path.join(args.in_path, args.filename), "r"))
		my_logger.logger.info(f"Processing {args.filename} containing {len(data)} documents")
		sample_size = math.ceil(len(data) / args.num_chunks)
		# sample_num = math.ceil(len(data) / 382)
		# my_logger.logger.info(f"splitting the data into 4 samples of {sample_size} documents")
		my_logger.logger.info(f"splitting the data into {args.num_chunks} samples of {sample_size} documents")
		doc_count = 0
		sample = {}
		for i in range(1, args.num_chunks+1):
			sample[i] = []
		# sample = {1: [], 2: [], 3: [], 4: []}
		for document in data:
			doc_count += 1

			if doc_count % sample_size == 0:
				s = int(doc_count/sample_size)
			else:
				s = int(doc_count/sample_size) + 1
			sample[s].append(document)
		# save_path = os.path.join(args.in_path, args.filename)
		sample_suffix = args.filename.split(".")[0]
		# Saving samples
		for s in sample.keys():
			my_logger.logger.info(f"Saving sample {s} of length {len(sample[s])} in "
								  f"{args.in_path}/{sample_suffix}_sample{s}.json")
			json.dump(sample[s], open(f"{args.in_path}/{sample_suffix}_sample{s}.json", "w"))
			my_logger.logger.info(f"Selecting title2pairs for sample {s}")
			t2p_save_path = os.path.join(args.in_path, f"{sample_suffix}_sample{s}_title2pairs.json")
			titles2pairs = json.load(open(f"{args.in_path}/{sample_suffix}_title2pairs.json"))
			t2p = {}
			for doc in sample[s]:
				t2p[doc["title"]] = titles2pairs[doc["title"]]
			my_logger.logger.info(f"Saving title2pairs for sample {s} at {t2p_save_path}")
			json.dump(t2p, open(t2p_save_path, "w"))

		ori_suffix = args.suffix
		# prepro each sample
		# for i in range(1, 5):
		for i in range(1, args.num_chunks+1):
			sample_name = f"{sample_suffix}_sample{i}.json"
			args.suffix = f"{ori_suffix}_sample{i}"
			my_logger.logger.info(f"Processing {sample_name} with suffix {args.suffix}")
			init(os.path.join(args.in_path, sample_name), rel2id, my_logger, max_length=512, is_training=False,
				 suffix=args.suffix, train_suffix=args.train_suffix)
	else:
		if args.filename != "":
			if args.suffix != "":
				my_logger.logger.info(f"Processing {args.filename} with suffix {args.suffix}")
				init(os.path.join(args.in_path, args.filename), rel2id, my_logger, max_length=512, is_training=args.isTraining,
					 suffix=args.suffix, train_suffix=args.train_suffix)
			else:
				raise ValueError(f"When you specify a filename you must also specify a suffix!")
		else:
			raise ValueError(f"No filename given!")

if __name__ == "__main__":
	main()