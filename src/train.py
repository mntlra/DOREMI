import csv
import os
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils.utils import collate_fn
from evaluation import to_official, official_evaluate, get_thresh_preds, get_preds
from tqdm import tqdm
from utils.utils import dump_to_file
from evaluation_relationwise import official_evaluate_per_rel, official_evaluate_long_tail

"""
Train and Test script for BERT model.
"""

def load_input(batch, device, tag="train", args=None):
    input = {
        'input_ids': batch[0].to(device),
        'attention_mask': batch[1].to(device),
        'labels': batch[2].to(device),
        'entity_pos': batch[3],
        'hts': batch[4],
        'relation_mask': batch[5].to(device) if args.exclude_pairs else None,
        'tag': tag
        }

    return input


def train_bert(args, model, train_features, dev_features):
    def finetune(args, features, optimizer, num_epoch, num_steps):
        best_score = -1
        best_threshold = 0.0
        thresh_path = os.path.join(args.save_path, "thresh.csv")
        with open(thresh_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "threshold"])
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        scaler = GradScaler()
        args.logger.logger.info("Total steps: {}".format(total_steps))
        args.logger.logger.info("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator, desc='Train epoch'):
            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                optimizer.zero_grad()
                model.train()

                inputs = load_input(batch, args.device, args=args)
                """labels = inputs["labels"]
                args.logger.logger.info(
                    f"[SIZE] Size of labels: {labels.size()}")"""
                outputs = model(**inputs)

                loss = outputs["loss"] / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if (step + 1) == len(train_dataloader) or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):

                    dev_scores, dev_output, official_results, threshold = evaluate(args, model, dev_features, tag="dev")
                    args.logger.logger.info(f"Saving threshold ..")
                    with open(thresh_path, "a") as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, threshold])

                    args.logger.logger.info(dev_output)
                    # Saving checkpoints to resume training if something goes wrong
                    ckpt_file = os.path.join(args.save_path, "checkpoint.ckpt")
                    args.logger.logger.info(f"[CHECKPOINT] Saving model checkpoint at epoch {epoch} into {ckpt_file} ...")
                    torch.save(model.state_dict(), ckpt_file)
                    if dev_scores["dev_F1_ign"] > best_score:
                        best_score = dev_scores["dev_F1_ign"]
                        best_offi_results = official_results
                        best_output = dev_output

                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        args.logger.logger.info(f"Saving best model (epoch: {epoch}) into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)
                        best_threshold = threshold

                    if epoch == train_iterator[-1]:  # last epoch

                        ckpt_file = os.path.join(args.save_path, "last.ckpt")
                        args.logger.logger.info(f"saving model checkpoint into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                        # basename = os.path.splitext(args.dev_file)[0]
                        pred_file = os.path.join(args.save_path, "results.json")
                        score_file = os.path.join(args.save_path, "scores.csv")

                        args.logger.logger.info(f"Saving threshold ..")
                        with open(thresh_path, "a") as file:
                            writer = csv.writer(file)
                            writer.writerow(["best", best_threshold])

                        dump_to_file(args.logger, best_offi_results, pred_file, best_output, score_file)

        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    num_steps = 0
    model.zero_grad()
    finetune(args, train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev", threshold=-1):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    probs, gt = [], []

    """for batch in dataloader:
        args.logger.logger.info(f"[BERT] {batch[1]}")
        break"""

    # model.eval()
    # with torch.no_grad():

    for batch in tqdm(dataloader, desc=f"Evaluating batches"):

        model.eval()
        inputs = load_input(batch, args.device, tag, args=args)
        # args.logger.logger.info(f"[BERT] Shape of batch {len(batch)}")
        # args.logger.logger.info(f"[BERT] Tokenizer input {inputs['input_ids']}")
        # args.logger.logger.info(f"[BERT] Attention mask {inputs['attention_mask']}")
        # args.logger.logger.info(f"[BERT] Shape of tokenizer input {inputs['input_ids'].shape}")
        # args.logger.logger.info(f"[BERT] Shape of attention mask {inputs['attention_mask'].shape}")

        with torch.no_grad():
            outputs = model(**inputs)
            prob = outputs["probs"]
            # hidden_states = outputs.hidden_states
            # args.logger.logger.info(f"[BERT] Length of hidden states {len(hidden_states)}")
            prob = prob.cpu().numpy()
            prob[np.isnan(prob)] = 0
            probs.append(prob)
            gt.append(inputs["labels"].cpu().numpy())
            # TODO new clear memory
            # del outputs # Clear memory


    probs = np.concatenate(probs, axis=0)
    gt = np.concatenate(gt, axis=0)
    if tag == "dev":
        preds, threshold = get_thresh_preds(probs, gt)
    elif tag == "infer":
        # Select all relations as correct
        preds = np.ones(probs.shape)
    else:
        preds = get_preds(probs, threshold)

    official_results = to_official(args.data_dir, preds, probs, features)

    if args.eval_mode == "preds":
        return {}, {}, official_results, threshold

    if tag == "infer":
        return official_results

    if len(official_results) > 0:
        if tag == "dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(args.logger, official_results, args.data_dir, args.train_file,
                                                                  args.dev_file)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(args.logger, official_results, args.data_dir, args.train_file,
                                                                  args.test_file)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign],
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {"dev_F1": best_re[-1] * 100, "dev_evi_F1": best_evi[-1] * 100, "dev_F1_ign": best_re_ign[-1] * 100}

    return scores, output, official_results, threshold

def evaluate_per_rel(args, model, features, tag="dev", threshold=-1):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    probs, gt = [], []

    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()
        inputs = load_input(batch, args.device, tag, args=args)

        with torch.no_grad():
            outputs = model(**inputs)
            prob = outputs["probs"]
            prob = prob.cpu().numpy()
            prob[np.isnan(prob)] = 0
            probs.append(prob)
            gt.append(inputs["labels"].cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    gt = np.concatenate(gt, axis=0)
    if tag == "dev":
        preds, threshold = get_thresh_preds(probs, gt)
    else:
        preds = get_preds(probs, threshold)

    official_results = to_official(args.data_dir, preds, probs, features)

    if len(official_results) > 0:
        if args.eval_mode == "per-relation":
            return official_evaluate_per_rel(args.logger, official_results, args.data_dir, args.train_file,
                                             args.test_file)
        else:
            return official_evaluate_long_tail(args, args.logger, official_results, args.data_dir, args.test_file)
    else:
        args.logger.logger.info(f"official_results has length zero, returning empty dictionary")
        return {}
