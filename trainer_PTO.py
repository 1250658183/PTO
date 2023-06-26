import argparse
import collections
import copy
import random
import pandas as pd

seed = random.randint(0, 10000)
import numpy as np
import torch
from tqdm import tqdm
import fitlog

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--num_token", type=int, default=300)
parser.add_argument("--frac", type=int, default=0)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='gpt2')
parser.add_argument("--model_name_or_path", default='gpt2')
parser.add_argument("--gpu_id", type=int, default=5)
parser.add_argument("--early_stop", type=int, default=7)
parser.add_argument("--turn_on_schedule", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=seed)
parser.add_argument("--monitor", type=str, default="auroc")
parser.add_argument("--dataset", type=str, default='clinc', choices=['clinc', 'IMDB'])
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()
print(args)

epochs = 50
import os

log_dir = f"~/PTO_check_logs/PTO_{args.dataset}"

if args.dataset == 'IMDB':
    from ClincDataProcessor import IMDBProcessor as OOD_DataProcessor

    datasets_dir = "./datasets/imdb_yelp"
    max_seq_length = 128
    batch_size = args.batch_size

elif args.dataset == 'clinc':
    from separate_prefix.ClincDataProcessor import ClincProcessor as OOD_DataProcessor

    datasets_dir = "./datasets/clinc150/"
    max_seq_length = 55
    batch_size = args.batch_size

os.makedirs(log_dir, exist_ok=True)
fitlog.set_rng_seed(args.seed)
device = torch.device(f"cuda:{args.gpu_id}")


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


dataset = {}
dataset['train'] = OOD_DataProcessor(True).get_examples(datasets_dir, "train", args.frac)
dataset['val'] = OOD_DataProcessor(True).get_examples(datasets_dir, "valid")
dataset["val_ood"] = OOD_DataProcessor(False).get_examples(datasets_dir, "valid")
dataset['test'] = OOD_DataProcessor(False).get_examples(datasets_dir, "test")

from load_model import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate

mytemplate = PrefixTuningTemplate(model=plm,
                                  tokenizer=tokenizer,
                                  text='{"special": "<eos>"}{"mask"}',
                                  using_decoder_past_key_values=True,
                                  num_token=args.num_token)

print(dataset["train"][0])
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

# load the pipeline model PromptForGeneration.
from PromptForLM import PromptLM

use_cuda = True
prompt_model = PromptLM(plm=plm, template=mytemplate, freeze_plm=True, tokenizer=tokenizer,
                        plm_eval_mode=args.plm_eval_mode)
if use_cuda: prompt_model = prompt_model.to(device)

if args.dataset == 'IMDB': from separate_prefix.tokenizer_wrapper import IMDBLMTokenizerWrapper as WrapperClass
from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
                                    decoder_max_length=max_seq_length,
                                    batch_size=batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                    # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
                                    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["val"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
                                         decoder_max_length=max_seq_length,
                                         batch_size=batch_size, shuffle=False, teacher_forcing=True,
                                         predict_eos_token=True,
                                         truncate_method="head")

val_ood_dataloader = PromptDataLoader(dataset=dataset["val_ood"], template=mytemplate, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
                                      decoder_max_length=max_seq_length,
                                      batch_size=batch_size, shuffle=False, teacher_forcing=True,
                                      predict_eos_token=True,
                                      truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length,
                                   decoder_max_length=max_seq_length,
                                   batch_size=32, shuffle=False, teacher_forcing=True, predict_eos_token=True,
                                   truncate_method="head")

fitlog.set_log_dir(log_dir)
fitlog.add_hyper(vars(args))

from transformers import AdamW

# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in prompt_model.named_parameters() if
                   (not any(nd in n for nd in no_decay)) and p.requires_grad],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in prompt_model.named_parameters() if
                   any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    }
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, len(train_dataloader) // 2, tot_step)

import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def get_ppl(prompt_model, dataloader):
    prompt_model.eval()
    # mytemplate.eval()
    total_ppl = []
    total_loss = []
    funct = nn.NLLLoss(reduction="mean")  # -sum log_i

    for step, inputs in enumerate(tqdm(dataloader)):
        if use_cuda:
            inputs = inputs.to(device)

        shift_logits, shift_label = prompt_model.get_shift_logits_and_labels(inputs)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))
        total_loss.append(loss.item())
        shift_probs = F.log_softmax(shift_logits, dim=-1)
        for p, l in zip(shift_probs, shift_label):
            ppl = torch.exp(funct(p, l)).item()
            total_ppl.append(ppl)

    return np.mean(total_ppl), np.mean(total_loss)


import sklearn.metrics as metrics


def get_auc(y, pred):
    # y: ood is 1，ind is 0
    # pred: ood is larger
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    fpr95 = 1000  # init
    auroc = metrics.auc(fpr, tpr)
    for i in range(len(tpr)):
        if tpr[i] >= 0.95:
            fpr95 = fpr[i]
            break
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=1)
    aupr_out = metrics.auc(recall, precision)

    pred = [-1 * one for one in pred]
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred, pos_label=0)
    aupr_in = metrics.auc(recall, precision)

    return {"auroc": auroc, "fpr95": fpr95, "aupr_out": aupr_out, "aupr_in": aupr_in}


result_2_csv = {'val': collections.defaultdict(list), 'test': collections.defaultdict(list)}


@torch.no_grad()
def get_ood_performance(prompt_model, dataloader, epoch, split):
    prompt_model.eval()
    # mytemplate.eval()
    total_pto_scores = []
    total_lle_scores = []
    total_is_oods = []
    funct = nn.NLLLoss(reduction="sum")  # -sum log_i
    for step, inputs in enumerate(tqdm(dataloader)):
        if use_cuda:
            inputs = inputs.to(device)

        inputs_ori = copy.deepcopy(inputs)

        shift_logits, shift_label = prompt_model.get_shift_logits_and_labels(inputs)
        shift_logits_ori, shift_label_ori = prompt_model.get_plm_shift_logits_and_labels(inputs_ori)
        shift_probs = F.log_softmax(shift_logits, dim=-1)
        shift_probs_ori = F.log_softmax(shift_logits_ori, dim=-1)
        for i, (l, p, p_o) in enumerate(zip(shift_label, shift_probs, shift_probs_ori)):
            sum_neg_log_ind = funct(p, l).item()
            sum_neg_log_x = funct(p_o, l).item()
            total_pto_scores.append(sum_neg_log_ind - sum_neg_log_x)
            total_lle_scores.append(sum_neg_log_ind)
        guids = [int(k) for k in inputs["guid"]]
        is_oods = [dataloader.raw_dataset[k].meta["is_ood"] for k in guids]
        total_is_oods.extend(is_oods)

        result_2_csv[split]['guids'].extend(guids)
    result_2_csv[split]['is_oods'].extend(total_is_oods)
    result_2_csv[split]['pto_scores'].extend(total_pto_scores)
    result_2_csv[split]['lle_scores'].extend(total_lle_scores)
    result_2_csv[split]['epoch'].extend([epoch] * len(total_is_oods))
    return {"pto": get_auc(total_is_oods, total_pto_scores), "lle": get_auc(total_is_oods, total_lle_scores)}


global_step = 0
tot_loss = 0
log_loss = 0
best_monitor = -99999999
best_epoch = 0
best_ppl = 0
best_valid_loss = 0
best_res = None
best_valid_res = None
best_parameters = None
for epoch in range(epochs):
    prompt_model.train()
    # mytemplate.train()
    epoch_step = 0
    epoch_loss = 0
    print(f"======epoch:{epoch + 1} begin=====")
    for step, inputs in enumerate(tqdm(train_dataloader)):
        global_step += 1
        epoch_step += 1
        if use_cuda:
            inputs = inputs.to(device)
        # logits, labels = prompt_model.get_shift_logits_and_labels(inputs)
        loss = prompt_model(inputs)
        loss.backward()
        epoch_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        optimizer.step()
        if args.turn_on_schedule:
            scheduler.step()
        optimizer.zero_grad()
    print(" loss:", epoch_loss / epoch_step)
    valid_ppl, valid_loss = get_ppl(prompt_model, validation_dataloader)
    valid_ppl, valid_loss = get_ppl(prompt_model, val_ood_dataloader)
    print(" valid ppl: ", valid_ppl)
    valid_ood_res = get_ood_performance(prompt_model, val_ood_dataloader, epoch, 'val')
    print(" valid ood:", valid_ood_res)

    fitlog.add_loss(value=epoch_loss / epoch_step, name="train loss", step=global_step, epoch=epoch + 1)
    fitlog.add_metric(value=valid_ppl, name="valid ppl", step=global_step, epoch=epoch + 1)
    fitlog.add_metric(value=valid_loss, name="valid loss", step=global_step, epoch=epoch + 1)
    fitlog.add_metric(value=valid_ood_res, name="valid ood res", step=global_step, epoch=epoch + 1)

    if args.monitor == "auroc":
        monitor = valid_ood_res["pto"]["auroc"]
    elif args.monitor == "val_loss":
        monitor = -1 * valid_loss

    test_ood_res = get_ood_performance(prompt_model, test_dataloader, epoch, 'test')
    print(" test ood: ", test_ood_res)
    fitlog.add_metric(value=test_ood_res, name="test ood res", step=global_step, epoch=epoch + 1)

    if monitor > best_monitor:
        best_monitor = monitor
        best_epoch = epoch
        best_res = test_ood_res
        best_ppl = valid_ppl
        best_valid_loss = valid_loss
        best_valid_res = valid_ood_res
        best_parameters = copy.deepcopy(mytemplate.state_dict())
        if best_epoch < 15:
            early_stop = args.early_stop + (5 if args.dataset == 'clinc' else 0)
        else:
            early_stop = args.early_stop
    if epoch - best_epoch == early_stop:
        break

print(f'best epoch is {best_epoch}')

torch.save(best_parameters, f"{log_dir}/{seed}_best_template.csv")
for k, v in result_2_csv.items():
    df = pd.DataFrame(v)
    df.to_csv(f"{log_dir}/{seed}_{k}_result.csv")

fitlog.add_best_metric({"test": best_res, "valid": best_valid_res, "valid_ppl": best_ppl, "valid_loss": best_valid_loss,
                        "epoch": best_epoch})
fitlog.finish()
