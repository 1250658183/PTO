import pandas as pd
import fitlog
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("--ood_num", type=int, default=250)
parser.add_argument("--seed", type=int, default=8187)
parser.add_argument("--ood_seed", type=int, default=1111)
parser.add_argument("--monitor", type=str, default="auroc")
parser.add_argument("--dataset", type=str, default='clinc', choices=['clinc', 'IMDB'])
parser.add_argument("--labeled", type=bool, default=True)
args = parser.parse_args()

if args.labeled:
    log = f'~/PTO_check_logs/PTO_Label_OOD_{args.dataset}'
else:
    log = f'~/PTO_check_logs/PTO_OOD_{args.dataset}'

if args.dataset == 'IMDB':
    if args.labeled:
        datasets_dir = "~/PTO_check_logs/PTO_Labed_IMDB"
    else:
        datasets_dir = "~/PTO_check_logs/PTO_IMDB"

    ood_datasets_dir = "~/PTO_check_logs/OOD/imdb_yelp_back_scores_128"  # This file needs to be trained independently
    max_seq_length = 128
    batch_size = 64

elif args.dataset == 'clinc':
    if args.labeled:
        datasets_dir = "~/PTO_check_logs/PTO_Labed_clinc"
    else:
        datasets_dir = "~/PTO_check_logs/PTO_clinc"
    ood_datasets_dir = "~/PTO_check_logs/OOD/clinc150_back_scores"  # This file needs to be trained independently
    max_seq_length = 55
    batch_size = 32

import os

log_dir = f"{log}"
if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)

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


@torch.no_grad()
def get_ood_performance_gather(lle_dataloader, ood_dataloader, epoch, split):
    total_life_scores = []
    total_lle_scores = []
    total_is_oods = []
    for guid in tqdm(lle_dataloader['guid'].unique()):
        step_lle_scores_value = lle_dataloader[(lle_dataloader['guid'] == guid)]['lle_scores'].values[0]
        step_life_scores_value = step_lle_scores_value - \
                                 ood_dataloader[(ood_dataloader['id'] == guid)]['back_scores'].values[0]

        total_life_scores.append(step_life_scores_value)
        total_lle_scores.append(step_lle_scores_value)
        total_is_oods.append(lle_dataloader[(lle_dataloader['guid'] == guid)]['is_oods'].values[0])

    return {"life": get_auc(total_is_oods, total_life_scores), "lle": get_auc(total_is_oods, total_lle_scores)}


if __name__ == '__main__':
    valid_path, test_path = f'{datasets_dir}/{args.seed}_val_result.csv', f'{datasets_dir}/{args.seed}_test_result.csv'
    ood_valid_path, ood_test_path = f'{ood_datasets_dir}/seed_{args.ood_seed}_train_num_{args.ood_num}_val.csv', f'{ood_datasets_dir}/seed_{args.ood_seed}_train_num_{args.ood_num}_test.csv'
    valid_lle_dataloader, valid_ood_dataloader = pd.read_csv(valid_path), pd.read_csv(ood_valid_path)
    test_lle_dataloader, test_ood_dataloader = pd.read_csv(test_path), pd.read_csv(ood_test_path)

    fitlog.set_log_dir(log_dir)
    fitlog.add_hyper(vars(args))
    best_monitor = 0.0

    epochs = valid_lle_dataloader['epoch'].unique()
    for epoch in epochs:
        print(f"======epoch:{epoch + 1} begin=====")
        valid_ood_res = get_ood_performance_gather(
            valid_lle_dataloader.loc[valid_lle_dataloader['epoch'].isin([epoch])], valid_ood_dataloader, epoch, 'val')
        print(" valid ood:", valid_ood_res)
        fitlog.add_metric(value=valid_ood_res, name="valid ood res", step=0, epoch=int(epoch) + 1)
        test_ood_res = get_ood_performance_gather(test_lle_dataloader.loc[test_lle_dataloader['epoch'].isin([epoch])],
                                                  test_ood_dataloader, epoch, 'test')
        print(" test ood: ", test_ood_res)
        fitlog.add_metric(value=test_ood_res, name="test ood res", step=0, epoch=int(epoch) + 1)

        monitor = valid_ood_res["life"]["auroc"]
        if monitor > best_monitor:
            best_monitor = monitor
            best_epoch = epoch
            fitlog.add_best_metric({"test": test_ood_res, "valid": valid_ood_res, "epoch": best_epoch})
