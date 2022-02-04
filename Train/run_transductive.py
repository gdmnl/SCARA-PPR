# Ref: https://github.com/chennnM/GBP
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from utils import muticlass_f1, get_max_memory_bytes, f1_score
from logger import Logger, ModelLogger, prepare_opt
from load_transductive import load_transductive_data
from model import GnnBP


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=0, help='random seed.')
# parser.add_argument('--data', default='reddit', help='dateset')
# parser.add_argument('--algo', default='featpush', help='algorithm')
# parser.add_argument('--epochs', type=int, default=200, help='number of epochs.')
# parser.add_argument('--patience', type=int, default=200, help='patience')
# parser.add_argument('--batch', type=int, default=820, help='batch size')
# parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')
# parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay.')
# parser.add_argument('--layer', type=int, default=2, help='number of layers.')
# parser.add_argument('--hidden', type=int, default=128, help='hidden dimensions.')
# parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
# parser.add_argument('--bias', default='none', help='bias.')
# parser.add_argument('--alpha', type=float, default=0.5, help='decay factor')
# parser.add_argument('--eps', type=float, default=8, help='relative error epsilon.')
# parser.add_argument('--rrz', type=float, default=0.5, help='r.')
parser.add_argument('-c', '--config', default='./config/reddit_featpush.json', help='config path.')
parser.add_argument('-v', '--dev', type=int, default=-1, help='device id')
args = prepare_opt(parser)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    torch.cuda.manual_seed(args.seed)

print('-' * 20)
flag_run = "L{}_e{}_{}".format(args.layer, args.eps, args.seed)
logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger, state_only=True)

features, labels, idx_train, idx_val, idx_test = load_transductive_data(args.algo,
            datastr=args.data, alpha=args.alpha, eps=args.eps,
            rrz=args.rrz, seed=args.seed)

model = GnnBP(nfeat=features.shape[1],
            nlayers=args.layer,
            nhidden=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            bias = args.bias)
model_logger.regi_model(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=20, verbose=False)
loss_fn = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(features[idx_train], labels[idx_train])
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=args.batch,
                         shuffle=True,
                         num_workers=0)
ds_val = Data.TensorDataset(features[idx_val], labels[idx_val])
loader_val = Data.DataLoader(dataset=ds_val,
                        batch_size=args.batch,
                        shuffle=False,
                        num_workers=0)
ds_test = Data.TensorDataset(features[idx_test], labels[idx_test])
loader_test = Data.DataLoader(dataset=ds_test,
                        batch_size=args.batch,
                        shuffle=False,
                        num_workers=0)

def train():
    model.train()
    loss_list = []
    time_epoch = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        if args.dev >= 0:
            batch_x = batch_x.cuda(args.dev)
            batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list), time_epoch

def validate(idx=idx_val):
    model.eval()
    with torch.no_grad():
        if args.dev >= 0:
            output = model(features[idx].cuda(args.dev))
        else:
            output = model(features[idx])
        micro_val = muticlass_f1(output, labels[idx]).item()
        return micro_val

def validate_batch(ld=loader_val):
    model.eval()
    output_list, labels_list = None, None
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(ld):
            if args.dev >= 0:
                batch_x = batch_x.cuda(args.dev)
                # batch_y = batch_y.cuda(args.dev)
            output = model(batch_x)
            output = output.max(1)[1]
            if output_list is None:
                output_list = output.cpu().detach().numpy()
                labels_list = batch_y.cpu().detach().numpy()
            else:
                output_list = np.append(output_list, output.cpu().detach().numpy(), axis=0)
                labels_list = np.append(labels_list, batch_y.cpu().detach().numpy(), axis=0)
        micro_val = f1_score(labels_list, output_list, average='micro')
        return micro_val

def test(idx=idx_test):
    model = model_logger.load_model('best')
    if args.dev >= 0:
        model = model.cuda(args.dev)
    model.eval()
    with torch.no_grad():
        if args.dev >= 0:
            output = model(features[idx].cuda(args.dev))
        else:
            output = model(features[idx])
        micro_test = muticlass_f1(output, labels[idx]).item()
        return micro_test

def test_batch(ld=loader_test):
    model.eval()
    output_list, labels_list = None, None
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(ld):
            if args.dev >= 0:
                batch_x = batch_x.cuda(args.dev)
                # batch_y = batch_y.cuda(args.dev)
            output = model(batch_x)
            output = output.max(1)[1]
            if output_list is None:
                output_list = output.cpu().detach().numpy()
                labels_list = batch_y.cpu().detach().numpy()
            else:
                output_list = np.append(output_list, output.cpu().detach().numpy(), axis=0)
                labels_list = np.append(labels_list, batch_y.cpu().detach().numpy(), axis=0)
        micro_test = f1_score(labels_list, output_list, average='micro')
        return micro_test

print('-' * 20)
print('Start training...')
train_time = 0
bad_counter = 0

for epoch in range(args.epochs):
    loss_tra, train_ep = train()
    train_time += train_ep
    acc_val = validate_batch()
    # scheduler.step(acc_val)
    if (epoch+1) % 1 == 0:
        res = f"Epoch:{epoch+1:04d} | train loss:{loss_tra:.4f}, val acc:{acc_val:.4f}, cost:{train_time:.4f}"
        logger.print(res)
    is_best = model_logger.save_best(acc_val, epoch=epoch)
    if is_best:
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break

acc_train = test_batch(ld=loader)
print(f"Train time cost: {train_time:0.4f}")
print(f"Train best acc: {acc_train:0.4f}, Val best acc: {model_logger.acc_best:0.4f}")

print("Start inference...")
start = time.time()
acc_test = test_batch()
time_inference = time.time() - start
memory = get_max_memory_bytes()
print(f"Test cost: {time_inference:0.4f}s, Memory: {memory / 2**20:.3f}GB")

print(f'Best epoch: {model_logger.epoch_best}, Test acc: {acc_test:.4f}')
print("--------------------------")
