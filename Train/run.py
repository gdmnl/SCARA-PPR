# Ref: https://github.com/chennnM/GBP
import time
import random
import argparse
import resource
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from logger import Logger, ModelLogger, prepare_opt
from loader import load_data
from model import MLP


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=0, help='random seed.')
parser.add_argument('-c', '--config', default='./config/reddit.json', help='config path.')
parser.add_argument('-v', '--dev', type=int, default=-1, help='device id.')
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

feat, labels, idx = load_data(args.algo, datastr=args.data, datapath=args.path,
            inductive=args.inductive, multil=args.multil, spt=args.spt,
            alpha=args.alpha, eps=args.eps, rrz=args.rrz, seed=args.seed)
nclass = labels.shape[1] if args.multil else int(labels.max()) + 1

model = MLP(nfeat=feat['train'].shape[1],
            nlayers=args.layer,
            nhidden=args.hidden,
            nclass=nclass,
            dropout=args.dropout,
            bias = args.bias)
model_logger.regi_model(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=20, verbose=False)
loss_fn = nn.BCEWithLogitsLoss() if args.multil else nn.CrossEntropyLoss()

ds_train = Data.TensorDataset(feat['train'], labels[idx['train']])
loader_train = Data.DataLoader(dataset=ds_train, batch_size=args.batch,
                               shuffle=True, num_workers=0)
ds_val = Data.TensorDataset(feat['val'], labels[idx['val']])
loader_val = Data.DataLoader(dataset=ds_val, batch_size=args.batch,
                             shuffle=False, num_workers=0)
ds_test = Data.TensorDataset(feat['test'], labels[idx['test']])
loader_test = Data.DataLoader(dataset=ds_test, batch_size=args.batch,
                              shuffle=False, num_workers=0)


def train(ld=loader_train):
    model.train()
    loss_list = []
    time_epoch = 0
    for _, (batch_x, batch_y) in enumerate(ld):
        if args.dev >= 0:
            batch_x = batch_x.cuda(args.dev)
            batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_batch = loss_fn(output, batch_y)
        loss_batch.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_batch.item())
    return np.mean(loss_list), time_epoch


def eval(ld, load_best=False):
    if load_best:
        model = model_logger.load_model('best')
    model.eval()
    micro, num_total = 0, 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(ld):
            output_list, labels_list = None, None
            # if ((step + 1) % (len(ld) // 20) == 0):
            #     print(f'{step + 1} {(step + 1) // (len(ld) // 10):g}: f1 {micro_test / num_test}')
            if args.dev >= 0:
                batch_x = batch_x.cuda(args.dev)
                # batch_y = batch_y.cuda(args.dev)
            output = model(batch_x)
            if not args.multil:
                output = output.max(1)[1]
            output_list = output.cpu().detach().numpy()
            labels_list = batch_y.cpu().detach().numpy()

            if args.multil:
                output_list[output_list > 0] = 1
                output_list[output_list <= 0] = 0
                micro_batch = f1_score(labels_list, output_list, average='micro')
            else:
                micro_batch = f1_score(labels_list, output_list, average='micro')
            micro += micro_batch * len(batch_y)
            num_total += len(batch_y)
        return micro / num_total


print('-' * 20)
# print('Start training...')
train_time = 0
conv_epoch = 0

for epoch in range(args.epochs):
    loss_train, train_ep = train()
    train_time += train_ep
    acc_val = eval(ld=loader_val)
    scheduler.step(acc_val)
    if (epoch+1) % 10 == 0:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, cost:{train_time:.4f}"
        logger.print(res)
    is_best = model_logger.save_best(acc_val, epoch=epoch)
    # Early stop if converge
    if is_best:
        conv_epoch = 0
    else:
        conv_epoch += 1
    if conv_epoch == args.patience:
        break

acc_train = eval(ld=loader_train, load_best=True)
print(f"Train time cost: {train_time:0.4f}")
print(f"Train best acc: {acc_train:0.4f}, Val best acc: {model_logger.acc_best:0.4f}")

print('-' * 20)
# print("Start inference...")
start = time.time()
acc_test = eval(ld=loader_test, load_best=True)
time_inference = time.time() - start
memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Test time cost: {time_inference:0.4f}, Memory: {memory / 2**20:.3f}GB")

print(f'Best epoch: {model_logger.epoch_best}, Test acc: {acc_test:.4f}')
print('-' * 20)
