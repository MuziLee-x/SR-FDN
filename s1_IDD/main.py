import torch
import time
import training
import model
import pickle
import utils
import dataloader
import parse1
from parse1 import args, log_file
from prettytable import PrettyTable

utils.set_seed(args.seed)
mem_manager = dataloader.MemLoader(args)
train_dataset = dataloader.Loader(args)

Recmodel = model.LightGCN(train_dataset)
print(Recmodel)

Recmodel = Recmodel.to(parse1.device)
utils.Logging(log_file, str(args))
results = []
args.lr /= 5
opt = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)

# ========== Phase I: Memorization ========== #
for epoch in range(args.epochs):
    time_train = time.time()
    output_information = training.memorization_train(train_dataset, Recmodel, opt)
    train_log = PrettyTable()
    train_log.field_names = ['Epoch', 'Loss', 'Time', 'Estimated Clean Ratio', 'Memory ratio']

    clean_ratio = training.estimate_noise(mem_manager, Recmodel)
    mem_ratio = training.memorization_test(mem_manager, Recmodel)
    train_log.add_row(
        [f'{epoch + 1}/{args.epochs}', output_information, f'{(time.time() - time_train):.3f}',
         f'{clean_ratio:.5f}', f'{mem_ratio:.5f}']
    )
    utils.Logging(log_file, str(train_log))

    # memorization point
    if mem_ratio >= clean_ratio:
        utils.Logging(log_file, f'==================Memorization Point==================')
        break