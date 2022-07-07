import numpy as np
import torch
import utils
import dataloader
from utils import timer
import model
import multiprocessing
from sklearn.mixture import GaussianMixture as GMM
from parse1 import args, log_file
import parse1

CORES = multiprocessing.cpu_count() // 2


def memorization_train(dataset, recommend_model, opt):
    Recmodel = recommend_model
    Recmodel.train()

    # sampling
    S = utils.UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(parse1.device)
    posItems = posItems.to(parse1.device)
    negItems = negItems.to(parse1.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // args.batch_size + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=args.batch_size)):

        loss, reg_loss = Recmodel.loss(batch_users, batch_pos, batch_neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        aver_loss += loss.cpu().item()

    aver_loss = aver_loss / total_batch
    timer.zero()
    return f"{aver_loss:.5f}"

def estimate_noise(dataset, recommend_model):
    '''
    estimate noise ratio based on GMM
    '''
    Recmodel: model.LightGCN = recommend_model
    Recmodel.eval()

    dataset: dataloader.MemLoader

    # sampling
    S = utils.UniformSample(dataset)
    users_origin = torch.Tensor(S[:, 0]).long()
    posItems_origin = torch.Tensor(S[:, 1]).long()
    negItems_origin = torch.Tensor(S[:, 2]).long()

    users_origin = users_origin.to(parse1.device)
    posItems_origin = posItems_origin.to(parse1.device)
    negItems_origin = negItems_origin.to(parse1.device)
    with torch.no_grad():
        losses = []
        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(utils.minibatch(users_origin,
                                                       posItems_origin,
                                                       negItems_origin,
                                                       batch_size=args.batch_size)):
            loss, _ = Recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
            # concat all losses
            if len(losses) == 0:
                losses = loss
            else:
                losses = torch.cat((losses, loss), dim=0)
        # split losses of each user
        losses_u = []
        st, ed = 0, 0
        for count in dataset.user_pos_counts:
            ed = st + count
            losses_u.append(losses[st:ed])
            st = ed
        # normalize losses of each user
        for i in range(len(losses_u)):
            if len(losses_u[i]) > 1:
                losses_u[i] = (losses_u[i] - losses_u[i].min()) / (losses_u[i].max() - losses_u[i].min())
        losses = torch.cat(losses_u, dim=0)
        losses = losses.reshape(-1, 1).cpu().detach().numpy()
        gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmax()]
        return 1 - np.mean(prob)

def memorization_test(dataset, Recmodel):
    '''
    memorization procedure,
    update memorization history matrix and generate memorized data
    '''
    u_batch_size = args.test_u_batch_size
    with torch.no_grad():
        users = dataset.trainUniqueUsers
        users_list = []
        items_list = []
        S = utils.sample_K_neg(dataset)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(parse1.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            excluded_users = []
            excluded_items = []
            k_list = []
            for range_i, u in enumerate(batch_users):
                neg_items = S[u]
                items = allPos[range_i]
                k_list.append(len(items))
                neg_items.extend(items)
                excluded_items.extend(neg_items)
                excluded_users.extend([range_i] * (len(neg_items)))

            rating[excluded_users, excluded_items] += 100

            # rating_K: [batch_size, K]
            max_K = max(k_list)
            _, rating_K = torch.topk(rating, k=max_K)
            for i in range(len(rating_K)):
                user = batch_users[i]
                items = rating_K[i].tolist()[:k_list[i]]
                users_list.extend([user] * len(items))
                items_list.extend(items)
            try:
                assert len(users_list) == len(items_list)
            except AssertionError:
                print('len(users_list) != len(items_list)')
            del rating
        dataset.updateMemDict(users_list, items_list)
    return dataset.generate_clean_data()

