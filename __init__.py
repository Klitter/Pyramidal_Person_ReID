from __future__ import print_function
from collections import defaultdict
import os.path as osp
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
import os
from torch import nn
from tqdm import tqdm
from time import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")
print(DEVICE)


def may_make_dir(path):
    if path in [None, '']:
        return
    if not osp.exists(path):
        os.makedirs(path)

def save_ckpt(modules_optims, ep, scores, ckpt_file):
    state_dicts = [m.state_dict() for m in modules_optims]
    ckpt = dict(state_dicts=state_dicts, ep=ep, scores=scores)
    may_make_dir(osp.dirname(osp.abspath(ckpt_file)))
    torch.save(ckpt, ckpt_file)


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True,strict=True):
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        m.load_state_dict(sd,strict=strict)
    if verbose:
        print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
        ckpt_file, ckpt['ep'], ckpt['scores']))
    return ckpt['ep'], ckpt['scores']


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)

    # print('gallery_ids', gallery_ids.shape )
    # print('query_ids[:, np.newaxis]', query_ids[:, np.newaxis].shape )
    # print('indices',indices.shape)

    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)

def creat_test_data_set_loader(test_path,data_set_class,test_transform ,batch_test):

    test_dataset = data_set_class(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False)

    return test_dataset, test_loader

def creat_train_data_set_loader(train_path,data_set_class,RandomIdSampler,train_transform ,batch_train, batch_id,batch_image):
    train_dataset = data_set_class(train_path, transform=train_transform)
    train_loader_tri = DataLoader(train_dataset,
                                       sampler=RandomIdSampler(train_dataset, batch_image=batch_image),
                                       batch_size=batch_id * batch_image)

    train_loader_all = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, drop_last=True)

    return train_dataset, train_loader_tri, train_loader_all


