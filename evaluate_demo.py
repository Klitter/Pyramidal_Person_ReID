# -*- coding: utf-8 -*-
from scipy.spatial.distance import cdist
from torchvision import transforms
from market1501 import Market1501
from __init__ import DEVICE, cmc, mean_ap, creat_test_data_set_loader
from pyrmaid import Pyramid, load_ckpt
import os
import torch
from torch import nn, optim
import numpy as np
import multiprocessing
num_workers = multiprocessing.cpu_count() / 2

root = "/raid/602/llx/market1501/"
model_path = './market/ckpt_ep112_re02_bs64_dropout02_GPU0_mAP0.882439013042_market.pth'
print("root = {}".format(root))
GPUID = "4, 8, 10, 14"
print("GPUID = {}".format(GPUID))
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID


if __name__ == '__main__':
    market_classes = 751
    duke_classes = 702
    cuhk_classes = 767

    batch_test = 32

    test_transform = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    query_dataset, query_loader = creat_test_data_set_loader(root + '/query',
                                                             Market1501, test_transform, batch_test)
    test_dataset, test_loader = creat_test_data_set_loader(
        root + '/bounding_box_test', Market1501, test_transform, batch_test)
    #
    model = Pyramid(num_classes=market_classes)

    model_w = nn.DataParallel(model).to(DEVICE)  # model.to(DEVICE)

    finetuned_params = list(model.base.parameters())
    # To train from scratch
    new_params = [p for n, p in model.named_parameters()
                  if not n.startswith('base.')]
    param_groups = [{'params': finetuned_params, 'lr': 0.01},
                    {'params': new_params, 'lr': 0.1}]
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)

    modules_optims = [model, optimizer]

    resume_ep, scores = load_ckpt(modules_optims,
                                  model_path)

    print(optimizer)
    print('Resume from EP: {}'.format(resume_ep))

    model_w.eval()

    query = np.concatenate([torch.cat(model_w(inputs.to(DEVICE))[0], dim=1).detach().cpu().numpy()
                            for inputs, _ in query_loader])

    test = np.concatenate([torch.cat(model_w(inputs.to(DEVICE))[0], dim=1).detach().cpu().numpy()
                           for inputs, _ in test_loader])

    dist = cdist(query, test)

    r = cmc(dist, query_dataset.ids, test_dataset.ids,
            query_dataset.cameras, test_dataset.cameras,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)

    m_ap = mean_ap(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras)

    print('evaluate_model: mAP=%f, r@1=%f, r@5=%f, r@10=%f' % (m_ap, r[0], r[4], r[9]))