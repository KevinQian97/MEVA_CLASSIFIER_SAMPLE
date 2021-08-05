import argparse
import time
import numpy as np
import torch.nn.parallel
import torch.optim
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import json
import os
from multiprocessing import Pool
from tools.gen_label_mevadet import *
import shutil
import glob
from ops.cube import ActivityTypeMEVA,CubeActivities
import pandas as pd
from ops.merger import OverlapCubeMerger
import gc
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

if __name__ == "__main__":
    args = get_parser()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)  
    if args.dataset in ["DET","MASK","MEVA"]:
        f = open("./labels_det.txt","r")
    elif args.dataset in ["VIRAT"]:
        f = open("./labels_virat.txt","r")
    events = f.readlines()
    event_dict = []
    for event in events:
        event_dict.append(event.strip())
    print(event_dict)

    aug_dict = json.load(open("./augments.json","r"))
    print(aug_dict)

# vids = os.listdir(os.path.join(args.prop_path,"proposal"))
# if os.path.exists(os.path.join(args.out_path,"preds")):
#     print("Found some videos already tackled, previous videos:{}".format(len(vids)))
#     red_vids = os.listdir(os.path.join(args.out_path,"preds"))
#     for vid in red_vids:
#         if vid in vids:
#             vids.pop(vids.index(vid))
#     print("After removing solved videos, current videos:{}".format(len(vids)))
# total_num = None

    this_weights = args.weights
    this_test_segments = int(args.test_segments)
    test_file = args.test_list
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    this_arch = this_weights.split('TSA_')[1].split('_')[2]
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                            modality)
    root_path = args.data_path

    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))


    net = TSN(num_class, this_test_segments, modality,
        base_model=this_arch,
        consensus_type=args.crop_fusion_type,
        dropout=args.dropout,
        img_feature_dim=args.img_feature_dim,
        pretrain=args.pretrain,
        is_shift=True, shift_div=8, shift_place=shift_place,
        non_local='_nl' in this_weights,
        is_TSA=False,
        is_sTSA=False,
        is_tTSA = False,
        shift_diff=None,
        shift_groups=None,
        is_ME = False,
        is_3D = args.is_3D,
        cfg_file=args.cfg_file)

    checkpoint = torch.load(this_weights)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
        }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)

    input_mean = net.input_mean 
    input_std = net.input_std

    input_size = net.scale_size if args.full_res else net.input_size
    if args.is_3D:
        cropping = torchvision.transforms.Compose([
        ChannelFlip(),
        Resize(net.crop_size),
    # CenterCrop(net.crop_size),
        ])
        transform = torchvision.transforms.Compose([
            cropping,
            Normalize(input_mean, input_std, div=True),
            ])
    else:
        transform = torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(input_mean,input_std),
                       ])
    net = torch.nn.DataParallel(net).cuda()
    net.eval()


    prepare_testlist_det(args,args.video)

    if args.topk > args.class_num:
        print(args.topk)
        print(args.class_num)
        raise RuntimeError("class number missmatch")

    data_loader = DataLoaderX(
        TSNDataSet(root_path, test_file if test_file is not None else val_list,             
        num_segments=this_test_segments,
        new_length=1 if modality == "RGB" else 5,
        modality=modality,
        image_tmpl=prefix,
        test_mode=True,
        transform=transform, 
        dense_sample=args.dense_sample, 
        twice_sample=args.twice_sample, 
        all_sample=args.all_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    total_num = len(data_loader.dataset)
        
    output = []
    proc_start_time = time.time()
    torch.no_grad()
    for i, (data, label) in enumerate(data_loader):
        this_rst_list = []
        rst = eval_video((i, data, label), net, args.test_segments, modality)
        this_rst_list.append(rst[1])
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)
        for p, g in zip(ensembled_predict, label.cpu().numpy()):
            output.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        if i % 20 == 0:
            print('video {} done, total {}/{}, average {:.3f} sec/video, '.format(i * args.batch_size, i * args.batch_size, total_num,float(cnt_time) / (i+1) / args.batch_size))

# video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

#update to contain probability
    video_pred_topall = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in output]
    video_prob_topall = [np.sort(np.mean(x[0], axis=0).reshape(-1))[::-1][:args.topk] for x in output]