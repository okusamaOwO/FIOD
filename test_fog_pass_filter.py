import datetime
import glob
import os
import matplotlib
from matplotlib import patches
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
import sys
import csv
path = "./yolov9_main"
sys.path.insert(0, path)
from tqdm import tqdm
from utils.dataloaders import create_dataloader
from utils.general import (check_amp, colorstr, one_cycle, one_flat_cycle, yaml_load)
from utils.loss_tal import ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import smart_optimizer, ModelEMA, EarlyStopping
import val as validate
from config.config import get_arguments
from dataset.PairedClearSyntheticDataset import PairedClearSyntheticDataset
from dataset.RealFogDataset import RealFogDataset
from model.feature_extractor import FeatureExtractor
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1, FogPassFilterLoss
from models.yolo import Model
from utilities import plot_losses
from train import parse_opt
from dataset.oldPairedClear import OldPairedClearSyntheticDataset

#HELPER FUNCTIONS
def gram_matrix(feature_map):
    '''
    This is used for calculating the grammatrix from feature maps 
    '''
    channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, torch.t(features))
    return gram

def intersect_dicts(da, db, exclude=()):
    '''
    Forgo its purpose but it helps the code run :)) 
    '''
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def get_model(checkpoint_path = "./cloud_dataset/yolov9_e.pt"):
    '''
    Look at function's name :) 
    '''
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Model(checkpoint['model'].yaml)
    csd = checkpoint['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=())
    model.load_state_dict(csd, strict=False)
    return model

def train_fogpass_filter(model, device, extractor, cwsf_pair_loader, rf_loader, cwsf_pair_loader_fogpass, rf_loader_fogpass, args, FogPassFilter1, FogPassFilter1_optimizer, FogPassFilter2, FogPassFilter2_optimizer, fogpassfilter_loss):
    print("bro chạy vào hàm def thật này")
    fl_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        num_batches = len(cwsf_pair_loader)
        num_batches_cw_sf = 0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx in pbar:
            foggy_image, clear_image, box, name = next(iter(cwsf_pair_loader_fogpass))
            rf_img, rf_name = next(iter(rf_loader_fogpass))


            # TRAINING FOGPASSFILTER PHRASE 
            model.eval() 
            for param in model.parameters():
                param.requires_grad = False # freeze the model yolo params 
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True


            # Get corresponding RF batch
            rf_batch_idx = batch_idx % len(rf_loader)
            rf_img, rf_name = next(iter(rf_loader))

            # Move images to GPU
            realfog_images = rf_img.to(device)
            foggy_images = foggy_image.to(device)
            clear_images = clear_image.to(device)


            # Get feature maps for each image type
            realfog_features = extractor.get_feature_maps(realfog_images)
            foggy_features = extractor.get_feature_maps(foggy_images)
            clear_features = extractor.get_feature_maps(clear_images)
            feature_realfog0, feature_realfog1 = realfog_features[0], realfog_features[1]
            feature_foggy0, feature_foggy1 = foggy_features[0], foggy_features[1]
            feature_clear0, feature_clear1 = clear_features[0], clear_features[1]
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_foggy0, 'layer1': feature_foggy1}
            cw_features = {'layer0': feature_clear0, 'layer1': feature_clear1}
            rf_features = {'layer0': feature_realfog0, 'layer1': feature_realfog1}

            total_fpf_loss = 0
            fogpassfilter = None
            fogpassfilter_optimizer = None

            # loss này là của từng layers
            for idx, layer in enumerate(fsm_weights):
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]
                rf_feature = rf_features[layer]
                # fog_pass_filter_loss = 0
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()
                fogpassfilter_optimizer.zero_grad()

                sf_gram = [0] * args.batch_size
                cw_gram = [0] * args.batch_size
                rf_gram = [0] * args.batch_size
                vector_sf_gram = [0] * args.batch_size
                vector_cw_gram = [0] * args.batch_size
                vector_rf_gram = [0] * args.batch_size
                fog_factor_sf = [0] * args.batch_size
                fog_factor_cw = [0] * args.batch_size
                fog_factor_rf = [0] * args.batch_size

                for i in range(args.batch_size):
                    sf_gram[i] = gram_matrix(sf_feature[i])
                    cw_gram[i] = gram_matrix(cw_feature[i])
                    rf_gram[i] = gram_matrix(rf_feature[i])
                    vector_sf_gram[i] = sf_gram[i][
                        torch.triu(torch.ones_like(sf_gram[i])) == 1
                        ].detach().clone().requires_grad_()

                    vector_cw_gram[i] = cw_gram[i][
                        torch.triu(torch.ones_like(cw_gram[i])) == 1
                        ].detach().clone().requires_grad_()

                    vector_rf_gram[i] = rf_gram[i][
                        torch.triu(torch.ones_like(rf_gram[i])) == 1
                        ].detach().clone().requires_grad_()

                    fog_factor_sf[i] = fogpassfilter(vector_sf_gram[i])
                    fog_factor_cw[i] = fogpassfilter(vector_cw_gram[i])
                    fog_factor_rf[i] = fogpassfilter(vector_rf_gram[i])

                embeddings_list = []
                for i in range(args.batch_size):
                    embeddings_list.append(fog_factor_sf[i].unsqueeze(0))
                    embeddings_list.append(fog_factor_cw[i].unsqueeze(0))
                    embeddings_list.append(fog_factor_rf[i].unsqueeze(0))
                fog_factor_embeddings = torch.cat(embeddings_list, dim=0)

                fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                size_fog_factor = fog_factor_embeddings.size()
                fog_factor_embeddings = fog_factor_embeddings.div(
                    fog_factor_embeddings_norm.expand(size_fog_factor[1], args.batch_size * 3).t())
                fog_factor_labels = torch.arange(3, device=device).long().repeat(args.batch_size)
                fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)
                print(fog_pass_filter_loss)
                total_fpf_loss += fog_pass_filter_loss
            with torch.autograd.detect_anomaly():
                total_fpf_loss.backward()
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()

            foggy_image, clear_image, box, name = next(iter(cwsf_pair_loader))
            rf_img, rf_name = next(iter(rf_loader))



def main():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #CREATE DATALOADER
    cwsf_dataset = PairedClearSyntheticDataset(args.sf_root, args.cw_root, set='train')
    rf_dataset = RealFogDataset(args.rf_root)
    cwsf_pair_loader = DataLoader(
        cwsf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=cwsf_dataset.collate_fn
    )
    rf_loader = DataLoader(
        rf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=rf_dataset.collate_fn
    )
    cwsf_pair_loader_fogpass = DataLoader(
        cwsf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=cwsf_dataset.collate_fn
    )

    rf_loader_fogpass = DataLoader(
        rf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=rf_dataset.collate_fn
    )


    # CREATE FOGPASS FILTER MODEL
    lr_fpf1 = 1e-3
    lr_fpf2 = 1e-3

    FogPassFilter1 = FogPassFilter_conv1(528)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf1)
    FogPassFilter1.to(device)
    FogPassFilter2 = FogPassFilter_res1(2080)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf2)
    FogPassFilter2.to(device)
    fogpassfilter_loss = losses.ContrastiveLoss(
      pos_margin=0.1,
      neg_margin=0.1,
      distance=CosineSimilarity(),
      reducer=MeanReducer()
    )

    # CREATE MODEL YOLOV9 PRESUMED FROM PRETRAINED
    model = get_model()
    model.to(device)

    # MIXED PRECISION TRAINING 
    amp = check_amp(model)
    amp_device = "cuda" if amp else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # GETTING HYPERPARAMS
    hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')

    # CONFIG MODEL 
    opt = parse_opt()
    optimizer = smart_optimizer(model, 'Adam', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    lf = one_flat_cycle(1, hyp['lrf'], opt.epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    RANK = int(os.getenv('RANK', -1))
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    compute_loss = ComputeLoss(model)
    best_fitness, start_epoch = 0.0, 0
    scheduler.last_epoch = start_epoch - 1
    extractor = FeatureExtractor(model, 8)
    train_fogpass_filter(model,
                        device,
                        extractor,
                        cwsf_pair_loader,
                        rf_loader,
                        cwsf_pair_loader_fogpass,
                        rf_loader_fogpass,
                        args,
                        FogPassFilter1, 
                        FogPassFilter1_optimizer,
                        FogPassFilter2,
                        FogPassFilter2_optimizer,
                        fogpassfilter_loss)

if __name__ == "__main__":
  main()