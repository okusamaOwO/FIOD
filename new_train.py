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
from torch.utils.data import DataLoader, RandomSampler
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

def train(model,
          device,
          extractor,
          cwsf_pair_loader,
          rf_loader,
          val_loader,
          args,
          FogPassFilter1,
          FogPassFilter1_optimizer,
          FogPassFilter2,
          FogPassFilter2_optimizer,
          optimizer,
          fogpassfilter_loss,
          compute_loss,
          amp_device,
          amp_dtype,
          scaler,
          ema,
          scheduler):
    # different kind of losses
    header = ["total_loss", "box_loss", "dfl_loss", "cls_loss", "fsm_loss", "con_loss"]
    with open("losses.csv", mode="w", newline="") as file:
      writer = csv.writer(file)
      writer.writerow(header)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(dim=1)
    log_m = nn.LogSoftmax(dim=1)
    box_losses = []
    cls_losses = []
    dfl_losses = []
    fsm_losses = []
    con_losses = []
    total_losses = []


    for epoch in range(args.num_epochs):
        total_batches = len(cwsf_pair_loader)
        batch_bar = tqdm(zip(cwsf_pair_loader, rf_loader), total=total_batches, desc=f"Epoch {epoch+1}", leave=False)
        loss_box_value = 0
        loss_cls_value = 0
        loss_dfl_value = 0
        loss_fsm_value = 0
        loss_con_value = 0
        epoch_fpf_loss = 0
        import math
        best_fpf_loss = 1e10
        for batch_idx, (paired_imgs_batch,realfog_imgs_batch) in enumerate(batch_bar):
            foggy_image, clear_image, box, name = paired_imgs_batch # that ra la phai co e, es moi dung nma ke di luoi bo me
            rf_img, rf_name = realfog_imgs_batch
            # Move images to GPU
            realfog_images = rf_img.to(device)
            foggy_images = foggy_image.to(device)
            clear_images = clear_image.to(device)

            # TRAINING FOGPASSFILTER PHRASE
            # TRAINING FOGPASSFILTER PHRASE
            # TRAINING FOGPASSFILTER PHRASE


            model.eval()
            for param in model.parameters():
                param.requires_grad = False # freeze the model yolo params
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True

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
                total_fpf_loss += fog_pass_filter_loss
            with torch.autograd.detect_anomaly():
                total_fpf_loss.backward()

            epoch_fpf_loss += total_fpf_loss
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()



            # TRAINING YOLOV9 MODEL
            # TRAINING YOLOV9 MODEL
            # TRAINING YOLOV9 MODEL

            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False
            optimizer.zero_grad()
            sf_loss = 0
            cw_loss = 0
            con_loss = 0

            if batch_idx % 3 == 0:
                sf_images = foggy_image.to(device, non_blocking=True).float()
                cw_images = clear_image.to(device, non_blocking=True).float()
                boxes = box.to(device)
                with torch.autocast(device_type=amp_device, dtype= amp_dtype):
                    sf_predictions = model(sf_images)  # forward
                    sf_loss, sf_loss_items = compute_loss(sf_predictions[1], boxes)
                    sf_box_loss, sf_class_loss, sf_dfl_loss = sf_loss_items
                    cw_predictions = model(cw_images)  # forward
                    cw_loss, cw_loss_items = compute_loss(cw_predictions[1], boxes)
                    cw_box_loss, cw_class_loss, cw_dfl_loss = cw_loss_items

                sf_features_list = extractor.get_feature_maps(sf_images)
                feature_sf0, feature_sf1 = sf_features_list[0], sf_features_list[1]
                cw_features_list = extractor.get_feature_maps(cw_images)
                feature_cw0, feature_cw1 = cw_features_list[0], cw_features_list[1]

                # CONSISTENCY LOSS
                pl = len(sf_predictions[1]) # prediction layers
                for i in range(len(sf_predictions[1])):
                    for j in range(args.batch_size):
                        sf_prediction_logsoftmax = log_m(torch.sigmoid(sf_predictions[1][i][j]))
                        cw_prediction_softmax = m(torch.sigmoid(cw_predictions[1][i][j]))
                        con_loss += 10 * kl_loss(sf_prediction_logsoftmax, cw_prediction_softmax)
                con_loss /= (pl * args.batch_size)

                if torch.isnan(sf_predictions[1][i][j]).any() or torch.isnan(cw_predictions[1][i][j]).any():
                    print("NaN detected in predictions at layer", i, "batch index", j)

                fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
                sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
                cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
            elif batch_idx % 3 == 1:
                # SF-RF training
                sf_images = foggy_image.to(device, non_blocking=True).float()
                rf_images = rf_img.to(device, non_blocking=True).float()
                boxes = box.to(device)

                # with torch.amp.autocast(amp_device):
                with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                    sf_predictions = model(sf_images)  # forward
                    sf_loss, sf_loss_items = compute_loss(sf_predictions[1], boxes)
                    sf_box_loss, sf_class_loss, sf_dfl_loss = sf_loss_items
                sf_features_list = extractor.get_feature_maps(sf_images)
                feature_sf0, feature_sf1 = sf_features_list[0], sf_features_list[1]

                rf_predictions = model(rf_images)
                rf_features_list = extractor.get_feature_maps(rf_images)
                feature_rf0, feature_rf1 = rf_features_list[0], rf_features_list[1]

                rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}
                sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
                fsm_weights = {'layer0': 0.5, 'layer1': 0.5}

            else:  # batch_idx % 3 == 2
                # CW-RF training
                cw_images = clear_image.to(device, non_blocking=True).float()
                rf_images = rf_img.to(device, non_blocking=True).float()
                boxes = box.to(device)

                # with torch.amp.autocast(amp_device):
                with torch.autocast(device_type=amp_device, dtype=torch.float16):
                    cw_predictions = model(cw_images)
                    cw_loss, cw_loss_items = compute_loss(cw_predictions[1], boxes)
                    cw_box_loss, cw_class_loss, cw_dfl_loss = cw_loss_items
                cw_features_list = extractor.get_feature_maps(cw_images)
                feature_cw0, feature_cw1 = cw_features_list[0], cw_features_list[1]

                rf_predictions = model(rf_images)
                rf_features_list = extractor.get_feature_maps(rf_images)
                feature_rf0, feature_rf1 = rf_features_list[0], rf_features_list[1]

                rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}
                cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
                fsm_weights = {'layer0': 0.5, 'layer1': 0.5}

            ### đây là cái bên trong if này
            loss_fsm = 0
            for idx, layer in enumerate(fsm_weights):
                a_feature = None
                b_feature = None
                # fog pass filter loss between different fog conditions a and b
                if batch_idx % 3 == 0:
                    a_feature = cw_features[layer]
                    b_feature = sf_features[layer]
                if batch_idx % 3 == 1:
                    a_feature = rf_features[layer]
                    b_feature = sf_features[layer]
                if batch_idx % 3 == 2:
                    a_feature = rf_features[layer]
                    b_feature = cw_features[layer]
                layer_fsm_loss = 0
                na, da, ha, wa = a_feature.size()
                nb, db, hb, wb = b_feature.size()

                fogpassfilter = None
                fogpassfilter_optimizer = None
                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.eval()
                for i in range(args.batch_size):
                    b_gram = gram_matrix(b_feature[i])
                    a_gram = gram_matrix(a_feature[i])
                    if batch_idx % 3 == 1 or batch_idx % 3 == 2:
                        a_gram = a_gram * (hb * wb) / (ha * wa)
                    vector_b_gram = b_gram[torch.triu(
                        torch.ones(b_gram.size()[0], b_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                    vector_a_gram = a_gram[torch.triu(
                        torch.ones(a_gram.size()[0], a_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                    fog_factor_b = fogpassfilter(vector_b_gram)
                    fog_factor_a = fogpassfilter(vector_a_gram)
                    half = int(fog_factor_b.shape[0] / 2)
                    layer_fsm_loss += fsm_weights[layer] * torch.mean(
                        (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2)
                loss_fsm += -torch.log10(layer_fsm_loss)/4
            total_loss = (
                sf_loss +
                cw_loss +
                args.weight_fsm * loss_fsm +  # FSM Loss
                args.weight_con * con_loss  # Consistency Loss
            )
            total_loss = total_loss / total_batches
            with torch.autograd.detect_anomaly():
                scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            if ema:
                ema.update(model)

            if (sf_box_loss + cw_box_loss) != 0:
                box_loss = sf_box_loss + cw_box_loss
                loss_box_value += box_loss.data.cpu().numpy()
            if (sf_class_loss + cw_class_loss) != 0:
                class_loss = sf_class_loss + cw_class_loss
                loss_cls_value += class_loss.data.cpu().numpy()
            if (sf_dfl_loss + cw_dfl_loss) != 0:
                loss_dfl = sf_dfl_loss + cw_dfl_loss
                loss_dfl_value += loss_dfl.data.cpu().numpy()
            if loss_fsm != 0:
                # loss_fsm = loss_fsm * args.weight_fsm
                loss_fsm_value += loss_fsm.data.cpu().numpy()
            if con_loss != 0:
                # con_loss = con_loss * args.weight_con
                loss_con_value += con_loss.data.cpu().numpy()
            scheduler.step()
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

        box_losses.append(loss_box_value)
        dfl_losses.append(loss_dfl_value)
        cls_losses.append(loss_cls_value)
        fsm_losses.append(loss_fsm_value)
        con_losses.append(loss_con_value)
        total_losses.append(total_loss)
        epoch_loss = [total_loss, loss_box_value, loss_dfl_value, loss_cls_value, loss_fsm_value, loss_con_value]
        with open("losses.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(epoch_loss)

        print(colorstr(f"Epoch {epoch + 1}: ") +
            f"{colorstr('bright_magenta', 'total_loss')}: {total_loss:.4f}, "
            f"{colorstr('bright_magenta', 'box_loss')}: {loss_box_value:.4f}, "
            f"{colorstr('bright_magenta', 'dfl_loss')}: {loss_dfl_value:.4f}, "
            f"{colorstr('bright_magenta', 'cls_loss')}: {loss_cls_value:.4f}, "
            f"{colorstr('bright_magenta', 'fsm_loss')}: {loss_fsm_value:.4f}, "
            f"{colorstr('bright_magenta', 'con_loss')}: {loss_con_value:.4f}"
        )
        # # End of epoch validation
        if epoch % 5 == 0:
            print("\nRunning validation...")
            results, maps, _ = validate.run(
                batch_size=args.batch_size,
                half=amp,
                model=ema.ema,
                single_cls=False,
                dataloader=val_loader,
                save_dir=save_dir,
                plots=False,
                compute_loss=compute_loss
            )
            print(results)
            print(maps)
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))
        save_dir = "./best_yolo_model/"
        if fi > best_fitness:
            best_fitness = fi
            # Save best model
            for old_model in glob.glob(os.path.join(save_dir, "best_*.pt")):
                os.remove(old_model)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fitness': best_fitness,
            }, os.path.join(save_dir, f'best_{timestamp}.pt'))
        fi_value = float(fi) if isinstance(fi, np.ndarray) else fi
        best_fitness_value = float(best_fitness) if isinstance(best_fitness, np.ndarray) else best_fitness
        print(f"Epoch {epoch}: Fitness Score = {fi_value:.4f}, Best Fitness = {best_fitness_value:.4f}")

        print(f"[Epoch {epoch+1}] Epoch Loss: {epoch_fpf_loss:.4f}")
        ## SAVE MODEL
        if (epoch_fpf_loss < best_fpf_loss):
            best_fpf_loss = epoch_fpf_loss
            print("saving best model..")
            torch.save({
                'epoch': epoch,
                'fpf1_state_dict': FogPassFilter1.state_dict(),
                'fpf2_state_dict': FogPassFilter2.state_dict(),
                'optimizer_fpf1_state_dict': FogPassFilter1_optimizer.state_dict(),
                'optimizer_fpf2_state_dict': FogPassFilter2_optimizer.state_dict(),
                'loss': fogpassfilter_loss,
            }, "./best_fpf.pth")

        print("END OF TRAINING FPF MODEL, SAVING LATEST MODEL")
        torch.save({
            'epoch': args.num_epochs_fpf,
            'fpf1_state_dict': FogPassFilter1.state_dict(),
            'fpf2_state_dict': FogPassFilter2.state_dict(),
            'optimizer_fpf1_state_dict': FogPassFilter1_optimizer.state_dict(),
            'optimizer_fpf2_state_dict': FogPassFilter2_optimizer.state_dict(),
            'loss': fogpassfilter_loss,
        }, "latest_fpf.pth")
    plot_losses(box_losses, cls_losses, dfl_losses, fsm_losses, con_losses, total_losses)
    print("end")
    return

def main():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #CREATE DATALOADER
    cwsf_dataset = PairedClearSyntheticDataset(args.cw_root,args.sf_root, set='train')
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
        sampler=RandomSampler(
            rf_dataset,
            replacement=True,  # Enable sampling with replacement
            num_samples=len(cwsf_pair_loader.dataset)
        ),
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
    # LOAD BEST FPF MODEL
    checkpoint = torch.load("./best_fpf.pth", map_location=device)
    FogPassFilter1.load_state_dict(checkpoint['fpf1_state_dict'])
    FogPassFilter2.load_state_dict(checkpoint['fpf2_state_dict'])
    FogPassFilter1_optimizer.load_state_dict(checkpoint['optimizer_fpf1_state_dict'])
    FogPassFilter2_optimizer.load_state_dict(checkpoint['optimizer_fpf2_state_dict'])
    start_epoch = checkpoint['epoch'] # don't need to care about this one :P
    print(f"Restored FPF models from epoch {start_epoch}")


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
    gs = max(int(model.stride.max()), 32)
    val_loader = create_dataloader("./dataset02/clear/val",
                                   640,
                                   args.batch_size,
                                   gs,
                                   single_cls=False,
                                   hyp=hyp,
                                   cache=None,
                                   rect=True,
                                   rank=-1,
                                   workers=args.num_workers,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]
    train(model,
          device,
          extractor,
          cwsf_pair_loader,
          rf_loader,
          val_loader,
          args,
          FogPassFilter1,
          FogPassFilter1_optimizer,
          FogPassFilter2,
          FogPassFilter2_optimizer,
          optimizer,
          fogpassfilter_loss,
          compute_loss,
          amp_device,
          amp_dtype,
          scaler,
          ema,
          scheduler)
    print("end.")

if __name__ == "__main__":
    main()