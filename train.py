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
import sys

from utilities import plot_losses

# path = r"D:\UNI\LAB\FIOD_\yolov9_main"
path = "./yolov9_main"
# path = r"E:\lab\FIOD_\yolov9_main"

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
from train import parse_opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def gram_matrix(feature_map):
    channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, torch.t(features))
    return gram

def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

# def get_model(checkpoint_path = r"D:\UNI\LAB\FIOD_\yolov9-s.pt"):
# def get_model(checkpoint_path = r"/content/drive/MyDrive/FIOD_/yolov9-s.pt"):
def get_model(checkpoint_path = "./yolov9-s.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Model(checkpoint['model'].yaml).to(device)

    csd = checkpoint['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=())
    model.load_state_dict(csd, strict=False)
    return model

def main():
    args = get_arguments()
    lr_fpf1 = 1e-3  # lr của fogpass filter
    lr_fpf2 = 1e-3

    FogPassFilter1 = FogPassFilter_conv1(528)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf1)
    FogPassFilter1.to(device)
    FogPassFilter2 = FogPassFilter_res1(2080)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad == True],
                                                  lr=lr_fpf2)
    FogPassFilter2.to(device)
    # loss của fpf
    fogpassfilter_loss = FogPassFilterLoss(margin=0.1)
    
    from dataset.oldPairedClear import OldPairedClearSyntheticDataset
    cwsf_dataset = OldPairedClearSyntheticDataset(args.sf_root, args.cw_root, set='train')
    cwsf_pair_loader = DataLoader(
        cwsf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=cwsf_dataset.collate_fn
    )
    rf_dataset = RealFogDataset(args.rf_root)
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

    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(dim=1)
    log_m = nn.LogSoftmax(dim=1)
    mse_loss = nn.MSELoss(reduction='mean')
    # model = CNNModel()

    ################# YOLOv9

    RANK = int(os.getenv('RANK', -1))
    hyp = yaml_load('yolov9_main/data/hyps/hyp.scratch-high.yaml')
    opt = parse_opt()
    nc = 2
    names = {0: 'person', 1: 'car'}

    # model = Model(cfg='yolov9_main/models/detect/yolov9-s.yaml')
    model = get_model()
    model.to(device)

    # checkpoint = torch.load('yolov9-s.pt', map_location='cpu')
    # model.load_state_dict(checkpoint, strict = False)

    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    amp = check_amp(model)
    amp_device = "cuda" if amp else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    optimizer = smart_optimizer(model, 'Adam', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    stopper, stop = EarlyStopping(patience=opt.patience), False

    lf = one_flat_cycle(1, hyp['lrf'], opt.epochs)  # flat cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model) if RANK in {-1, 0} else None

    compute_loss = ComputeLoss(model)

    best_fitness, start_epoch = 0.0, 0
    scheduler.last_epoch = start_epoch - 1

    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    gs = max(int(model.stride.max()), 32)
    val_loader = create_dataloader("./dataset01/clear/val",
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

    extractor = FeatureExtractor(model, 8)

    box_losses = []
    cls_losses = []
    dfl_losses = []
    fsm_losses = []
    con_losses = []
    total_losses = []

    for epoch in range(args.num_epochs):
        model.train()

        loss_box_value = 0
        loss_cls_value = 0
        loss_dfl_value = 0
        loss_fsm_value = 0
        loss_con_value = 0
        num_batches = len(cwsf_pair_loader)
        num_batches_cw_sf = 0 # number of batches of SF-CW pair (batch_idx % 3 == 0)

        print("Number of batches: ", num_batches)
        print("Loader fogpass: ", len(cwsf_pair_loader_fogpass))

        # Progress bar for the current epoch
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx in pbar:
            print('1')
            ##############################
            # Fog-pass filtering training using fogpass loader
            foggy_image, clear_image, box, name = next(iter(cwsf_pair_loader_fogpass))
            rf_img, rf_name = next(iter(rf_loader_fogpass))
            print('2')
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True

            # Get corresponding RF batch
            rf_batch_idx = batch_idx % len(rf_loader)
            rf_img, rf_name = next(iter(rf_loader))
            print('3')
            # Move images to GPU
            realfog_images = rf_img.to(device)
            foggy_images = foggy_image.to(device)
            clear_images = clear_image.to(device)
            print('A')
            # Get feature maps for each image type
            realfog_features = extractor.get_feature_maps(realfog_images)
            foggy_features = extractor.get_feature_maps(foggy_images)
            clear_features = extractor.get_feature_maps(clear_images)
            print('B')
            feature_realfog0, feature_realfog1 = realfog_features[0], realfog_features[1]
            feature_foggy0, feature_foggy1 = foggy_features[0], foggy_features[1]
            feature_clear0, feature_clear1 = clear_features[0], clear_features[1]
            print('C')
            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_foggy0, 'layer1': feature_foggy1}
            cw_features = {'layer0': feature_clear0, 'layer1': feature_clear1}
            rf_features = {'layer0': feature_realfog0, 'layer1': feature_realfog1}

            total_fpf_loss = 0
            fogpassfilter = None
            fogpassfilter_optimizer = None
            print('4')
            for idx, layer in enumerate(fsm_weights):
                print('5')
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]
                rf_feature = rf_features[layer]

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
                # fogpassfilter_optimizer.step()
                total_fpf_loss += fog_pass_filter_loss

            print('6')
            # print(f'total_fpf_loss: {total_fpf_loss}')
            with torch.autograd.detect_anomaly():
                total_fpf_loss.backward()
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()
            ##############################

            ##############################
            # Detection training using NORMAL loader
            foggy_image, clear_image, box, name = next(iter(cwsf_pair_loader))
            rf_img, rf_name = next(iter(rf_loader))
            print('7')
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False
            print('8')
            optimizer.zero_grad()

            sf_loss = 0
            cw_loss = 0
            con_loss = 0
            if batch_idx % 3 == 0:
                num_batches_cw_sf += 1
                # SF-CW training
                sf_images = foggy_image.to(device, non_blocking=True).float()
                cw_images = clear_image.to(device, non_blocking=True).float()
                boxes = box.to(device)

                # Get predictions and features
                # with torch.amp.autocast(amp_device):

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
                        con_loss += mse_loss(sf_predictions[1][i][j], cw_predictions[1][i][j])
                        # con_loss = kl_loss(log_m(sf_predictions[1][i][j]), m(cw_predictions[1][i][j]))
                    con_loss /= args.batch_size
                con_loss /= pl

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
                
            loss_fsm = 0
            fog_pass_filter_loss = 0
            print('9')

            print(colorstr(f"batch index {batch_idx + 1}: ") +
                f"{colorstr('yellow', 'box_loss')}: {sf_box_loss:.4f}, "
                f"{colorstr('yellow', 'dfl_loss')}: {cw_dfl_loss:.4f}, "
                f"{colorstr('yellow', 'cls_loss')}: {cw_class_loss:.4f}, "
                f"{colorstr('yellow', 'con_loss')}: {con_loss:.4f}"
            )

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
                        (fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / b_feature.size(0)

                loss_fsm += layer_fsm_loss / 4.

            total_loss = (
                sf_loss +
                cw_loss +
                args.weight_fsm * loss_fsm +  # FSM Loss
                args.weight_con * con_loss  # Consistency Loss
            )
            total_loss = total_loss / num_batches
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
                loss_box_value += box_loss.data.cpu().numpy() / num_batches
            if (sf_class_loss + cw_class_loss) != 0:
                class_loss = sf_class_loss + cw_class_loss
                loss_cls_value += class_loss.data.cpu().numpy() / num_batches
            if (sf_dfl_loss + cw_dfl_loss) != 0:
                loss_dfl = sf_dfl_loss + cw_dfl_loss
                loss_dfl_value += loss_dfl.data.cpu().numpy() / num_batches
            if loss_fsm != 0:
                # loss_fsm = loss_fsm * args.weight_fsm
                loss_fsm_value += loss_fsm.data.cpu().numpy() / num_batches
            if con_loss != 0:
                # con_loss = con_loss * args.weight_con
                loss_con_value += con_loss.data.cpu().numpy()

            scheduler.step()

            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

        print("Number of batches of pair CW-SF: ", num_batches_cw_sf)
        loss_con_value /= num_batches_cw_sf

        box_losses.append(loss_box_value)
        dfl_losses.append(loss_dfl_value)
        cls_losses.append(loss_cls_value)
        fsm_losses.append(loss_fsm_value)
        con_losses.append(loss_con_value)
        total_losses.append(total_loss)

        # Print losses after each epoch
        print(colorstr(f"Epoch {epoch + 1}: ") +
            f"{colorstr('bright_magenta', 'total_loss')}: {total_loss:.4f}, "
            f"{colorstr('bright_magenta', 'box_loss')}: {loss_box_value:.4f}, "
            f"{colorstr('bright_magenta', 'dfl_loss')}: {loss_dfl_value:.4f}, "
            f"{colorstr('bright_magenta', 'cls_loss')}: {loss_cls_value:.4f}, "
            f"{colorstr('bright_magenta', 'fsm_loss')}: {loss_fsm_value:.4f}, "
            f"{colorstr('bright_magenta', 'con_loss')}: {loss_con_value:.4f}"
        )

        # # End of epoch validation
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
        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))
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

        # # Early stopping check
        # if stopper(epoch=epoch, fitness=fi):
        #     break

    plot_losses(box_losses, cls_losses, dfl_losses, fsm_losses, con_losses, total_losses)
    print("end")

    return


if __name__ == '__main__':
    main()
