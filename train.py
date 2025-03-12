from dataset.PairedClearSyntheticDataset import PairedClearSyntheticDataset
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
from dataset.RealFogDataset import RealFogDataset
from torch.utils.data import DataLoader
import os
datapath = r"D:\Downloads\lab\FIOD_\dataset01"
clear_root = os.path.join(datapath, 'clear')
foggy_root = os.path.join(datapath, 'foggy')
dataset = PairedClearSyntheticDataset(clear_root, foggy_root, 'train')
dataloader = DataLoader(dataset,
             batch_size=4,
            shuffle=True,
            collate_fn=dataset.collate_fn)
labels = []
for i, batch in enumerate(dataloader):
    # mình expect cái batch[0] shape này là gì ? 
    first_img = batch[0][0]
    img_name = batch[3][0]
    for bounding_box in batch[2]:
        batch_idx, label, x, y, w, h = bounding_box
        if(batch_idx) == 0:
            labels.append([float(x), float(y), float(w), float(h)])
    break
import matplotlib
import matplotlib.pyplot as plt
plt.imshow(first_img.permute(1, 2, 0))
plt.show()
print(img_name)
for label in labels:
    x, y, w, h = label
    x, y, w, h = x*640, y*640, w*640, h*640
    fig, ax = plt.subplots(1)
    ax.imshow(first_img.permute(1, 2, 0))
    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

