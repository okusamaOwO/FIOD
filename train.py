from dataset.PairedClearSyntheticDataset import PairedClearSyntheticDataset
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
from dataset.RealFogDataset import RealFogDataset

def show_img(img):
    img = img.clone().detach().cpu()
    img = img.permute(1, 2, 0) 
    plt.imshow(img)
    plt.axis('off') 
    plt.show()

def show_img_with_boxes(img, boxes):
    img = img.clone().detach().cpu()
    img = img.permute(1, 2, 0)  # Convert CHW to HWC
    
    h, w, _ = img.shape  # Get image dimensions
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    
    # Ensure boxes is a tensor
    boxes = torch.tensor(boxes).cpu()

    # Convert YOLO format to (x_min, y_min, x_max, y_max)
    for box in boxes:
        center_x, center_y, width, height = box.tolist()
        
        # Convert from normalized to pixel coordinates
        x_min = (center_x - width / 2) * w
        y_min = (center_y - height / 2) * h
        box_w = width * w
        box_h = height * h
        
        # Draw the bounding box
        rect = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

dataset = RealFogDataset(r"D:\Downloads\lab\dataset\data\dataset02\Foggy_Driving\Processed_foggy_driving")
img = dataset[0][0]
show_img(img)