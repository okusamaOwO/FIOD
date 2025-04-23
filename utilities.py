import datetime
import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def overlapScore(rects1, rects2):
    avgScore = 0
    scores = []

    for i, _ in enumerate(rects1):
        rect1 = rects1[i]
        rect2 = rects2[i]

        left = np.max((rect1[0], rect2[0]))
        right = np.min((rect1[0] + rect1[2], rect2[0] + rect2[2]))

        top = np.max((rect1[1], rect2[1]))
        bottom = np.min((rect1[1] + rect1[3], rect2[1] + rect2[3]))

        # area of intersection
        i = np.max((0, right - left)) * np.max((0, bottom - top))

        # combined area of two rectangles
        u = rect1[2] * rect1[3] + rect2[2] * rect2[3] - i

        # return the overlap ratio
        # value is always between 0 and 1
        score = np.clip(i / u, 0, 1)
        avgScore += score
        scores.append(score)

    return avgScore, scores


def draw_box(ax, box: np.ndarray, color: str = 'r', label: str = None):
    rect = patches.Rectangle(
        (box[0], box[1]), box[2], box[3],
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    if label:
        ax.text(box[0], box[1] - 2, label, color=color)


def visualize_batch(images: np.ndarray,
                    pred_boxes: np.ndarray,
                    gt_boxes: np.ndarray,
                    img_size: Tuple[int, int] = (100, 100),
                    num_samples: int = 4,
                    save_dir: str = None):
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(num_samples):
        # Get image and convert to proper format
        img = images[i].squeeze()
        if len(img.shape) == 2:
            img = img.reshape(img_size)
        else:
            img = img.transpose(1, 2, 0)

        # Show image
        axes[i].imshow(img, cmap='gray')

        # Draw boxes
        draw_box(axes[i], pred_boxes[i], 'r', 'Pred')
        draw_box(axes[i], gt_boxes[i], 'g', 'GT')

        axes[i].set_title(f'Sample {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/batch_visualization.png')
        plt.close()
    else:
        plt.show()
# Convert tensors to numpy arrays (with GPU handling)
def tensor_to_numpy(tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        # Move tensor to CPU if it's on GPU
        if tensor_or_list.is_cuda:
            return tensor_or_list.detach().cpu().numpy()
        else:
            return tensor_or_list.detach().numpy()
    elif isinstance(tensor_or_list, list):
        if all(isinstance(item, torch.Tensor) for item in tensor_or_list):
            # Convert list of tensors to numpy
            return np.array([tensor.item() if tensor.numel() == 1
                            else tensor.detach().cpu().numpy() if tensor.is_cuda
                            else tensor.detach().numpy()
                            for tensor in tensor_or_list])
        else:
            # It's a list of numbers
            return np.array(tensor_or_list)
    else:
        # Already a numpy array or other numeric type
        return np.array(tensor_or_list)

def plot_losses(box_losses, cls_losses, dfl_losses, fsm_losses, con_losses, total_losses, save_path='./training_losses.png'):
    """
    Plot various training losses over epochs and save the figure.
    Handles PyTorch tensors on both CPU and CUDA devices.

    Args:
        box_losses (list or tensor): Box regression losses
        cls_losses (list or tensor): Classification losses
        dfl_losses (list or tensor): Distribution focal losses
        fsm_losses (list or tensor): Feature selection module losses
        con_losses (list or tensor): Contrastive losses
        total_losses (list or tensor): Total combined losses
        save_path (str, optional): Path to save the plot. Defaults to 'training_losses.png'.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert all loss data to numpy arrays
    box_losses_np = tensor_to_numpy(box_losses)
    cls_losses_np = tensor_to_numpy(cls_losses)
    dfl_losses_np = tensor_to_numpy(dfl_losses)
    fsm_losses_np = tensor_to_numpy(fsm_losses)
    con_losses_np = tensor_to_numpy(con_losses)
    total_losses_np = tensor_to_numpy(total_losses)

    # Create epoch numbers
    epochs = np.arange(1, len(total_losses_np) + 1)

    # Create figure and axis
    plt.figure(figsize=(12, 8))

    # Plot each loss
    plt.plot(epochs, box_losses_np, 'o-', label='Box Loss', linewidth=2)
    plt.plot(epochs, cls_losses_np, 's-', label='Classification Loss', linewidth=2)
    plt.plot(epochs, dfl_losses_np, '^-', label='DFL Loss', linewidth=2)
    plt.plot(epochs, fsm_losses_np, 'd-', label='FSM Loss', linewidth=2)
    plt.plot(epochs, con_losses_np, 'x-', label='Contrastive Loss', linewidth=2)
    plt.plot(epochs, total_losses_np, '*-', label='Total Loss', linewidth=3)

    # Add labels and legend
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.title('Training Losses Over Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Ensure x-axis shows integer epoch numbers
    plt.xticks(epochs)

    # Add padding to make the plot more readable
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    print(f"Plot saved to {save_path}")

def visualize_sample(dataset, index=0, target_size=640):
    # Lấy một mẫu từ dataset
    src_image, trg_image, boxes, labels, name = dataset[index]
    print(f"Name: {name}")
    print(f"Boxes: {boxes.shape}")
    # src_image có định dạng tensor (C, H, W) và đã ở định dạng BGR
    # Chuyển đổi từ tensor sang numpy, chuyển từ CHW sang HWC
    img = src_image.numpy().transpose((1, 2, 0))

    # Chuyển từ BGR sang RGB để hiển thị đúng màu
    img = img[:, :, ::-1]

    # Nếu các bounding box được lưu theo định dạng normalized (0-1), chuyển về pixel
    # Giả sử boxes có dạng (num_boxes, 4) với [x_center, y_center, width, height]
    # Bạn có thể kiểm tra giá trị: nếu các giá trị nhỏ hơn 1, coi là normalized.
    if boxes.numel() > 0 and boxes.max() <= 1.0:
        boxes_abs = boxes.clone()
        boxes_abs[:, 0] = boxes[:, 0] * target_size  # x_center
        boxes_abs[:, 1] = boxes[:, 1] * target_size  # y_center
        boxes_abs[:, 2] = boxes[:, 2] * target_size  # width
        boxes_abs[:, 3] = boxes[:, 3] * target_size  # height
    else:
        boxes_abs = boxes  # giả sử đã là pixel

    # Chuyển từ format [x_center, y_center, width, height] sang [x1, y1, x2, y2]
    boxes_xyxy = boxes_abs.clone()
    boxes_xyxy[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2  # y2

    # Hiển thị ảnh
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    # Vẽ bounding boxes
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        # Tạo một rectangle, bạn có thể thêm label vào
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Vẽ text cho class
        ax.text(x1, y1, f'{int(labels[i])}', color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    ax.set_title(f"Image: {name}")
    plt.axis('off')
    os.makedirs('visualize', exist_ok=True)
    save_path = os.path.join('visualize', 'visual_{:04d}.png'.format(index))

    # Lưu hình vào file
    plt.savefig(save_path)
    plt.close()

def create_infinite_iterator(loader):
    """
    Creates an infinite iterator that automatically resets when the dataset is exhausted.

    Args:
        loader: DataLoader instance

    Returns:
        A generator that yields (iteration_index, batch) pairs indefinitely
    """
    iteration = 0
    while True:
        for i, batch in enumerate(loader):
            yield iteration + i, batch
        iteration += len(loader)
