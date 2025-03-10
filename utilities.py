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

def plot_losses(box_losses, cls_losses, dfl_losses, fsm_losses, con_losses, total_losses):
    box_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in
                     box_losses]
    cls_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in
                     cls_losses]
    dfl_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in
                     dfl_losses]
    fsm_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in
                     fsm_losses]
    con_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in
                        con_losses]
    total_losses_np = [loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else np.array(loss) for loss in
                       total_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(box_losses)), box_losses_np, label='Box Loss', color='r')
    plt.plot(range(len(cls_losses)), cls_losses_np, label='Class Loss', color='g')
    plt.plot(range(len(dfl_losses)), dfl_losses_np, label='DFL Loss', color='b')
    plt.plot(range(len(fsm_losses)), fsm_losses_np, label='FSM Loss', color='y')
    plt.plot(range(len(con_losses)), con_losses_np, label='Con Loss', color='m')
    plt.plot(range(len(total_losses)), total_losses_np, label='Total Loss', color='k')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join('results', f'loss_plot_{timestamp}.png')

    # Lưu hình vào file
    plt.savefig(save_path)
    plt.close()

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