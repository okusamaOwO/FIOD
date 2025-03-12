import cv2
import matplotlib.pyplot as plt

img_path = r"D:\Downloads\lab\FIOD_\dataset01\clear\train\images\hanover_000000_005732_leftImg8bit.png"
img = cv2.imread(img_path)
label_path = img_path.replace('.png', '.txt').replace('images', 'labels')
with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        _, x, y, w, h = line.strip().split()
        x, y, w, h = map(float, [x, y, w, h])
        weight, height = img.shape[1], img.shape[0]
        x, y, w, h = (x - w/2) * weight , (y - h/2) * height, w*weight, h*height
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

plt.imshow(img)
plt.show()
