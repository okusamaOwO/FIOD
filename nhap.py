import cv2
import matplotlib.pyplot as plt
from dataset.RealFogDataset import RealFogDataset
from torch.utils.data import DataLoader
from dataset.oldPairedClear import OldPairedClearSyntheticDataset

dataset = OldPairedClearSyntheticDataset(src_root=r'D:\Downloads\lab\dataset\data\dataset02\foggy', trg_root=r'D:\Downloads\lab\dataset\data\dataset02\clear', set='train', max_iters=None)