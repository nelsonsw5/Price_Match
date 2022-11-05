import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
import cv2
import numpy as np
import os
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pymongo
import csv
from tqdm import tqdm
import json
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--image', type=str, default=None, help='specify the config for training')
    parser.add_argument('--model', type=str, default=None, help='specify the config for training')
    parser.add_argument('--Database', type=str, default=None, help='specify the config for training')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()

    return


if __name__ == '__main__':
    main()