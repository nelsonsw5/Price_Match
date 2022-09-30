# import the necessary packages
import torch
import cv2
import os
import numpy as np
import argparse
from torchvision import models
import pandas as pd
import csv
import pymongo
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--image_path', type=str, default=None, help='Log path [default: None]')
    # parser.add_argument('--batch-size', type=int, default=16, help='Batch Size during training [default: 16]')
    # parser.add_argument('--epochs', default=10, type=int, help='Epoch to run [default: 32]')
    # parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    # parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    # parser.add_argument('--log-dir', type=str, default=None, help='Log path [default: None]')
    # parser.add_argument('--cache', action="store_true", help='Cache pointclouds in memory during training')
    # parser.add_argument('--features', type=str, help="features to use in model {'density', 'hist', 'gradient'}", default=None)
    # parser.add_argument('--n-workers', type=int, help="number of dataloader workers", default=0)
    # parser.add_argument('--max-from-img', type=int, help="number of dataloader workers", default=64)
    # parser.add_argument('--weight-decay', type=float, help="L2 norm penalty", default=0.1)
    # parser.add_argument('--rescale-outputs', action='store_true', help="max rescale")
    parser.add_argument('--model', type=str, default='resnet')
    return parser.parse_args()

def load_encoding_array(db,ring_df):
    counter = 0
    for i in tqdm(ring_df['Image']):
        key = i + '.jpg'
        # print(key)
        x = db.find_one({"_id":key})
        if x != None:
            embedding = x['embedding']
            if counter == 0:
                final_array = np.array(embedding).reshape(1,len(np.array(embedding)))
                counter = counter + 1
            else:
                final_array = np.vstack((final_array,np.array(embedding).reshape(1,len(np.array(embedding)))))
    return final_array


def main(args):
    device = ('cpu')
    if args.model == 'resnet':
        model = models.resnet152(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        print('Resnet model loaded')
    model.to(device)
    # input_image_path = args.image_path
    # initial_image = plt.imread(input_image_path)
    # plt.imshow(initial_image)  
    # plt.show() 
    ring_annos = []
    # open file for reading
    with open('/Users/stephennelson/Documents/Personal/Tanner_Trading/Ring_Annos - Sheet1.csv') as csvDataFile:

        # read file as csv file 
        csvReader = csv.reader(csvDataFile)

        # for every row, print the row
        for row in csvReader:
            ring_annos.append(row)
    ring_df = pd.DataFrame(ring_annos[1::], columns=['Image','Price'])
    for i in range(len(ring_df['Price'])):
        ring_df['Price'][i] = float(ring_df['Price'][i])
    client = pymongo.MongoClient("mongodb+srv://nelsonsw5:joshuahiggins@tannertrading.zh8eh.mongodb.net/TannerTrading?retryWrites=true&w=majority")
    db = client.Jewelry.Rings
    
    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"
    model.eval()

    input_image_path = args.image_path
    f = os.path.join(input_image_path)
    if os.path.isfile(f):
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype("float32") / 255.0
        image -= MEAN
        image /= STD
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        encoding = model(torch.tensor(image).to(device)).cpu()
        encoding = encoding.reshape(1,2048)
        encoding = encoding[0].detach().numpy()
        array = load_encoding_array(db,ring_df)

        dist_list = list(np.linalg.norm(array - encoding, axis=1))



        image_title = ring_df['Image'][dist_list.index(min(dist_list))]
        image_price = ring_df['Price'][dist_list.index(min(dist_list))]
        path = '/Users/stephennelson/Documents/Personal/Tanner_Trading/Rings_Front/'
        full_image_title = image_title + '.jpg'
        path_result = path + full_image_title
        predicted_image = plt.imread(path_result)
        initial_image = plt.imread(input_image_path)
        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        # showing image
        plt.imshow(initial_image)
        plt.axis('off')
        plt.title("Input Image")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing image
        plt.imshow(predicted_image)
        plt.axis('off')
        predction_title = "Match - $" + str(image_price)
        plt.title(predction_title)
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)