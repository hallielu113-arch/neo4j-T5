import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def get_datasets(root="data"):
    label_names=sorted(os.listdir("data/train"))
    for name in ["train","test"]:
        dir_path=root+"/"+name
        with open(f"datasets/{name}.txt", "w", encoding="utf-8") as f:
            for label_name in os.listdir(dir_path):
                label_path=dir_path+"/"+label_name
                for img_name in os.listdir(label_path):
                    img_path=label_path+"/"+img_name
                    label=label_names.index(label_name)
                    f.write(img_path+"\t"+str(label)+"\n")


def get_equalizeHist_image(image):
    return Image.fromarray(cv2.equalizeHist(np.array(image.convert("L")))).convert('RGB')


def get_canny_image(image):
    edges = cv2.Canny(np.array(image), 100, 200)
    return Image.fromarray(edges).convert("RGB")


def get_sobel_image(image):
    image = np.array(image)
    x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    # 对y方向梯度进行sobel边缘提取
    y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    # 对x方向转回uint8
    absX = cv2.convertScaleAbs(x)
    # 对y方向转会uint8
    absY = cv2.convertScaleAbs(y)
    # x，y方向合成边缘检测结果
    dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 与原图像堆叠
    res = dst1 + image

    return Image.fromarray(res).convert("RGB")


class DataGenerator(Dataset):

    def __init__(self,root,process=True):
        super(DataGenerator, self).__init__()
        self.root=root
        if process == "all":
            self.transforms=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Lambda(lambda image:get_canny_image(image)),
                transforms.Lambda(lambda image:Image.fromarray(cv2.equalizeHist(np.array(image.convert("L")))).convert('RGB')),  # 直方图均衡
                transforms.Lambda(lambda image:get_sobel_image(image)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ])
        elif process == "sobel":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda image:get_sobel_image(image)),  # 高斯滤波
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif process == "qualize_hist":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda image: Image.fromarray(cv2.equalizeHist(np.array(image.convert("L")))).convert('RGB')),
                # 直方图均衡
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif process == "canny":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda image:get_canny_image(image)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.img_paths,self.labels=self.get_datasets()

    def __getitem__(self, item):
        img_path=self.img_paths[item]
        img=Image.open(img_path).convert("RGB")

        return torch.FloatTensor(self.transforms(img)),torch.from_numpy(np.array(self.labels[item])).long()

    def __len__(self):
        return len(self.labels)

    def get_datasets(self):
        img_paths,labels=[],[]
        with open(self.root,"r",encoding="utf-8") as f:
            for line in f.readlines():
                line_split=line.strip().split("\t")
                img_paths.append(line_split[0])
                labels.append(int(line_split[-1]))

        return img_paths,labels


if __name__ == '__main__':
    get_datasets()
    # get_process_image(method="canny")
    # get_process_image(method="sobel")
    # get_process_image(method="qualize_hist")














