from torch.utils.data import Dataset
import numpy as np
import torch
import os
from utils import loadTxt, loadDict, saveDict
from PIL import Image
from data_augment import gauss_noise
from RandAugment import RandAugment_space_work_q

def default_loader(path):
    return Image.open(path)


class ImageDataSet(Dataset):
    # validation
    # labeled input
    def __init__(self, txt_path, img_dir, label_dir, loader=default_loader, transform=False, sort=False, seed=42):
        fileNames = [name for name in loadTxt(str(txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.loader = loader
        self.seed = seed
        self.transform = transform

    def preprocess(self, img, size, label=False):
        img = img.resize(size, Image.BILINEAR)
        img = np.array(img)

        if len(img.shape) == 2:
            # (512, 512) -> (512, 512, 1)
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            # # (512, 512, n) -> (512, 512) -> (512, 512, 1)
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        if label:
            # transpose to (1, 512, 512)
            img_trans = img.transpose((2, 0, 1))
            return img_trans
        else:
            # normalize
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]

        img = self.loader(str(self.img_dir / fileName))
        label = self.loader(str(self.label_dir / fileName))

        img = self.preprocess(img, (512, 512))
        label = self.preprocess(label, (512, 512), label=True)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet1(Dataset):
    def __init__(self, labeled_txt_path, un_txt_path, img_dir, label_dir, loader=default_loader, transform=False,
                 sort=False):
        fileNames = [name for name in loadTxt(str(labeled_txt_path))] + [name for name in loadTxt(str(un_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, label=False, DA_s=False, DA_w=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            img = self.transform(img)

        img = np.array(img)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)
        #img = gauss_noise(img, (0.0, 0.01))
        if label:            
            return img.transpose((2, 0, 1))
        else:
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512), label=False)
        img2 = self.preprocess(img, (512, 512), label=False, DA_s=True)

        if os.path.exists(str(self.label_dir / fileName)):
            label = self.loader(str(self.label_dir / fileName))
            label = self.preprocess(label, (512, 512), label=True)
        else:
            label = np.zeros((1, 512, 512))

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet2(Dataset):
    # UDA & MPL
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            # RandAugment
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)

    

class ImageDataSet3(Dataset):
    # Gaussian Noise as Noise
    # MeanTeacher
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
           img1, img2 = self.transform(img)

        img1 = np.array(img)
        img2 = np.array(img)
        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=2)
            img2 = np.expand_dims(img2, axis=2)
            
        elif len(img1.shape) == 3:
            img1 = img1[:, :, 0][:, :, np.newaxis]
            img2 = img2[:, :, 0][:, :, np.newaxis]

        assert img1.shape == (512, 512, 1)
        assert img2.shape == (512, 512, 1)

        img1 = gauss_noise(img1, (0.0, 0.01))
        img2 = gauss_noise(img2, (0.0, 0.01))

        img1 = img1 / 255.0
        img2 = img2 / 255.0
        return img1.transpose((2, 0, 1)), img2.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))
        
        img1, img2 = self.preprocess(img, (512, 512))

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet4(Dataset):
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)
        img = gauss_noise(img, (0.0, 0.01))

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)


class ImageDataSet_aug_Gauss(Dataset):
    # MPL : aug*2 + Gauss(only strong, weak not used)
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            # RandAugment
            img = self.transform(img)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)
        img2 = gauss_noise(img2, (0.0, 0.01))

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_Gauss_on_un_s(Dataset):
    # Gaussian Noise as Noise
    # MeanTeacher
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            # RandAugment
            img = self.transform(img)
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512))
        img2 = gauss_noise(img2, (0.0, 0.01))

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_only_unlabeled(Dataset):
    # UDA & MPL
    def __init__(self, un_txtPath, img_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        if DA_s:
            # RandAugment
            img = self.transform(img)

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)
    
class ImageDataSet_filter_n_space_aug(Dataset):
    # UDA & MPL
    def __init__(self, labeled_txt_path, un_txtPath, img_dir, save_dir, loader=default_loader, transform=False, sort=False):
        fileNames = [name for name in loadTxt(str(un_txtPath))] + [name for name in loadTxt(str(labeled_txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.img_dir = img_dir
        self.loader = loader
        self.transform = transform
        self.augment_index = 0
        self.save_dir = save_dir

    def preprocess(self, img, size, DA_s=False):
        img = img.resize(size, Image.BILINEAR)
        ops_work_q = RandAugment_space_work_q()
        ops_work_q.load_work_q_list(loadDict("{}/space_work_q.pickle".format(self.save_dir)))
        
        if DA_s:
            # RandAugment
            img, (op, val) = self.transform(img)
            ops_work_q.append(op, val)
            saveDict("{}/space_work_q.pickle".format(self.save_dir), ops_work_q.get_work_q_list())
            
#             print(self.augment_index, ops_work_q.get_work_q_list())
            self.augment_index += 1

        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        img = img / 255.0
        return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]
        img = self.loader(str(self.img_dir / fileName))

        img1 = self.preprocess(img, (512, 512))
        img2 = self.preprocess(img, (512, 512), DA_s=True)

        return torch.from_numpy(img1).float(), torch.from_numpy(img2).float()

    def __len__(self):
        return len(self.fileNames)

class ImageDataSet_multi(Dataset):
    def __init__(self, txt_path, img_dir, label_dir, loader=default_loader, transform=False, sort=False, seed=42):
        fileNames = [name for name in loadTxt(str(txt_path))]
        if sort:
            fileNames.sort()
        self.fileNames = fileNames
        self.imgDir = img_dir
        self.labelDir = label_dir
        self.loader = loader
        self.seed = seed
        self.transform = transform

    def one_hot(self, label):
        width, height = label.shape[0], label.shape[1]
        background = np.zeros((width, height, 1))
        class1 = np.zeros((width, height, 1))
        class2 = np.zeros((width, height, 1))
        background[label==0] = 1
        class1[label==1] = 1
        class2[label==2] = 1
        return np.concatenate([background, class1, class2], axis=2)

    def preprocess(self, img, size, label=False):
        img = img.resize(size, Image.BILINEAR)
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        elif len(img.shape) == 3:
            img = img[:, :, 0][:, :, np.newaxis]

        assert img.shape == (512, 512, 1)

        if label:
            img = self.one_hot(img)
            img_trans = img.transpose((2, 0, 1))
            return img_trans
        else:
            img = img / 255.0
            return img.transpose((2, 0, 1))

    def __getitem__(self, idx):
        fileName = self.fileNames[idx]

        img = self.loader(str(self.imgDir / fileName))
        label = self.loader(str(self.labelDir / fileName))

        img = self.preprocess(img, (512, 512))
        label = self.preprocess(label, (512, 512), label=True)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.fileNames)
