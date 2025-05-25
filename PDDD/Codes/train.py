import numpy as np
from random import random

from transformers import BertTokenizer
from torchvision import transforms

from utils.config import get_config
# from data_loader import get_loader
from utils.solver import Solver
from torch.utils.data import DataLoader

import torch
import os
import sys

sys.path.append(os.getcwd())
from utils.custom_dataset import CustomDataset

from modelpy.visual_model.ResNet_18_34 import ResNet18,ResNet34
from modelpy.visual_model.ResNet_50_101_152 import ResNet50,ResNet101,ResNet152
from modelpy.visual_model.DenseNet_169_264 import DenseNet169,DenseNet264
from modelpy.visual_model.AlexNet import AlexNet
from modelpy.visual_model.ConvNeXt import convnext_base
from modelpy.visual_model.DarkNet_19_53 import DarkNet19,DarkNet53
from modelpy.visual_model.EfficientNet import EfficientNet
from modelpy.visual_model.EfficientNetV2 import EfficientNetV2
from modelpy.visual_model.GoogleNet import GoogLeNet
from modelpy.visual_model.MobileNetV2 import MobileNetV2
from modelpy.visual_model.MobileNetV3 import mobilenet_v3_small
from modelpy.visual_model.RegNet import create_regnet
from modelpy.visual_model.VggNet_16 import VGG16
from modelpy.visual_model.VggNet_19 import VGG19
from modelpy.visual_model.ShuffleNet import shufflenet_v2_x0_5
from modelpy.visual_model.ViT_L import VisionTransformer
from modelpy.textual_model.BiLSTM import BiLSTM
from modelpy.textual_model.modeling_albert import AlbertModel
from modelpy.textual_model.configuration_albert import AlbertConfig
from modelpy.textual_model.LSTM import LSTM

# from transformers import BertModel, BertConfig
if __name__ == '__main__':
    # Setting random seed
    random_name = str(random())
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    print(train_config)

    # Creating pytorch dataloaders
    # train_data_loader = get_loader(train_config, shuffle = True)
    # dev_data_loader = get_loader(dev_config, shuffle = False)
    # test_data_loader = get_loader(test_config, shuffle = False)

    rgb_mean = [0.425, 0.463, 0.385]
    rgb_std = [0.215, 0.216, 0.244]
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # train_dataset = CustomDataset('CUB_200_2011/bird_train.txt', 'CUB_200_2011/images', transform_val, tokenizer)
    # dev_dataset = CustomDataset('CUB_200_2011/bird_test.txt', 'CUB_200_2011/images', transform_val, tokenizer)

    # create dataset object from csv + dir_path
    train_dataset = CustomDataset("/media/icnlab/Data/Manh/tinyML/FieldPlant-11/train.csv", "/media/icnlab/Data/Manh/tinyML/FieldPlant-11/cropped", transform_val)
    dev_dataset = CustomDataset("/media/icnlab/Data/Manh/tinyML/FieldPlant-11/test.csv", "/media/icnlab/Data/Manh/tinyML/FieldPlant-11/cropped", transform_val)
    # DataLoader loop through dataset for batch
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

# from config => run proper model
    if train_config.textual_model=="LSTM":
        textual_model = LSTM(class_num=118, vocab_size=500000, embedding_dim=128, hidden_dim=768, num_layers=8,
                         dropout=0.5)
    elif train_config.textual_model=="BERT":
        textual_model = BertModel(BertConfig())
    elif train_config.textual_model=="ALBERT":
        config=AlbertConfig(hidden_size=768)
        textual_model = AlbertModel(config)
    elif train_config.textual_model=="BiLSTM":
        textual_model = BiLSTM(vocab=500000)
        
    if train_config.visual_model=="ResNet18":
        visual_model = ResNet18()
    elif train_config.visual_model=="ResNet34":
        visual_model = ResNet34()
    elif train_config.visual_model=="ResNet50":
        visual_model = ResNet50()
    elif train_config.visual_model=="ResNet101":
        visual_model = ResNet101()
    elif train_config.visual_model=="ResNet152":
        visual_model = ResNet152()        
    elif train_config.visual_model=="DenseNet169":
        visual_model = DenseNet169(3,118)
    elif train_config.visual_model=="DenseNet264":
        visual_model = DenseNet264(3,118)                             
    elif train_config.visual_model=="VGG16":
        visual_model = VGG16()                
    elif train_config.visual_model=="VGG19":
        visual_model = VGG19()    
    elif train_config.visual_model=="ViT_B":
        visual_model = VisionTransformer()
    elif train_config.visual_model=="ViT_L":
        visual_model = VisionTransformer()
    elif train_config.visual_model=="AlexNet":
        visual_model = AlexNet()           
    elif train_config.visual_model=="RegNet":
        visual_model = create_regnet(num_classes=1024)         
    elif train_config.visual_model=="ShuffleNet":
        visual_model = shufflenet_v2_x0_5()         
    elif train_config.visual_model=="GoogleNet":
        visual_model = GoogLeNet()                                             
    elif train_config.visual_model=="ConvNeXt":
        visual_model = convnext_base(118)   
    elif train_config.visual_model=="DarkNet19":
        visual_model = DarkNet19()   
    elif train_config.visual_model=="DarkNet53":
        visual_model = DarkNet53()   
    elif train_config.visual_model=="EfficientNet":
        visual_model = EfficientNet(1.0, 1.0)   
    elif train_config.visual_model=="EfficientNetV2":
        visual_model = EfficientNetV2([[2, 3, 1, 1, 24, 24, 0, 0],
                                   [4, 3, 2, 4, 24, 48, 0, 0],
                                   [4, 3, 2, 4, 48, 64, 0, 0],
                                   [6, 3, 2, 4, 64, 128, 1, 0.25],
                                   [9, 3, 1, 6, 128, 160, 1, 0.25],
                                   [15, 3, 2, 6, 160, 256, 1, 0.25]])   
    elif train_config.visual_model=="MobileNetV2":
        visual_model = MobileNetV2()  
    elif train_config.visual_model=="MobileNetV3":
        visual_model = mobilenet_v3_small(118)  
                
# Solver is a wrapper for model traiing and testing
    from utils.solver import Solver  # Make sure to import the Solver class
    textual_model = None  # Set to None when 'none' is specified
    solver = Solver(visual_model, textual_model, train_config, dev_config, test_config, 
                   train_loader, dev_loader, dev_loader, is_train=True)

    # Build and train the model
    solver.build()
    solver.train()
