# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from . import config

def create_model():
    """
    Cria um modelo usando ResNet18 em vez de EfficientNet
    """
    # Usar ResNet18 pré-treinado (mais leve e download mais estável)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modificar a última camada para nossa classificação
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)  # 1 saída para classificação binária
    )
    
    # Mover para GPU se disponível
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

def get_training_parameters():
    """
    Retorna os parâmetros de treinamento
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer_params = {
        'lr': config.LEARNING_RATE,
        'weight_decay': 1e-4
    }
    
    return criterion, optimizer_params