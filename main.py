# main.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.optim as optim
from src.create_directories import create_directory_structure
from src.data_organization import organize_isic_dataset
from src.model import create_model, get_training_parameters
from src.preprocessing import create_data_generators
from src import config

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

def main():
    # Criar estrutura de diretórios
    print("Criando estrutura de diretórios...")
    create_directory_structure()
    
    # Organizar dataset
    print("Organizando dataset...")
    organize_isic_dataset()
    
    # Criar geradores de dados
    print("Criando geradores de dados...")
    train_loader, val_loader = create_data_generators()
    
    # Criar modelo
    print("Criando modelo...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    model = create_model()
    criterion, optimizer_params = get_training_parameters()
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    
    # Treinar modelo
    print("\nIniciando treinamento...")
    best_val_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f'\nÉpoca {epoch+1}/{config.EPOCHS}')
        print('-' * 10)
        
        # Treinar uma época
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Treino - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # Validar
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Validação - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(config.MODEL_DIR / 'best_model.pth'))
            print(f'Melhor modelo salvo com accuracy: {val_acc:.2f}%')
    
    print("\nTreinamento concluído!")
    print(f"Melhor accuracy de validação: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()