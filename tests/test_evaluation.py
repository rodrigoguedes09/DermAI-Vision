import torch
from src.model import create_model
from src.preprocessing import create_data_generators
from src.evaluation.evaluation import evaluate_model

def main():
    # Carregar modelo treinado
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.to(device)
    
    # Criar data loader
    _, val_loader = create_data_generators()
    
    # Avaliar modelo
    print("Iniciando avaliação detalhada do modelo...")
    results = evaluate_model(model, val_loader, device)
    
    print(f"\nAUC-ROC: {results['roc_auc']:.4f}")
    print("\nAvaliação completa! Verifique a pasta 'logs/evaluation_reports' para ver os resultados.")

if __name__ == "__main__":
    main()