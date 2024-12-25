import torch
from pathlib import Path
from src.model import create_model
from src.explainability import visualize_gradcam, get_target_layer

def test_gradcam():
    # Carregar modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.to(device)
    model.eval()
    
    # Obter camada alvo
    target_layer = get_target_layer(model)
    
    # Diretório para salvar resultados
    output_dir = Path('logs/gradcam_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Testar com algumas imagens
    test_dirs = [
        Path('data/processed/validation/benign'),
        Path('data/processed/validation/malignant')
    ]
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            print(f"Diretório não encontrado: {test_dir}")
            continue
        
        print(f"\nProcessando imagens de: {test_dir}")
        
        # Pegar algumas imagens para teste
        image_paths = list(test_dir.glob('*.jpg'))[:3]  # Testar com 3 imagens de cada classe
        
        for image_path in image_paths:
            print(f"Processando: {image_path.name}")
            try:
                visualize_gradcam(
                    image_path,
                    model,
                    target_layer,
                    output_dir=output_dir
                )
                print("✓ Sucesso!")
            except Exception as e:
                print(f"✗ Erro: {str(e)}")

if __name__ == "__main__":
    test_gradcam()