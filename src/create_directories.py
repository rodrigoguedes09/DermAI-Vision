# src/create_directories.py
from pathlib import Path
from . import config

def create_directory_structure():
    """
    Cria a estrutura inicial de diretórios
    """
    # Lista de diretórios para criar
    dirs_to_create = [
        config.PROCESSED_DIR / "train" / "benign",
        config.PROCESSED_DIR / "train" / "malignant",
        config.PROCESSED_DIR / "validation" / "benign",
        config.PROCESSED_DIR / "validation" / "malignant",
        config.RAW_DIR,
        config.MODEL_DIR,
        config.LOGS_DIR
    ]

    # Criar cada diretório
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Estrutura de diretórios criada com sucesso!")
    
    # Mostrar estrutura criada
    print("\nEstrutura de diretórios:")
    for dir_path in dirs_to_create:
        print(f"- {dir_path}")