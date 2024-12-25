import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        model_output[0, target_class].backward()
        
        # Get weights
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU on CAM
        cam = torch.relu(cam)
        
        # Normalize CAM
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

def apply_gradcam(model, image_tensor, target_layer):
    """
    Aplica Grad-CAM em uma imagem
    """
    # Configurar modelo para avaliação
    model.eval()
    
    # Criar instância do Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Gerar mapa de calor
    cam = grad_cam.generate_cam(image_tensor)
    
    return cam

def visualize_gradcam(image_path, model, target_layer, output_dir=None):
    """
    Visualiza e salva os resultados do Grad-CAM
    """
    from torchvision import transforms
    
    # Carregar e preprocessar imagem
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Gerar Grad-CAM
    cam = apply_gradcam(model, input_tensor, target_layer)
    
    # Redimensionar CAM para tamanho da imagem original
    cam_resized = cv2.resize(cam, (299, 299))
    
    # Converter para heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Converter imagem para array numpy
    image_array = np.array(image.resize((299, 299)))
    
    # Sobrepor heatmap na imagem original
    superimposed = cv2.addWeighted(image_array, 0.7, heatmap, 0.3, 0)
    
    # Criar visualização
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_array)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Mapa de Atenção')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title('Sobreposição')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Salvar resultados se output_dir for especificado
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        plt.savefig(output_dir / f'{base_name}_gradcam.png')
        
        # Salvar componentes individuais
        cv2.imwrite(str(output_dir / f'{base_name}_heatmap.png'), 
                   cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f'{base_name}_superimposed.png'), 
                   cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    
    return superimposed, cam_resized

def get_target_layer(model):
    """
    Retorna a última camada convolucional do modelo
    """
    # Para ResNet
    return model.layer4[-1]