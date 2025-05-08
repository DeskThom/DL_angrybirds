from ultralytics import YOLO
import torch


class YOLOModel:
    def __init__(self, model_path='yolo11n.pt', device='cpu'):
        self.device = device
        # Load YOLO model - Pretrained
        self.model = YOLO(model_path).to(self.device)  # Load YOLO model

    def train(self, data_yaml='data.yaml', epochs=50, imgsz=640, batch_size=32):
        """
        Use YOLO's built-in training functionality with settings
        """
        # Calling the model's train function directly
        self.model.train(
            data=data_yaml,  # Path to dataset YAML file
            epochs=epochs,  # Number of epochs
            imgsz=imgsz,  # Image size for training
            batch=batch_size,  # Batch size
            device=self.device  # Device (CPU or CUDA)
        )

    def test(self, data_yaml='data.yaml', imgsz=640, batch_size=32):
        """
        Use YOLO's built-in testing functionality with settings
        """
        # Calling the model's test function directly
        self.model.val(
            data=data_yaml,  # Path to dataset YAML file
            imgsz=imgsz,  # Image size for testing
            batch=batch_size,  # Batch size
            device=self.device,  # Device (CPU or CUDA)
            split='test'  # Split for testing (train, val, test)
        )
        
    def val(self, data_yaml='data.yaml', imgsz=640, batch_size=32):
        """
        Use YOLO's built-in testing functionality with settings
        """
        # Calling the model's test function directly
        self.model.val(
            data=data_yaml,  # Path to dataset YAML file
            imgsz=imgsz,  # Image size for testing
            batch=batch_size,  # Batch size
            device=self.device,  # Device (CPU or CUDA)
            split='val'  # Split for testing (train, val, test)
        )
        
    #### TOOLING ####

    def save(self, save_path='best_model.pt'):
        """
        Save the trained model
        """
        self.model.save(save_path)

    def load(self, model_path='best_model.pt'):
        """
        Load pre-trained model weights
        """
        self.model = YOLO(model_path).to(self.device)
        
    def set_system(self, device='cpu'):
        """
        Set the system device (CPU or CUDA)
        cpu: 'cpu'
        gpu: 0
        """
        self.device = device
        self.model.to(self.device)