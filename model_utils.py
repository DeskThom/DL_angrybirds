from ultralytics import YOLO
import torch


class YOLOModel:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        self.device = device
        # Load YOLO model
        self.model = YOLO(model_path).to(self.device)  # Load YOLO model

    def train(self, data_yaml='data.yaml', epochs=50, imgsz=640, batch_size=32):
        """
        Use YOLO's built-in training functionality
        """
        # Calling the model's train function directly
        self.model.train(
            data=data_yaml,  # Path to dataset YAML file
            epochs=epochs,  # Number of epochs
            imgsz=imgsz,  # Image size for training
            batch=batch_size,  # Batch size
            device=self.device  # Device (CPU or CUDA)
        )

    def validate(self, val_loader):
        """
        Custom validation loop (if needed)
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                loss = self.model(images, labels)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def test(self, test_loader):
        """
        Custom testing loop (if needed)
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predictions = outputs[0]  # Get predictions
                _, predicted = torch.max(predictions, 1)  # Get the class with the max score
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

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