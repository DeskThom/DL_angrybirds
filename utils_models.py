from ultralytics import YOLO
import yaml, json


# YOLO MODEL CLASS - V11 = BASE
class YOLOModel:
    def __init__(self, model_path='models/pretrained/yolo11n.pt', device='cpu'):
        self.device = device
        # Load YOLO model - Pretrained
        self.model = YOLO(model_path).to(self.device)  # Load YOLO model

    ### TRAIN, TEST, VAL ###
    def train(self, data_yaml='data.yaml', epochs=50, imgsz=640, batch_size=32, cls = 0.5, iou=0.7, conf=0.25, debug_mode=False):
        """
        Use YOLO's built-in training functionality with standard settings. The train function includes a validation step.
        
        Input:
        - debug_mode = True: Turn off all data augmentations
        - Takes a YAML file as input for the dataset
        - epochs, imgsz, batch_size, cls, iou, conf - Adjustable hyperparameters and training settings (details: https://docs.ultralytics.com/modes/train/#train-settings)
        """
        if debug_mode:
            # Debug mode: Turn off all data augmentations
            self.model.train(
                data=data_yaml,  # Path to dataset YAML file
                epochs=epochs,  # Number of epochs
                imgsz=imgsz,  # Image size for training
                batch=batch_size,  # Batch size
                device=self.device,  # Device (CPU or CUDA)
                hsv_h = 0,
                hsv_s = 0,
                hsv_v = 0,
                translate = 0,
                scale = 0,
                fliplr = 0,
                mosaic = 0,
                erasing = 0
            )
        else:
        # Calling the model's train function directly
            self.model.train(
                data=data_yaml,  # Path to dataset YAML file
                device=self.device,  # Device (CPU or CUDA)
                
                # Training settings
                epochs=epochs,  # Number of epochs
                batch=batch_size,  # Batch size
                
                # Hyperparameters - Adjustable
                imgsz=imgsz,  # Image size for training
                cls=cls, # Class confidence threshold
                iou=iou,
                conf=conf,  # Confidence threshold for predictions
                
                # Standard data augmentations
                fliplr = 0.5,
                degrees = 90, # randomly rotates between -90 degrees and 90 degrees
                perspective = 0.0005,
                shear = 10,
                hsv_h = 0.1,
                hsv_v = 0.5,
                hsv_s = 0.2,
                scale = 0.5

            )

    def grid_search(self, data_yaml='data.yaml', epochs=50, imgsz=640, batch_size=32, param_grid=None, result_file='grid_search_results.json'):
        """
        Perform grid search over hyperparameters using the existing train() function.

        Args:
            data_yaml (str): Path to YAML dataset config.
            epochs (int): Number of epochs.
            imgsz (int): Image size.
            batch_size (int): Batch size.
            debug_mode (bool): If True, disables augmentation.
            param_grid (dict): Dictionary of hyperparameters to search. Example:
                {
                    'lr0': [0.01, 0.001],
                    'momentum': [0.9, 0.937],
                    'weight_decay': [0.0005, 0.001]
                }
                
        Note/improvements:
        - The current function does NOT allow the user to turn off the automatic optimizer/lr/weight_decay/momentum settings.
        """
        from itertools import product

        if param_grid is None:
            print("No parameter grid provided. Using default values.")
            with open('param_grid.json', 'r') as f:
                param_grid = yaml.safe_load(f)

        # Generate all combinations
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        print(f"Running grid search over {len(combinations)} combinations...")
    
        # Store performance metrics for each combination
        results = []

        for i, params in enumerate(combinations):
            print(f"\n➡ Running combination {i + 1}/{len(combinations)}: {params}")

            # Call existing train method with extra hyperparameters
            metrics = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=self.device,
                **params                  # Unpack hyperparameters from the dictionary
            )

        
        # Example: Save mAP and other metrics
        result = {
            "params": params,
            "mAP_50": metrics.box.map50,
            "mAP_50_95": metrics.box.map,
            "loss": metrics.loss 
        }
        results.append(result)
        # Save results to a JSON file for later analysis
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)

        print("\n✅ Grid search complete.")

# TEST/VAL PLACEHOLDERS - VALDIDATION IS DONE DURING TRAINING
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
    def predict(self, image_path, conf=0.25):
        # Single prediction
        results = self.model.predict(
            source=image_path,  # Path to image or video file
            conf=conf,  # Confidence threshold for predictions
            device=self.device,  # Device (CPU or CUDA)
            save=True,  # Save the predictions
            save_txt=True,  # Save the predictions in YOLO format
            save_conf=True,  # Save confidence scores
            show=True  # Show the predictions on the image
        )

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