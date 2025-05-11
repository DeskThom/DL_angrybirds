from ultralytics import YOLO
import yaml, json
import os, shutil

# YOLO MODEL CLASS - V11 = BASE
class YOLOModel:
    def __init__(self, model_path='models/pretrained/yolo11n.pt', device='cpu'):
        self.device = device
        self.model = YOLO(model_path).to(self.device)

    ### TRAIN, TEST, VAL ###
    def train(self, data_yaml='data.yaml', epochs=50, imgsz=640, batch_size=32, cls=0.5, iou=0.7, conf=0.25, debug_mode=False, seed=42):
        if debug_mode:
            self.model.train(
                seed=seed,
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=self.device,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                translate=0,
                scale=0,
                fliplr=0,
                mosaic=0,
                erasing=0
            )
        else:
            self.model.train(
                seed=seed,
                data=data_yaml,
                device=self.device,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                cls=cls,
                iou=iou,
                conf=conf,
                fliplr=0.5,
                degrees=90,
                perspective=0.0005,
                shear=10,
                hsv_h=0.1,
                hsv_v=0.5,
                hsv_s=0.2,
                scale=0.5
            )

    def transfer_training(self, additionaldata_yaml, originaldata_yaml, intermediate_weights='models/phase1.pt', debug_mode=False,  seed=42, **kwargs):
        """
        Transfer training: Train on one dataset, save best weights, then fine-tune on another dataset.
        """
        print(f"\n🔁 Phase 1: Training on {additionaldata_yaml}")
        self.train(data_yaml=additionaldata_yaml, debug_mode=debug_mode, **kwargs)
        self.save(save_path=intermediate_weights)

        print(f"\n🚀 Phase 2: Fine-tuning on {originaldata_yaml} using weights from Phase 1")
        self.load(model_path=intermediate_weights)
        self.train(data_yaml=originaldata_yaml, debug_mode=debug_mode, **kwargs)

    def grid_search(self, data_yaml='data.yaml', epochs=50, param_grid=None, result_file='grid_search_results.json', seed=42):
        from itertools import product
        if param_grid is None:
            print("No parameter grid provided. Using default values.")
            with open('param_grid.json', 'r') as f: 
                param_grid = yaml.safe_load(f)

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        print(f"Running grid search over {len(combinations)} combinations...")

        results = []
        for i, params in enumerate(combinations):
            print(f"\n➡ Running combination {i + 1}/{len(combinations)}: {params}")
            metrics = self.model.train(
                data=data_yaml,
                epochs=epochs,
                device=self.device,
                seed=seed,
                **params
            )
            result = {
                "params": params,
                "mAP_50": metrics.box.map50,
                "mAP_50_95": metrics.box.map,
            }
            results.append(result)

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)

        print("\n✅ Grid search complete.")

    def predict(self, image_path, conf=0.25):
        results = self.model.predict(
            source=image_path,
            conf=conf,
            device=self.device,
            save=True,
            save_txt=True,
            save_conf=True,
            show=True
        )
        return results

    def save(self, save_path='best_model.pt'):
        best_path = 'runs/detect/train/weights/best.pt'
        if os.path.exists(best_path):
            shutil.copy(best_path, save_path)
            print(f"✅ Best model weights saved to {save_path}")
        else:
            print("❌ No best.pt found. Has training been run?")

    def load(self, model_path='best_model.pt'):
        self.model = YOLO(model_path).to(self.device)

    def set_system(self, device='cpu'):
        self.device = device
        self.model.to(self.device)
