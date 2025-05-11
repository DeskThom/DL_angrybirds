import os
import cv2
import torch
import random
import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

setup_logger()

class DetectronModel:
    def __init__(self, seed=42):

        # Set reproducibility seed
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Print environment info
        print("✅ PyTorch version:", torch.__version__)
        print("✅ CUDA available:", torch.cuda.is_available())

        # === Register COCO-style datasets ===
        # Could be a dataloader in the future but still have to look into it
        register_coco_instances("scarecrow_train", {}, "scarecrow_coco_dataset/train/_annotations.coco.json", "scarecrow_coco_dataset/train")
        register_coco_instances("scarecrow_val", {}, "scarecrow_coco_dataset/valid/_annotations.coco.json", "scarecrow_coco_dataset/valid")
        register_coco_instances("scarecrow_test", {}, "scarecrow_coco_dataset/test/_annotations.coco.json", "scarecrow_coco_dataset/test")

        self.output_dir = "./output"
        os.makedirs(self.output_dir, exist_ok=True)

    def build_train_cfg(self):
        
        cfg = get_cfg()  # Create a new default config object
        cfg.SEED = self.seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        # Load base config for Faster R-CNN with ResNet-50 and FPN from the model zoo

        cfg.DATASETS.TRAIN = ("scarecrow_train",)  
        cfg.DATASETS.TEST = ("scarecrow_val",)     
        cfg.DATALOADER.NUM_WORKERS = 8             # Number of CPU workers for data loading

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        # Use pre-trained COCO weights to initialize the model (helps speed up convergence)

        cfg.SOLVER.IMS_PER_BATCH = 2               # Number of images per batch (depends on GPU memory)
        cfg.SOLVER.BASE_LR = 0.0025                # Base learning rate
        cfg.SOLVER.MAX_ITER = 3000                 # Total training iterations
        cfg.SOLVER.STEPS = (1000, 2000)            # Iteration steps to reduce learning rate (LR decay)
        cfg.SOLVER.WARMUP_ITERS = 200              # Gradual warm-up for learning rate during initial steps

        cfg.TEST.EVAL_PERIOD = 500                 # Run evaluation every 500 iterations during training

        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 64    # Number of RPN samples per image (for loss computation)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Number of samples per image in ROI heads
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1        # Number of classes in your dataset (excluding background)

        cfg.INPUT.MIN_SIZE_TRAIN = (400, 600, 800) # Randomly choose one of these sizes during training
        cfg.INPUT.MAX_SIZE_TRAIN = 1000            # Maximum size of the longer image side during training
        cfg.INPUT.MIN_SIZE_TEST = 800              # Minimum size for test images

        #cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True  # Skip images without annotations

        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3  # NMS threshold during inference (lower = stricter suppression)

        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0    # Max gradient norm
        cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0     # Type of norm (L2 norm)

        cfg.SOLVER.CHECKPOINT_PERIOD = 500           # Save model checkpoint every 500 iterations

        cfg.OUTPUT_DIR = self.output_dir             # Where to save logs and model checkpoints
        self.cfg = cfg

    def train(self):
        # === Trainer Initialization and Training Start ===
        trainer = DefaultTrainer(self.cfg)               # Create a trainer with the given config
        trainer.train()                                  # Start training

    def build_eval_cfg(self):
        # Load the same config used for training

        cfg = get_cfg() 
        cfg.SEED = self.seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TEST = ("scarecrow_val",)  # Your validation dataset
        cfg.MODEL.WEIGHTS = os.path.join(self.output_dir, "model_final.pth")  # Load the trained model
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Set the number of classes correctly
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4  # Fixes overlapping boxes
        self.cfg = cfg

    def evaluate(self):
        # Create evaluator and test data loader
        evaluator = COCOEvaluator("scarecrow_val", self.cfg, False, output_dir=self.output_dir)
        val_loader = build_detection_test_loader(self.cfg, "scarecrow_val")
        predictor = DefaultPredictor(self.cfg)
        return inference_on_dataset(predictor.model, val_loader, evaluator)

    def build_predict_cfg(self):
        # Load configuration

        cfg = get_cfg() 
        cfg.SEED = self.seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "output/model_final.pth"  
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for inference
        cfg.DATASETS.TEST = ("scarecrow_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Set the number of classes in your dataset
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1  # Fixes overlapping boxes
        self.cfg = cfg

    def predict_and_visualize(self, image_path):
        # Initialize the predictor
        predictor = DefaultPredictor(self.cfg)
        im = cv2.imread(image_path)
        outputs = predictor(im)

        # Visualize the predictions
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TEST[0]), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        return out.get_image()[:, :, ::-1]
