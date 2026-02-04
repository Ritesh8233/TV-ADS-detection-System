from ultralytics import YOLO

def train_model():
    # Load YOLO model (Nano or Medium depending on GPU)
    model = YOLO(r"yolov8n.pt")  # Change to yolov8m.pt if GPU allows

    # Train
    model.train(
        data=r"D:\Inditronics_advance\Yoolo1\yolo_dataset\New dataset\data.yaml",

        # Image & Batch
        imgsz=640,
        batch=4,
        fraction=0.2,  # Adjust if you get CUDA OOM

        # Epochs & Learning Rate
        epochs=30,
        optimizer="AdamW",
        lr0=0.001,          # Base learning rate
        lrf=0.01,           # Final LR factor
        weight_decay=0.0005,

        # Regularization & Augmentation
        dropout=0.1,        # Regularization
        scale=0.1,          # Random scale
        shear=0.0,          # Shear augmentation
        perspective=0.0005, # Small perspective change

        # Disable color augmentations (for logo training clarity)
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # Training strategy
        patience=5,        # Early stopping
        project="ARM demo",
        name="demo",
        device=0,           # CUDA:0
        workers=8,
        save_period=5,      # Save every 5 epochs
        cos_lr=True,        # Cosine LR scheduler
        cache=False,        # Avoid RAM cache (use only if fast disk)
        resume=False        # Start fresh training
    )

    # Evaluate (on validation/test set defined in data.yml)
    metrics = model.val(split="test")
    print(metrics)

    # Export (optional, TensorFlow SavedModel for deployment)
    model.export(format="saved_model")


if __name__ == "__main__":
    train_model()
