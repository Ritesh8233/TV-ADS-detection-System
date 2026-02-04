import make_yaml
import os

# Paths
base_path = r"D:\Inditronics_advance\Yoolo1\yolo_dataset"
yaml_path = os.path.join(base_path, "data.yaml")

# YOLO data.yaml content
data = {
    "train": os.path.join(base_path, "images", "train").replace("\\", "/"),
    "val": os.path.join(base_path, "images", "val").replace("\\", "/"),
    "test": os.path.join(base_path, "images", "test").replace("\\", "/"),
    "nc": 4,
    "names": ['Tata', 'TataMotors', 'TataCarlogo', 'Tatalogo']
}

# Save yaml file
with open(yaml_path, "w") as f:
    make_yaml.dump(data, f, default_flow_style=False)

print(f"✅ data.yaml created at: {yaml_path}")
