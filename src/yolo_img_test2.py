import os
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# ================= CONFIG =================
CONF_THRESHOLD = 0.80
BATCH_SIZE = 16

MODEL_PATH = r"F:\Trupti\best.pt"
IMAGE_ROOT = r"F:\Trupti\Ads"
OUTPUT_ROOT = r"F:\Trupti\output"
EXCEL_PATH = os.path.join(OUTPUT_ROOT, "detection_report.xlsx")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)
class_names = model.names

print("Model loaded")
print("Classes:", class_names)

# ================= GLOBAL METRICS =================
TP = FP = FN = TN = 0

# ================= CLASS-WISE METRICS (UNCHANGED) =================
class_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

# ================= CHANNEL-WISE METRICS =================
channel_stats = defaultdict(lambda: {
    "total": 0,
    "detected": 0,
    "not_detected": 0,
    "TP": 0,
    "FN": 0,
    "TN": 0,
    "FP": 0,
})

# ================= COLLECT IMAGES =================
image_entries = []  # (image_path, relative_folder)

for root, _, files in os.walk(IMAGE_ROOT):
    for file in files:
        if file.lower().endswith(IMG_EXTS):
            full_path = os.path.join(root, file)
            rel_folder = os.path.relpath(root, IMAGE_ROOT)
            image_entries.append((full_path, rel_folder))

print(f"Total images found: {len(image_entries)}")

# ================= INFERENCE =================
for i in range(0, len(image_entries), BATCH_SIZE):
    batch = image_entries[i:i + BATCH_SIZE]
    batch_paths = [x[0] for x in batch]

    results = model(batch_paths, conf=CONF_THRESHOLD, verbose=False)

    for (img_path, rel_folder), result in zip(batch, results):
        img = cv2.imread(img_path)
        detected = False
        channel = rel_folder

        # Update total count
        channel_stats[channel]["total"] += 1

        # Output directories
        base_out = os.path.join(OUTPUT_ROOT, rel_folder)
        det_dir = os.path.join(base_out, "Detected")
        nodet_dir = os.path.join(base_out, "Not_detected")
        os.makedirs(det_dir, exist_ok=True)
        os.makedirs(nodet_dir, exist_ok=True)

        if result.boxes is not None and len(result.boxes) > 0:
            detected = True

            TP += 1
            channel_stats[channel]["detected"] += 1
            channel_stats[channel]["TP"] += 1

            for b in result.boxes:
                cls = int(b.cls)
                conf = float(b.conf)

                class_stats[class_names[cls]]["TP"] += 1

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{class_names[cls]} {conf:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        else:
            FN += 1
            TN += 1

            channel_stats[channel]["not_detected"] += 1
            channel_stats[channel]["FN"] += 1
            channel_stats[channel]["TN"] += 1

        save_dir = det_dir if detected else nodet_dir
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)

# ================= SUMMARY SHEET =================
summary_rows = []
sr_no = 1

for channel, s in sorted(channel_stats.items()):
    total = s["total"]
    TPc = s["TP"]
    FNc = s["FN"]
    TNc = s["TN"]
    FPc = s["FP"]

    accuracy = (TPc / total * 100) if total else 0
    FAR = (FPc / (FPc + TNc) * 100) if (FPc + TNc) else 0
    FRR = (FNc / (TPc + FNc) * 100) if (TPc + FNc) else 0

    summary_rows.append({
        "Sr.No": sr_no,
        "Channel Name": os.path.basename(channel),
        "Total Images": total,
        "Detected Images": s["detected"],
        "Not Detected Images": s["not_detected"],
        "Detection Accuracy (%)": round(accuracy, 2),
        "False Detection": FPc,
        "FN (Missed)": FNc,
        "TN (Correct Rejection)": TNc,
        "FAR (%)": round(FAR, 2),
        "FRR (%)": round(FRR, 2),
    })

    sr_no += 1

summary_df = pd.DataFrame(summary_rows)

# ================= CLASS-WISE REPORT (UNCHANGED) =================
class_df = pd.DataFrame.from_dict(class_stats, orient="index").reset_index()
class_df.columns = ["Class", "TP", "FP", "FN", "TN"]

# ================= SAVE EXCEL =================
with pd.ExcelWriter(EXCEL_PATH) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    class_df.to_excel(writer, sheet_name="Class_Wise_Report", index=False)

print("\nExcel saved at:", EXCEL_PATH)
print("Done.")
