# 🧪 Google Colab Training Guide — SeeClearly AI

Step-by-step guide to train the Diabetic Retinopathy model on Google Colab.

---

## Step 1: Open Google Colab

Go to: **https://colab.research.google.com**

Click: **New Notebook**

---

## Step 2: Enable GPU Runtime

1. Click **Runtime** (top menu) → **Change runtime type**
2. Select **T4 GPU** (or any GPU available)
3. Click **Save**

**Verify GPU** — paste this in the first cell and run (Shift+Enter):

```python
# CELL 1: Verify GPU
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
```

You should see something like:
```
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Step 3: Install Dependencies

```python
# CELL 2: Install dependencies (most are pre-installed in Colab)
!pip install -q scikit-learn seaborn
```

---

## Step 4: Download APTOS 2019 Dataset

### Option A: Using Kaggle API (Recommended)

```python
# CELL 3A: Setup Kaggle API
# First, go to kaggle.com → Your Profile → Account → Create New API Token
# This downloads a kaggle.json file

from google.colab import files
print("Upload your kaggle.json file:")
uploaded = files.upload()  # Upload the kaggle.json file when prompted
```

```python
# CELL 4A: Configure Kaggle and Download Dataset
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
!kaggle competitions download -c aptos2019-blindness-detection -p /content/aptos_data

# Unzip
!cd /content/aptos_data && unzip -q -o aptos2019-blindness-detection.zip
!ls /content/aptos_data/
```

### Option B: Upload from Google Drive (If you already have the dataset)

```python
# CELL 3B: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from your Drive to Colab local storage (faster I/O)
!cp -r "/content/drive/MyDrive/YOUR_DATASET_FOLDER/aptos2019-blindness-detection" /content/aptos_data/
```

### Option C: Manual Upload

If you have the dataset on your computer:
1. In the Colab left sidebar, click the **📁 folder icon**
2. Create a folder called `aptos_data`
3. Upload `train.csv` and the `train_images/` folder into it

⚠️ **This is slow for 3,662 images. Use Option A or B instead.**

---

## Step 5: Verify Dataset

```python
# CELL 5: Verify dataset structure
import os
import pandas as pd

DATA_PATH = "/content/aptos_data"

# Check files
print("Files in dataset directory:")
for f in os.listdir(DATA_PATH):
    full_path = os.path.join(DATA_PATH, f)
    if os.path.isdir(full_path):
        count = len(os.listdir(full_path))
        print(f"  📁 {f}/ ({count} files)")
    else:
        print(f"  📄 {f}")

# Load and check CSV
df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
print(f"\nTotal samples: {len(df)}")
print(f"\nClass distribution:")
print(df['diagnosis'].value_counts().sort_index())

# Show sample images
import matplotlib.pyplot as plt
import cv2

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
labels = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"}

for cls in range(5):
    sample = df[df['diagnosis'] == cls].iloc[0]
    img_path = os.path.join(DATA_PATH, "train_images", sample['id_code'] + ".png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[cls].imshow(img)
    axes[cls].set_title(f"Class {cls}: {labels[cls]}", fontsize=12, fontweight='bold')
    axes[cls].axis('off')

plt.suptitle("Sample Images from Each DR Stage", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

Expected output:
```
Total samples: 3662

Class distribution:
0    1805
1     370
2     999
3     193
4     295
```

---

## Step 6: Training Code (THE MAIN CELL)

Copy and paste this **entire block** into one Colab cell:

```python
# CELL 6: ==========================================
#          COMPLETE TRAINING PIPELINE
#          ==========================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
import json
import time

# ==========================================
# CONFIGURATION — EDIT THESE IF NEEDED
# ==========================================
DATA_PATH = "/content/aptos_data"          # Path to dataset
MODEL_SAVE_PATH = "/content/dr_efficientnet_b3.keras"  # Where to save model
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE1_EPOCHS = 20    # Head training
PHASE2_EPOCHS = 15    # Fine-tuning
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
LABEL_MAP = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "PDR"}

# ==========================================
# 1. FOCAL LOSS
# ==========================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        focal_weight = alpha * K.pow(1 - y_pred, gamma)
        loss = focal_weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss_fixed

# ==========================================
# 2. DATA LOADING
# ==========================================
print("📂 Loading dataset...")
df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df['id_code'] = df['id_code'].apply(lambda x: x if x.endswith('.png') else x + '.png')
df['diagnosis'] = df['diagnosis'].astype(str)

print(f"   Total samples: {len(df)}")
for i in range(5):
    count = (df['diagnosis'] == str(i)).sum()
    print(f"   {i} ({LABEL_MAP[i]:>10}): {count} samples ({count/len(df)*100:.1f}%)")

# Compute class weights
labels_int = df['diagnosis'].astype(int).values
class_weights = compute_class_weight('balanced', classes=np.unique(labels_int), y=labels_int)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n⚖️  Class weights: {class_weight_dict}")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

image_dir = os.path.join(DATA_PATH, "train_images")

train_gen = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='id_code',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='training',
    seed=42,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='id_code',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation',
    seed=42,
    shuffle=False
)

print(f"\n   Train: {train_gen.samples} samples")
print(f"   Val:   {val_gen.samples} samples")

# ==========================================
# 3. BUILD MODEL
# ==========================================
print("\n🏗️  Building EfficientNetB3...")
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = BatchNormalization(name='bn_head')(x)
x = Dense(512, activation='relu', name='dense_512')(x)
x = Dropout(0.4, name='dropout_1')(x)
x = Dense(256, activation='relu', name='dense_256')(x)
x = Dropout(0.3, name='dropout_2')(x)
output = Dense(5, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=output)
print(f"   Total params: {model.count_params():,}")

# ==========================================
# 4. PHASE 1 — TRAIN HEAD
# ==========================================
print("\n" + "="*60)
print("  PHASE 1: Training Custom Head (Base Frozen)")
print("="*60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
    loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
]

start_time = time.time()

history_p1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

p1_time = time.time() - start_time
print(f"\n✅ Phase 1 done in {p1_time/60:.1f} minutes")
print(f"   Best val_accuracy: {max(history_p1.history['val_accuracy']):.4f}")
print(f"   Best val_auc:      {max(history_p1.history['val_auc']):.4f}")

# ==========================================
# 5. PHASE 2 — FINE-TUNE TOP LAYERS
# ==========================================
print("\n" + "="*60)
print("  PHASE 2: Fine-Tuning Top 30 Layers")
print("="*60)

for layer in model.layers[-30:]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
    loss=categorical_focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

start_time = time.time()

history_p2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

p2_time = time.time() - start_time
print(f"\n✅ Phase 2 done in {p2_time/60:.1f} minutes")
print(f"   Best val_accuracy: {max(history_p2.history['val_accuracy']):.4f}")
print(f"   Best val_auc:      {max(history_p2.history['val_auc']):.4f}")

# Save final model
model.save(MODEL_SAVE_PATH)
print(f"\n💾 Model saved to: {MODEL_SAVE_PATH}")

# Save training history as JSON
combined_history = {}
for key in history_p1.history:
    combined_history[key] = history_p1.history[key] + history_p2.history.get(key, [])

with open('/content/training_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in combined_history.items()}, f)
print("💾 Training history saved to: /content/training_history.json")

total_time = p1_time + p2_time
print(f"\n⏱️  Total training time: {total_time/60:.1f} minutes")
print(f"🎉 TRAINING COMPLETE!")
```

⏱️ **Expected time**: 60-120 minutes on T4 GPU

---

## Step 7: Evaluation (Run after training completes)

```python
# CELL 7: ==========================================
#          COMPREHENSIVE EVALUATION
#          ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

LABEL_LIST = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]

# Load best model
from tensorflow.keras.models import load_model
model = load_model(
    MODEL_SAVE_PATH,
    custom_objects={'focal_loss_fixed': categorical_focal_loss()}
)

# Get predictions
print("🔄 Running predictions on validation set...")
val_gen.reset()
Y_pred_proba = model.predict(val_gen, verbose=1)
y_pred = np.argmax(Y_pred_proba, axis=1)
y_true = val_gen.classes

# --- 1. CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
accuracy = np.trace(cm) / np.sum(cm)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
            ax=ax, linewidths=0.5, square=True)
ax.set_ylabel('Actual', fontweight='bold', fontsize=13)
ax.set_xlabel('Predicted', fontweight='bold', fontsize=13)
ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontweight='bold', fontsize=15)
plt.tight_layout()
plt.savefig('/content/confusion_matrix.png', dpi=150)
plt.show()
print(f"\n✅ Overall Accuracy: {accuracy:.2%}")

# --- 2. CLASSIFICATION REPORT ---
report = classification_report(y_true, y_pred, target_names=LABEL_LIST, digits=4)
print("\n" + "="*60)
print("  CLASSIFICATION REPORT")
print("="*60)
print(report)

# Save report
with open('/content/classification_report.txt', 'w') as f:
    f.write(report)

# --- 3. ROC-AUC CURVES ---
y_true_bin = label_binarize(y_true, classes=[0,1,2,3,4])
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

fig, ax = plt.subplots(figsize=(10, 8))
for i in range(5):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], Y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], linewidth=2.5,
            label=f'{LABEL_LIST[i]} (AUC = {roc_auc:.3f})')

ax.plot([0,1], [0,1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
ax.set_title('ROC Curves (One-vs-Rest)', fontweight='bold', fontsize=15)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/content/roc_curves.png', dpi=150)
plt.show()

# --- 4. TRAINING HISTORY ---
with open('/content/training_history.json', 'r') as f:
    history = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(history['accuracy'], label='Train')
axes[0].plot(history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['loss'], label='Train', color='orange')
axes[1].plot(history['val_loss'], label='Val', color='red')
axes[1].set_title('Focal Loss', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history['auc'], label='Train', color='green')
axes[2].plot(history['val_auc'], label='Val', color='teal')
axes[2].set_title('AUC', fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Training History', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('/content/training_history.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("  ALL EVALUATION COMPLETE!")
print("="*60)
print("  Files generated:")
print("    📊 /content/confusion_matrix.png")
print("    📋 /content/classification_report.txt")
print("    📈 /content/roc_curves.png")
print("    📉 /content/training_history.png")
print("    🧠 /content/dr_efficientnet_b3.keras")
```

---

## Step 8: Download Model & Results

```python
# CELL 8: Download all files to your computer
from google.colab import files

# Download the trained model (MOST IMPORTANT)
files.download('/content/dr_efficientnet_b3.keras')

# Download evaluation outputs
files.download('/content/confusion_matrix.png')
files.download('/content/classification_report.txt')
files.download('/content/roc_curves.png')
files.download('/content/training_history.png')
files.download('/content/training_history.json')

print("✅ All files downloaded!")
print("\n📌 Next: Place dr_efficientnet_b3.keras in your project's models/ folder")
```

---

## Step 9: Put Model in Your Project

After downloading, move the file:

```
dr_efficientnet_b3.keras  →  c:\Projects\seeclearly_ai\models\dr_efficientnet_b3.keras
```

Then start your local app:
```bash
# Terminal 1
python backend/app.py

# Terminal 2
npm run dev
```

The status badge will change from **"Mock Mode"** → **"Backend Online"**! 🎉

---

## ⚠️ Common Errors & Fixes

### "ResourceExhaustedError: OOM when allocating tensor"
**Fix**: Reduce batch size in CELL 6:
```python
BATCH_SIZE = 16  # was 32
```

### "train_images folder not found"
**Fix**: Check your dataset path. The folder structure should be:
```
/content/aptos_data/
├── train.csv
├── train_images/
│   ├── 000c1434d8d7.png
│   ├── ...
```

### "Kaggle API Token not found"
**Fix**: 
1. Go to kaggle.com → Account → Create New API Token
2. Download kaggle.json
3. Upload it in Cell 3A

### "Low accuracy (< 60%)"
Possible causes:
- Data path wrong (model trains on broken/zero images)
- Not using class weights
- Training for too few epochs (let EarlyStopping decide)

### "Colab disconnects during training"
**Fix**: 
- Keep the browser tab active
- Use Google Colab Pro for longer sessions
- Or: save checkpoints to Google Drive instead of /content/
```python
# Change MODEL_SAVE_PATH to:
MODEL_SAVE_PATH = "/content/drive/MyDrive/dr_efficientnet_b3.keras"
```
