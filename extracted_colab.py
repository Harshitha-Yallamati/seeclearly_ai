!pip install tensorflow

import os
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

# ---

pip install kaggle

# ---

import os
os.environ['KAGGLE_API_TOKEN'] = "KGAT_0987b67ee25e73c19a61a9165902b715"

# ---

!kaggle competitions list

# ---

!kaggle competitions download -c aptos2019-blindness-detection
!unzip aptos2019-blindness-detection.zip -d aptos_data

# ---

data_path = "/content/aptos_data"

train_df = pd.read_csv(os.path.join(data_path, "train.csv"))

# Add .png extension
train_df['id_code'] = train_df['id_code'] + ".png"

# ---

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# ---

# IMPORTANT FIX
train_df['diagnosis'] = train_df['diagnosis'].astype(str)

train_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="/content/aptos_data/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(128,128),
    class_mode="categorical",
    batch_size=32,
    subset="training"
)

val_data = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="/content/aptos_data/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(128,128),
    class_mode="categorical",
    batch_size=32,
    subset="validation"
)

# ---

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['diagnosis']),
    y=train_df['diagnosis']
)

class_weights = dict(enumerate(class_weights))

# ---

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ---

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights
)

# ---

loss, acc = model.evaluate(val_data)
print("Accuracy:", acc)

# ---

import cv2

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def predict_image(img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return np.argmax(pred)

# ---

def get_severity(label):
    return ["No DR", "Mild", "Moderate", "Severe", "PDR"][label]

# ---

img_path = "/content/sorted_test_images/PDR/b21c1ff1bb64.png"

result = predict_image(img_path)
print("Prediction:", get_severity(result))

# ---

import os
import shutil
import numpy as np
import cv2

# Define label names
label_map = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]

test_dir = "/content/aptos_data/test_images"
output_dir = "/content/sorted_test_images"

# Create output folders
for label in label_map:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Preprocess function (same as training)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

# Predict + Move images
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    if img_name.endswith(".png"):
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        label = np.argmax(pred)

        label_name = label_map[label]

        # Move image to respective folder
        dest_path = os.path.join(output_dir, label_name, img_name)
        shutil.copy(img_path, dest_path)

print("✅ Images sorted successfully!")