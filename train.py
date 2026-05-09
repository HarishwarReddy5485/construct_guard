import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
train_dir = 'G:/project-main/dataset/train'  # Use full path to avoid issues
model_save_path = 'model_building_defects_vgg16.h5'

# -----------------------------
# Hyperparameters
# -----------------------------
img_height, img_width = 224, 224
batch_size = 32
epochs = 20
learning_rate = 1e-4
num_classes = 3  # Crack, Flake, Roof

# -----------------------------
# Check dataset folder
# -----------------------------
print("Checking dataset folder...")
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))])
        print(f"Class '{class_name}': {num_images} images")

# -----------------------------
# Data augmentation with validation split
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# Build model
# -----------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model.layers:
    layer.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // batch_size),
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

print(f"Model saved to {model_save_path}")

# -----------------------------
# Plot training history
# -----------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.show()