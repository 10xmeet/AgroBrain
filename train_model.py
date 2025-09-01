import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import os

# Dataset Paths
TRAIN_DIR = 'dataset/PlantVillage'
VAL_DIR = 'dataset/PlantVillage'

# Image Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),  # Adjusted to match VGG-16 input size
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),  # Adjusted to match VGG-16 input size
    batch_size=32,
    class_mode='categorical'
)

# Load the VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)  # Dynamically adjust the number of classes

# Create the new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Update dataset paths
TRAIN_DIR = 'dataset/PlantVillage'
VAL_DIR = 'dataset/PlantVillage'

# Model Checkpoint to save best weights
checkpoint = ModelCheckpoint('plant_disease_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train Model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Adjust based on dataset size
    callbacks=[checkpoint]
)

# Save Final Model
model.save('final_plant_disease_model.h5')

print("Training complete. Best model saved as 'plant_disease_model.h5'")
