from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image  # Add this import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from math import ceil

from google.colab import drive
drive.mount('/content/drive')

# Set the directories for the training and test images
TRAIN_DIR = '/content/drive/MyDrive/Alzheimer_s Dataset/train'
TEST_DIR = '/content/drive/MyDrive/Alzheimer_s Dataset/test'

# Set the image size and batch size
IMAGE_SIZE = 176
BATCH_SIZE = 10

datagen = ImageDataGenerator(rescale=1./255)

CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Function to load images from a folder
def load_images_from_folder(folder, klass, num_images=4):
    images = []
    for filename in os.listdir(os.path.join(folder, klass))[:num_images]:
        img_path = os.path.join(folder, klass, filename)
        img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='grayscale')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    return np.vstack(images)

num_images_to_display = 8  # or any other number that you prefer

# Create a subplot grid
fig, axes = plt.subplots(nrows=len(CLASSES), ncols=num_images_to_display, figsize=(20, 10))

# Load and display the images
for i, klass in enumerate(CLASSES):
    imgs = load_images_from_folder(TRAIN_DIR, klass, num_images_to_display)
    for j in range(num_images_to_display):
        if j < imgs.shape[0]:  # Check if the image index is within the range of loaded images
            ax = axes[i, j]
            img = imgs[j] / 255.0
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.axis('off')
        else:
            axes[i, j].axis('off')  # Hide axes if there are no more images to display

# Set class names as titles for the first column
for ax, klass in zip(axes[:, 0], CLASSES):
    ax.set_ylabel(klass, rotation=90, size='large')

plt.tight_layout()
plt.show()

# Define the CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Build the CNN model
model_cnn = build_model()

# Compile the model
model_cnn.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Create data generators
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Train the model
history_cnn = model_cnn.fit_generator(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // BATCH_SIZE))

# Save the model
model_cnn.save('cnn_model.h5')

# Evaluate the model on the test set
evaluate_cnn = model_cnn.evaluate_generator(validation_generator, steps=validation_generator.samples // BATCH_SIZE, verbose=1)
print('CNN Model Loss: {}, CNN Model Accuracy: {}'.format(evaluate_cnn[0], evaluate_cnn[1]))

# Predict the test set
validation_generator.reset()
predictions_cnn = model_cnn.predict(validation_generator, steps=len(validation_generator))

# Convert predictions to class indices
predicted_classes_cnn = np.argmax(predictions_cnn, axis=1)

# Get the true class indices
true_classes_cnn = validation_generator.classes

# Get the confusion matrix
cm_cnn = confusion_matrix(true_classes_cnn, predicted_classes_cnn)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Generate and plot the confusion matrix for CNN
confusion_mtx_cnn = confusion_matrix(true_classes_cnn, predicted_classes_cnn)
plt.figure(figsize=(10, 8))
class_labels_cnn = list(validation_generator.class_indices.keys())
sns.heatmap(confusion_mtx_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_cnn, yticklabels=class_labels_cnn)
plt.title('CNN Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Define the advanced CNN model
def build_advanced_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))  # Assuming 4 classes
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# Build the advanced CNN model
model_advanced_cnn = build_advanced_model()

# Compile the advanced CNN model
model_advanced_cnn.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_generator_advanced = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

# Validation generator
validation_generator_advanced = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Important for correct label mapping
)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator_advanced.classes),
    y=train_generator_advanced.classes
)

class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Callbacks for advanced CNN model
callbacks_advanced = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint('best_advanced_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the advanced CNN model with class weights
history_advanced_cnn = model_advanced_cnn.fit(
    train_generator_advanced,
    steps_per_epoch=train_generator_advanced.samples // BATCH_SIZE,
    epochs=50,
    validation_data=validation_generator_advanced,
    validation_steps=validation_generator_advanced.samples // BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks_advanced
)

# Evaluate the advanced CNN model on the test set
evaluate_advanced_cnn = model_advanced_cnn.evaluate(
    validation_generator_advanced,
    steps=validation_generator_advanced.samples // BATCH_SIZE,
    verbose=1
)
print('Advanced CNN Model Loss: {}, Advanced CNN Model Accuracy: {}'.format(evaluate_advanced_cnn[0], evaluate_advanced_cnn[1]))

# Plot training history for advanced CNN model
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_advanced_cnn.history['loss'], label='Training Loss')
plt.plot(history_advanced_cnn.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss (Advanced CNN)')

plt.subplot(1, 2, 2)
plt.plot(history_advanced_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_advanced_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy (Advanced CNN)')

plt.show()

# Correctly calculate the number of steps to cover all samples in the validation set
validation_steps_advanced = ceil(validation_generator_advanced.samples / BATCH_SIZE)

# Reset the validation generator to ensure the order of the samples
validation_generator_advanced.reset()

# Predict the validation set results for advanced CNN
predictions_advanced = model_advanced_cnn.predict(validation_generator_advanced, steps=validation_steps_advanced)

# Convert predictions to class indices
predicted_classes_advanced = np.argmax(predictions_advanced, axis=1)

# Ensure that true_classes and predicted_classes have the same length
true_classes_advanced = validation_generator_advanced.classes[:len(predicted_classes_advanced)]

# Generate and plot the confusion matrix for advanced CNN
confusion_mtx_advanced = confusion_matrix(true_classes_advanced, predicted_classes_advanced)
plt.figure(figsize=(10, 8))
class_labels_advanced = list(validation_generator_advanced.class_indices.keys())
sns.heatmap(confusion_mtx_advanced, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_advanced, yticklabels=class_labels_advanced)
plt.title('Advanced CNN Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Generate and plot the confusion matrix for advanced CNN
confusion_mtx_advanced = confusion_matrix(true_classes_advanced, predicted_classes_advanced)
plt.figure(figsize=(10, 8))
class_labels_advanced = list(validation_generator_advanced.class_indices.keys())
sns.heatmap(confusion_mtx_advanced, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_advanced, yticklabels=class_labels_advanced)
plt.title('Advanced CNN Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Define class labels
class_labels_advanced = list(validation_generator_advanced.class_indices.keys())

# ...

# Plot the confusion matrix for Advanced CNN with class labels
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx_advanced, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels_advanced, yticklabels=class_labels_advanced)
plt.title('Advanced CNN Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the directories for the training and test images
TRAIN_DIR_RESNET = '/content/drive/MyDrive/Alzheimer_s Dataset/train'
TEST_DIR_RESNET = '/content/drive/MyDrive/Alzheimer_s Dataset/test'

# Set the image size and batch size
IMAGE_SIZE_RESNET = 176
BATCH_SIZE_RESNET = 10
NUM_CLASSES_RESNET = 4
EPOCHS_RESNET = 50

datagen_resnet = ImageDataGenerator(rescale=1./255)

CLASSES_RESNET = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Function to load images from a folder
def load_images_from_folder(folder, klass, num_images=4):
    images = []
    for filename in os.listdir(os.path.join(folder, klass))[:num_images]:
        img_path = os.path.join(folder, klass, filename)
        img = image.load_img(img_path, target_size=(IMAGE_SIZE_RESNET, IMAGE_SIZE_RESNET), color_mode='grayscale')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
    return np.vstack(images)

num_images_to_display_resnet = 8  # or any other number that you prefer

# Create a subplot grid
fig_resnet, axes_resnet = plt.subplots(nrows=len(CLASSES_RESNET), ncols=num_images_to_display_resnet, figsize=(20, 10))

# Load and display the images
for i, klass in enumerate(CLASSES_RESNET):
    imgs_resnet = load_images_from_folder(TRAIN_DIR_RESNET, klass, num_images_to_display_resnet)
    for j in range(num_images_to_display_resnet):
        if j < imgs_resnet.shape[0]:  # Check if the image index is within the range of loaded images
            ax_resnet = axes_resnet[i, j]
            img_resnet = imgs_resnet[j] / 255.0
            ax_resnet.imshow(np.squeeze(img_resnet), cmap='gray')
            ax_resnet.axis('off')
        else:
            axes_resnet[i, j].axis('off')  # Hide axes if there are no more images to display

# Set class names as titles for the first column
for ax_resnet, klass in zip(axes_resnet[:, 0], CLASSES_RESNET):
    ax_resnet.set_ylabel(klass, rotation=90, size='large')

plt.tight_layout()
plt.show()

# Build the ResNet50-based model
def build_resnet_model():
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE_RESNET, IMAGE_SIZE_RESNET, 3))
    x_resnet = base_model_resnet.output
    x_resnet = GlobalAveragePooling2D()(x_resnet)
    x_resnet = Dense(1024, activation='relu')(x_resnet)
    x_resnet = Dropout(0.5)(x_resnet)
    predictions_resnet = Dense(NUM_CLASSES_RESNET, activation='softmax')(x_resnet)
    model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)
    return model_resnet

# Build the ResNet50 model
model_resnet = build_resnet_model()

# Compile the ResNet50 model
model_resnet.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create data generators for ResNet50 model
train_generator_resnet = datagen_resnet.flow_from_directory(
    TRAIN_DIR_RESNET,
    target_size=(IMAGE_SIZE_RESNET, IMAGE_SIZE_RESNET),
    batch_size=BATCH_SIZE_RESNET,
    color_mode='rgb',  # Note: ResNet50 uses RGB images
    class_mode='categorical')

validation_generator_resnet = datagen_resnet.flow_from_directory(
    TEST_DIR_RESNET,
    target_size=(IMAGE_SIZE_RESNET, IMAGE_SIZE_RESNET),
    batch_size=BATCH_SIZE_RESNET,
    color_mode='rgb',  # Note: ResNet50 uses RGB images
    class_mode='categorical',
    shuffle=False
)

# Train the ResNet50 model
history_resnet = model_resnet.fit_generator(
    train_generator_resnet,
    steps_per_epoch=max(1, train_generator_resnet.samples // BATCH_SIZE_RESNET),
    epochs=EPOCHS_RESNET,
    validation_data=validation_generator_resnet,
    validation_steps=max(1, validation_generator_resnet.samples // BATCH_SIZE_RESNET))

# Save the ResNet50 model
model_resnet.save('resnet_model.h5')

# Evaluate the ResNet50 model on the test set
evaluate_resnet = model_resnet.evaluate_generator(validation_generator_resnet, steps=validation_generator_resnet.samples // BATCH_SIZE_RESNET, verbose=1)
print('ResNet50 Model Loss: {}, ResNet50 Model Accuracy: {}'.format(evaluate_resnet[0], evaluate_resnet[1]))

# Predict the test set for ResNet50 model
validation_generator_resnet.reset()
predictions_resnet = model_resnet.predict(validation_generator_resnet, steps=len(validation_generator_resnet))

# Convert predictions to class indices for ResNet50 model
predicted_classes_resnet = np.argmax(predictions_resnet, axis=1)

# Get the true class indices for ResNet50 model
true_classes_resnet = validation_generator_resnet.classes

# Get the confusion matrix for ResNet50 model
cm_resnet = confusion_matrix(true_classes_resnet, predicted_classes_resnet)

# Plot the confusion matrix for ResNet50 model
plt.figure(figsize=(10, 8))
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues')
plt.title('ResNet50 Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()
