import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


path = '/mnt/nyan/'


dataset = 'data.csv'

class ImageClassifier:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        
    def build_model(self):
        """Build model using transfer learning with MobileNetV2"""
        # Load pre-trained MobileNetV2 (lightweight and fast)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers (faster training)
        base_model.trainable = False
        
        # Build classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def load_images_from_directory(self, image_dir, labels_dict):
        """
        Load images and labels
        labels_dict: dictionary mapping filename to label (0 or 1)
        e.g., {'img1.jpg': 0, 'img2.jpg': 1, ...}
        """
        images = []
        labels = []
        
        for filename, label in labels_dict.items():
            img_path = Path(image_dir) / filename
            
            if not img_path.exists():
                continue
                
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    def load_labels_from_csv(self, csv_path, filename_col='filename', label_col='label'):
        """Load labels from CSV file"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        labels_dict = dict(zip(df[filename_col], df[label_col]))
        print(f"Loaded {len(labels_dict)} labels from CSV")
        return labels_dict
    
    def validate_images(self, image_dir, labels_dict):
        """Check for corrupted images and return cleaned labels_dict"""
        import warnings
        from PIL import Image
        print("Validating images...")
        valid_labels = {}
        corrupted = []
        
        for i, (filename, label) in enumerate(labels_dict.items()):
            if i % 5000 == 0:
                print(f"Validated {i}/{len(labels_dict)} images...")
            
            img_path = Path(image_dir) / filename
            if not img_path.exists():
                corrupted.append(filename)
                continue
            
            try:
                # Use PIL for stricter validation
                with Image.open(img_path) as img:
                    img.verify()  # Verify it's actually a valid image
                
                # Re-open for actual loading (verify() closes it)
                with Image.open(img_path) as img:
                    img.load()  # Force load to catch truncated images
                    
                    # Check minimum size
                    if img.size[0] < 10 or img.size[1] < 10:
                        corrupted.append(filename)
                        continue
                
                valid_labels[filename] = label
                
            except Exception as e:
                corrupted.append(filename)
        
        print(f"\nValidation complete!")
        print(f"Valid images: {len(valid_labels)}")
        print(f"Corrupted/missing images: {len(corrupted)}")
        if corrupted[:10]:  # Show first 10
            print(f"Examples of bad files: {corrupted[:10]}")
        
        return valid_labels
    
    def create_data_generators(self, image_dir, labels_dict, batch_size=32, validation_split=0.2):
        """Create efficient data generators for large datasets"""
        # Get all image paths and labels
        image_paths = []
        labels = []
        
        for filename, label in labels_dict.items():
            img_path = Path(image_dir) / filename
            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(label)
        
        # Split into train and validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=validation_split, random_state=42
        )
        
        # Create TensorFlow datasets
        train_dataset = self._create_dataset(train_paths, train_labels, batch_size, shuffle=True)
        val_dataset = self._create_dataset(val_paths, val_labels, batch_size, shuffle=False)
        
        return train_dataset, val_dataset, len(train_paths), len(val_paths)
    
    def _create_dataset(self, image_paths, labels, batch_size, shuffle):
        """Create TensorFlow dataset from paths"""
        def load_image(path, label):
            try:
                img = tf.io.read_file(path)
                # Try to decode as JPEG first, then PNG if that fails
                try:
                    img = tf.image.decode_jpeg(img, channels=3)
                except:
                    img = tf.image.decode_png(img, channels=3)
                img = tf.image.resize(img, self.img_size)
                img = tf.cast(img, tf.float32) / 255.0
                return img, label
            except:
                # Return a black image if loading fails
                return tf.zeros((*self.img_size, 3), dtype=tf.float32), label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(self, train_dataset, val_dataset, epochs=10):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
            keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
        
        # Let TensorFlow automatically determine steps
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def predict_image(self, image_path):
        """Predict label for a single image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.resize takes (width, height), so reverse img_size
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = self.model.predict(img, verbose=0)[0][0]
        label = 1 if prediction > 0.5 else 0
        confidence = prediction if label == 1 else 1 - prediction
        
        return label, confidence
    
    def predict_batch(self, image_paths):
        """Predict labels for multiple images efficiently"""
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.resize takes (width, height), so reverse img_size
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
                img = img.astype('float32') / 255.0
                images.append(img)
                valid_paths.append(path)
            except:
                continue
        
        images = np.array(images)
        predictions = self.model.predict(images, verbose=0)
        labels = (predictions > 0.5).astype(int).flatten()
        confidences = np.where(labels == 1, predictions, 1 - predictions).flatten()
        
        return list(zip(valid_paths, labels, confidences))
    
    def save_model(self, path='image_classifier.keras'):
        """Save trained model"""
        import json
        self.model.save(path)
        # Save image size config
        config_path = path.replace('.keras', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({'img_size': self.img_size}, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path='image_classifier.keras'):
        """Load trained model"""
        import json
        self.model = keras.models.load_model(path)
        # Load image size config
        config_path = path.replace('.keras', '_config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.img_size = tuple(config['img_size'])
                print(f"Model loaded from {path} with img_size={self.img_size}")
        except FileNotFoundError:
            print(f"Model loaded from {path} (using default img_size={self.img_size})")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = ImageClassifier(img_size=(224, 224))
    
    # Load labels from CSV
    image_dir = path
    print(image_dir)
    csv_path = dataset
    
    print("Loading labels from CSV...")
    labels_dict = classifier.load_labels_from_csv(
        csv_path, 
        filename_col='path',  # Change to your CSV column name
        label_col='label'          # Change to your CSV column name
    )
    labels_dict = classifier.validate_images(image_dir, labels_dict)  # This will take a few minutes
    print("Creating data generators...")
    train_ds, val_ds, n_train, n_val = classifier.create_data_generators(
        image_dir, labels_dict, batch_size=32, validation_split=0.2
    )
    
    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    
    # Build and train model
    print("Building model...")
    classifier.build_model()
    
    print("Training model...")
    history = classifier.train(
        train_ds, val_ds,
        epochs=10
    )
    
    # Save model
    classifier.save_model('detector.keras')
    
    # Later, to use the model for predictions:
    # classifier = ImageClassifier()
    # classifier.load_model('my_image_classifier.keras')
    # label, confidence = classifier.predict_image('new_image.jpg')
    # print(f"Predicted: {label} (confidence: {confidence:.2%})")
