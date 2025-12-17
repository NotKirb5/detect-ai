import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




class ImageClassifier:
    def __init__(self, img_size=(64, 64)):
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
    
    def create_data_generators(self, image_dir, labels_dict, batch_size=8, validation_split=0.2):
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
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label
        
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[0],self.img_size[1]))
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
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
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='image_classifier.keras'):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
        config_path = path.replace('.keras', '_config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.img_size = tuple(config['img_size'])
                print(f"Model loaded from {path} with img_size={self.img_size}")
        except FileNotFoundError:
            print(f"Model loaded from {path} (using default img_size={self.img_size})")


