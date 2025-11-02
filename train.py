import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

path = os.getcwd()

class GestureModelTrainer:
    
    def __init__(self, dataset_path, output_dir='models'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.num_classes = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, val_split=0.15, test_split=0.15):
        """Load and split dataset"""
        print(f"Loading dataset from {self.dataset_path}")
        
        with h5py.File(self.dataset_path, 'r') as f:
            data = f['landmarks'][:]
            labels = f['labels'][:]
            
        print(f"Loaded {len(data)} samples")
        print(f"Unique classes: {np.unique(labels)}")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            data, labels_encoded, test_size=test_split, random_state=42, stratify=labels_encoded
        )
        
        val_size = val_split / (1 - test_split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        print(f"Train samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
        # Convert to one-hot encoding
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        
    def build_model(self, hidden_units=[128, 64, 32], dropout_rate=0.3):
        input_dim = self.X_train.shape[1]  # 42 features (21 landmarks * 2 coords, normalized)
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
        ])

        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        print("\nModel Architecture:")
        model.summary()
        
    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs...")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def convert_to_tflite(self, quantize=True):
        """Convert model to TFLite format with optimization"""
        print("\nConverting to TFLite format...")
        
        # Representative dataset for quantization
        def representative_dataset():
            for data in tf.data.Dataset.from_tensor_slices(self.X_train).batch(1).take(100):
                yield [tf.cast(data, tf.float32)]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            print("Applying INT8 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            output_path = os.path.join(self.output_dir, 'gesture_model_quant.tflite')
        else:
            tflite_model = converter.convert()
            output_path = os.path.join(self.output_dir, 'gesture_model.tflite')
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return output_path
    
    def verify_tflite_model(self, tflite_path, num_samples=10):
        """Verify TFLite model accuracy"""
        print(f"\nVerifying TFLite model: {tflite_path}")
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        num_samples = min(num_samples, len(self.X_test))
        correct = 0
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        for idx in indices:
            input_data = self.X_test[idx:idx+1].astype(np.float32)
            true_label = np.argmax(self.y_test[idx])
            
            # Handle quantized input
            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Handle quantized output
            if output_details[0]['dtype'] == np.uint8:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            pred_label = np.argmax(output_data[0])
            
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / num_samples
        print(f"TFLite Model Accuracy (on {num_samples} samples): {accuracy:.4f}")
        
    def save_class_labels(self):
        """Save class labels mapping"""
        labels_path = os.path.join(self.output_dir, 'gesture_labels.txt')
        with open(labels_path, 'w') as f:
            for label in self.label_encoder.classes_:
                f.write(f"{label}\n")
        print(f"Class labels saved to {labels_path}")


if __name__ == "__main__":
    dataset_path = path + "/gesture_dataset.h5"
    output_dir = path + "/trained_models"
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    hidden_units = [128, 64, 32]
    dropout = 0.3
    quantize = True
    
    trainer = GestureModelTrainer(dataset_path, output_dir)
    trainer.load_dataset()
    trainer.build_model(hidden_units=hidden_units, dropout_rate=dropout)
    
    history = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    trainer.evaluate()
    plot_path = os.path.join(output_dir, 'training_history.png')
    trainer.plot_training_history(history, save_path=plot_path)
    
    tflite_path = trainer.convert_to_tflite(quantize=quantize)
    trainer.verify_tflite_model(tflite_path, num_samples=50)
    trainer.save_class_labels()