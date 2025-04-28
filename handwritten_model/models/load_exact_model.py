
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

# Define TrueDivide layer for loading
class TrueDivide(Layer):
    def __init__(self, scalar=1.0, **kwargs):
        self.scalar = float(scalar)
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return inputs / self.scalar
    
    def get_config(self):
        config = super().get_config()
        config.update({'scalar': self.scalar})
        return config

# Define binary focal loss for loading
def binary_focal_loss(gamma=2.0, alpha=0.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_loss = alpha * tf.math.pow(1 - tf.where(y_true > 0.5, y_pred, 1 - y_pred), gamma) * cross_entropy
        return tf.reduce_mean(focal_loss)
    return binary_focal_loss_fixed

def load_model():
    """Load the model from Keras format"""
    model = keras.models.load_model(
        'exactly_matched_model.keras',
        custom_objects={
            'TrueDivide': TrueDivide,
            'binary_focal_loss_fixed': binary_focal_loss(gamma=2.0, alpha=0.75)
        }
    )
    return model

def recreate_and_load_weights():
    """Recreate the model and load weights"""
    # Define the model architecture
    inputs = Input(shape=(224, 224, 3), name='input_1')
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
        name='mobilenetv2_1.00_224'
    )
    base_model.trainable = False
    
    x = base_model(inputs)
    
    # Based on the extracted architecture
    x = GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    
    # First dense layer
    x = Dense(128, activation='relu', name='dense')(x)
    x = Dropout(0.5, name='dropout')(x)
    
    # Second dense layer
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    
    # Final dense layer
    x = Dense(1, use_bias=True, name='dense_2')(x)
    
    # Temperature scaling with TrueDivide
    x = TrueDivide(scalar=0.5)(x)
    
    # Activation
    outputs = Activation('sigmoid', name='activation')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='model')
    
    # Load weights
    model.load_weights('exactly_matched_model.weights.h5')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=binary_focal_loss(gamma=2.0, alpha=0.75),
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

# Example usage
if __name__ == "__main__":
    # Try loading from Keras format first
    try:
        model = load_model()
        print("Successfully loaded model from Keras format")
    except Exception as e:
        print(f"Error loading from Keras format: {e}")
        
        # Fall back to recreating and loading weights
        try:
            model = recreate_and_load_weights()
            print("Successfully loaded model by recreating and loading weights")
        except Exception as e2:
            print(f"Error loading with weights: {e2}")
            raise Exception("Could not load the model using any method")
    
    # Test the model
    import numpy as np
    dummy_input = np.zeros((1, 224, 224, 3))
    prediction = model.predict(dummy_input, verbose=0)
    print(f"Model prediction shape: {prediction.shape}")
    print(f"Sample prediction value: {prediction[0][0]}")
    
    # Test with a ones input
    ones_input = np.ones((1, 224, 224, 3))
    ones_prediction = model.predict(ones_input, verbose=0)
    print(f"Prediction with ones input: {ones_prediction[0][0]}")
    
    # Test with a random input
    random_input = np.random.random((1, 224, 224, 3))
    random_prediction = model.predict(random_input, verbose=0)
    print(f"Prediction with random input: {random_prediction[0][0]}")
    
    return model
