# Copyright 2025 NXP
# SPDX-License-Identifier: Apache-2.0
"""
gesture_classifier.py

This module provides the Classifier class, which uses 21 2D landmarks
to predict hand gestures with unknown detection.

Usage:
    import gesture_classifier

    classifier = gesture_classifier.Classifier()
    result = classifier(landmarks)

Classes:
    Classifier: Predict hand gestures with confidence and uncertainty estimation.
"""
import numpy as np
import tflite_runtime.interpreter as tflite


class Classifier:
    """Gesture classifier with unknown detection using thresholds"""
    
    # Unknown detection thresholds
    UNKNOWN_CONFIDENCE_THRESHOLD = 0.9
    UNKNOWN_ENTROPY_THRESHOLD = 0.5
    
    def __init__(self, model):
        self._interpreter = tflite.Interpreter(model)
        self._interpreter.allocate_tensors()

        _out_details = self._interpreter.get_output_details()
        _in_details = self._interpreter.get_input_details()

        self._in_idx = _in_details[0]["index"]
        self._out_idx = _out_details[0]["index"]

    def __call__(self, landmarks):
        """
        Classify gesture from landmarks with unknown detection
        
        Args:
            landmarks: (21, 2) array of hand landmarks
            
        Returns:
            dict with:
                - predictions: probability array
                - max_confidence: highest probability
                - entropy: uncertainty measure
                - is_unknown: boolean flag
                - predicted_class: index of predicted class
        """
        # Normalize landmarks
        landmarks = landmarks - landmarks[0]
        landmarks = landmarks.flatten()
        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val

        # Prepare input data
        input_data = landmarks.astype("float32")[None]
        
        # Handle quantized input (INT8 model)
        input_details = self._interpreter.get_input_details()[0]
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)

        # Run inference
        self._interpreter.set_tensor(self._in_idx, input_data)
        self._interpreter.invoke()
        output_data = self._interpreter.get_tensor(self._out_idx)
        
        # Handle quantized output (INT8 model)
        output_details = self._interpreter.get_output_details()[0]
        if output_details['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predictions = output_data.squeeze()

        # Calculate confidence metrics
        max_confidence = np.max(predictions)
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        
        # Determine if unknown based on thresholds
        is_unknown = (max_confidence < self.UNKNOWN_CONFIDENCE_THRESHOLD or 
                     entropy > self.UNKNOWN_ENTROPY_THRESHOLD)
        
        return {
            'predictions': predictions,
            'max_confidence': float(max_confidence),
            'entropy': float(entropy),
            'is_unknown': bool(is_unknown),
            'predicted_class': int(np.argmax(predictions))
        }