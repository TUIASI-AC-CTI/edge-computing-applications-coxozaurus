# Copyright 2025 NXP
# SPDX-License-Identifier: Apache-2.0
"""
gesture_classifier_seq.py

This module provides the Classifier class for sequential gesture recognition,
which uses sequences of 21 3D landmarks to predict dynamic hand gestures with 
unknown detection. It also includes the SequenceBuffer class for collecting
and managing landmark sequences.

Usage:
    import gesture_classifier_seq

    classifier = gesture_classifier_seq.Classifier()
    sequence_buffer = gesture_classifier_seq.SequenceBuffer(sequence_length=10, feature_size=63)
    result = classifier(sequence)

Classes:
    SequenceBuffer: Manages collection and normalization of landmark sequences.
    Classifier: Predict sequential hand gestures with confidence and uncertainty estimation.
"""
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# Hardcoded constant
RESET_COOLDOWN_DURATION = 4.0

# Unknown detection thresholds
UNKNOWN_CONFIDENCE_THRESHOLD = 0.95
UNKNOWN_ENTROPY_THRESHOLD = 0.3


class SequenceBuffer:
    """Sequential buffer for sequence collection"""
    
    def __init__(self, sequence_length, feature_size=63, movement_threshold=1.0):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.movement_threshold = movement_threshold
        self.buffer = []
        self.frame_count = 0
        self.is_ready = False
        
        self.sequence_reference_wrist = None
        self.sequence_reference_scale = None
        
        self.consecutive_nulls = 0
        self.max_consecutive_nulls = 2
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        
        self.last_reset_time = None
        self.reset_reason = None
        
    def reset(self, reason="Unknown"):
        """Reset buffer to initial state"""
        self.buffer = []
        self.frame_count = 0
        self.is_ready = False
        self.sequence_reference_wrist = None
        self.sequence_reference_scale = None
        self.consecutive_nulls = 0
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        self.last_reset_time = time.time()
        self.reset_reason = reason
    
    def is_in_reset_cooldown(self):
        """Check if currently in reset cooldown period"""
        if self.last_reset_time is None:
            return False
        elapsed = time.time() - self.last_reset_time
        return elapsed < RESET_COOLDOWN_DURATION
    
    def get_reset_cooldown_remaining(self):
        """Get remaining cooldown time in seconds"""
        if self.last_reset_time is None:
            return 0
        elapsed = time.time() - self.last_reset_time
        remaining = RESET_COOLDOWN_DURATION - elapsed
        return max(0, remaining)
    
    def add_frame(self, landmarks):
        """Add a frame with landmarks (21, 3)"""
        if self.is_in_reset_cooldown():
            return True
            
        if self.pending_interpolations and self.last_valid_landmarks is not None:
            current_normalized = self._normalize_landmarks(landmarks)
            num_missing = len(self.pending_interpolations)
            
            interpolated_frames = self._perform_linear_interpolation(
                self.last_valid_landmarks,
                current_normalized,
                num_missing
            )
            
            for idx, interp_frame in zip(self.pending_interpolations, interpolated_frames):
                self.buffer[idx] = interp_frame
            
            self.pending_interpolations = []
        
        self.consecutive_nulls = 0
        normalized = self._normalize_landmarks(landmarks)
        self.buffer.append(normalized)
        self.last_valid_landmarks = normalized
        
        self.frame_count = len(self.buffer)
        if self.frame_count >= self.sequence_length:
            self.is_ready = True
        
        return True
    
    def add_null_frame(self):
        """Handle missing detection"""
        if self.is_in_reset_cooldown():
            return True
            
        self.consecutive_nulls += 1
        
        if self.consecutive_nulls > self.max_consecutive_nulls:
            self.reset("Hand lost")
            return False
        
        if self.last_valid_landmarks is not None and len(self.buffer) > 0:
            placeholder = self.last_valid_landmarks.copy()
            self.buffer.append(placeholder)
            
            # Mark this frame index for interpolation
            self.pending_interpolations.append(len(self.buffer) - 1)
            
            self.frame_count = len(self.buffer)
            if self.frame_count >= self.sequence_length:
                self.is_ready = True
        
        return True
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks"""
        if len(self.buffer) == 0:
            self.sequence_reference_wrist = landmarks[0].copy()
            
            # Calculate scale based on hand span (wrist to middle fingertip)
            hand_span = np.linalg.norm(landmarks[12] - landmarks[0])
            
            # Fallback to max finger spread if hand_span is too small
            if hand_span < 0.01:
                hand_span = np.max(np.linalg.norm(landmarks - landmarks[0], axis=1))
            
            if hand_span == 0:
                hand_span = 1.0
                
            self.sequence_reference_scale = hand_span
        
        normalized = landmarks - self.sequence_reference_wrist
        normalized = normalized / self.sequence_reference_scale
        
        return normalized.flatten().astype(np.float32)
    
    def _perform_linear_interpolation(self, start_landmarks, end_landmarks, num_steps):
        """Perform linear interpolation between two landmark sets"""
        interpolated = []
        for i in range(1, num_steps + 1):
            t = i / (num_steps + 1)
            interpolated_frame = start_landmarks + t * (end_landmarks - start_landmarks)
            interpolated.append(interpolated_frame)
        
        return interpolated
    
    def get_sequence(self):
        """Get the current sequence"""
        if not self.is_ready:
            return None
        
        return np.array(self.buffer[:self.sequence_length], dtype=np.float32)
    
    def calculate_movement(self):
        """Calculate total movement in the sequence"""
        if len(self.buffer) < 2:
            return 0.0
        
        sequence = np.array(self.buffer[:self.sequence_length])
        
        # Calculate frame-to-frame differences
        diffs = np.diff(sequence, axis=0)
        
        # Calculate total movement (sum of L2 norms of differences)
        total_movement = np.sum(np.linalg.norm(diffs, axis=1))
        
        return total_movement
    
    def has_significant_movement(self):
        """Check if sequence has significant movement"""
        if not self.is_ready:
            return False
        
        movement = self.calculate_movement()
        return movement >= self.movement_threshold
    
    def get_status(self):
        """Get current buffer status"""
        movement = self.calculate_movement() if len(self.buffer) > 0 else 0.0
        
        return {
            'frame_count': self.frame_count,
            'is_ready': self.is_ready,
            'consecutive_nulls': self.consecutive_nulls,
            'has_pending_interp': len(self.pending_interpolations) > 0,
            'movement': movement,
            'has_significant_movement': movement >= self.movement_threshold if self.is_ready else False,
            'in_reset_cooldown': self.is_in_reset_cooldown(),
            'reset_cooldown_remaining': self.get_reset_cooldown_remaining(),
            'reset_reason': self.reset_reason
        }


class Classifier:
    """Sequential gesture classifier with unknown detection using thresholds"""
    
    def __init__(self, model, external_delegate=None):
        if external_delegate:
            external_delegate = [tflite.load_delegate(external_delegate)]
            
        self._interpreter = tflite.Interpreter(
            model, experimental_delegates=external_delegate
        )
        self._interpreter.allocate_tensors()

        _out_details = self._interpreter.get_output_details()
        _in_details = self._interpreter.get_input_details()

        self._in_idx = _in_details[0]["index"]
        self._out_idx = _out_details[0]["index"]
        
        # Get sequence length from model input shape
        self.sequence_length = _in_details[0]['shape'][1]
        self.feature_size = _in_details[0]['shape'][2]

        # Ignore the first invoke (Warm-up time)
        batch, seq_len, features = tuple(_in_details[0]["shape"].tolist())
        self._interpreter.set_tensor(
            self._in_idx, np.random.rand(batch, seq_len, features).astype("float32")
        )
        self._interpreter.invoke()

    def __call__(self, sequence):
        """
        Classify gesture from sequence of landmarks with unknown detection
        
        Args:
            sequence: (sequence_length, feature_size) array of normalized landmarks
            
        Returns:
            dict with:
                - predictions: probability array
                - max_confidence: highest probability
                - entropy: uncertainty measure
                - is_unknown: boolean flag
                - predicted_class: index of predicted class
        """
        # Prepare input data
        if len(sequence.shape) == 2:
            input_data = np.expand_dims(sequence, axis=0).astype("float32")
        else:
            input_data = sequence.astype("float32")

        # Run inference
        self._interpreter.set_tensor(self._in_idx, input_data)
        self._interpreter.invoke()
        output_data = self._interpreter.get_tensor(self._out_idx)
        
        predictions = output_data.squeeze()

        # Calculate confidence metrics
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Calculate entropy for uncertainty
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        normalized_entropy = entropy / np.log(len(predictions))
        
        # Determine if unknown based on thresholds
        is_unknown = (max_confidence < self.UNKNOWN_CONFIDENCE_THRESHOLD) or \
                     (normalized_entropy > self.UNKNOWN_ENTROPY_THRESHOLD)
        
        return {
            'predictions': predictions,
            'max_confidence': float(max_confidence),
            'entropy': float(normalized_entropy),
            'is_unknown': bool(is_unknown),
            'predicted_class': int(predicted_class)
        }