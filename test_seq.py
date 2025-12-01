import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from time_series.hand_tracker import HandTracker
import time

path = os.getcwd()

PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
ANCHORS = path + "/models/anchors.csv"

GESTURE_MODEL = path + "/trained_models/gesture_conv1d_quant.tflite"
LABELS_FILE = path + "/trained_models/gesture_labels.txt"

SEQUENCE_LENGTH = 10
UNKNOWN_CONFIDENCE_THRESHOLD = 0.95
UNKNOWN_ENTROPY_THRESHOLD = 0.3

MOVEMENT_THRESHOLD = 1.0

RESET_COOLDOWN_DURATION = 4.0

GESTURE_NAMES = {
    0: "Swipe Up",
    1: "Swipe Down",
    2: "Swipe Left",
    3: "Swipe Right",
    4: "Zoom In",
    5: "Zoom Out",
    6: "Rotate Clockwise",
    7: "Rotate Counter Clockwise"
}


class SequenceBuffer:
    """Sequential buffer for sequence collection"""
    
    def __init__(self, sequence_length, feature_size=63, movement_threshold=MOVEMENT_THRESHOLD):
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
        """Normalize landmarks (same as data collection)"""
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


class Gesture1D_ConvClassifier:
    
    def __init__(self, model_path, labels_path=None, 
                 unknown_confidence_threshold=UNKNOWN_CONFIDENCE_THRESHOLD,
                 unknown_entropy_threshold=UNKNOWN_ENTROPY_THRESHOLD):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.unknown_confidence_threshold = unknown_confidence_threshold
        self.unknown_entropy_threshold = unknown_entropy_threshold
        
        # Get sequence length from model input shape
        self.sequence_length = self.input_details[0]['shape'][1]
        self.feature_size = self.input_details[0]['shape'][2]
        
        self.labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        print(f"Loaded 1D_Conv gesture classifier:")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Feature size: {self.feature_size}")
        print(f"  Classes: {len(self.labels)}")
        print(f"  Unknown confidence threshold: {unknown_confidence_threshold:.2f}")
        print(f"  Unknown entropy threshold: {unknown_entropy_threshold:.2f}")
        print(f"  Movement threshold: {MOVEMENT_THRESHOLD:.3f}")
        
    def __call__(self, sequence):
        """Classify gesture from sequence of landmarks"""
        if len(sequence.shape) == 2:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        else:
            input_data = sequence.astype(np.float32)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predictions = output_data[0]
        
        # Calculate confidence metrics
        max_confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Calculate entropy for uncertainty
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        normalized_entropy = entropy / np.log(len(predictions))
        
        is_unknown = (max_confidence < self.unknown_confidence_threshold) or \
                     (normalized_entropy > self.unknown_entropy_threshold)
        
        return {
            'predictions': predictions,
            'max_confidence': max_confidence,
            'entropy': normalized_entropy,
            'is_unknown': is_unknown,
            'predicted_class': predicted_class
        }


class GestureRecognitionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("1D_Conv Gesture Recognition")
        self.root.geometry("800x700")
        
        self.cap = None
        self.tracker = None
        self.classifier = None
        self.sequence_buffer = None
        self.current_frame = None
        self.is_running = False
        
        # Track if we're in cooldown after prediction
        self.in_cooldown = False
        self.cooldown_frames = 0
        self.cooldown_duration = 5
        
        self.last_prediction_result = None
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_models()
        
    def _detect_cameras(self):
        """Detect available cameras"""
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available if available else [0]
    
    def _setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        center_frame = ttk.Frame(video_frame)
        center_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(center_frame, background='black')
        self.video_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Control panel
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=1, column=0, pady=5)
        
        # Camera selection
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Selection", padding="10")
        camera_frame.pack()
        
        ttk.Label(camera_frame, text="Select Camera:").grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_index, 
                                     values=self.available_cameras, state="readonly", width=10)
        camera_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self._change_camera)
        
    def _initialize_models(self):
        """Initialize hand tracker and gesture classifier"""
        try:
            self.tracker = HandTracker(
                palm_detection_model=PALM_MODEL,
                hand_landmark_model=LANDMARK_MODEL,
                anchors=ANCHORS,
                num_hands=1
            )
            
            self.classifier = Gesture1D_ConvClassifier(
                GESTURE_MODEL,
                LABELS_FILE,
                unknown_confidence_threshold=UNKNOWN_CONFIDENCE_THRESHOLD,
                unknown_entropy_threshold=UNKNOWN_ENTROPY_THRESHOLD
            )
            
            self.sequence_buffer = SequenceBuffer(
                sequence_length=self.classifier.sequence_length,
                feature_size=self.classifier.feature_size,
                movement_threshold=MOVEMENT_THRESHOLD
            )
            
            self._open_camera()
            self.is_running = True
            self._update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize models:\n{str(e)}\n\nCheck file paths.")
            self.root.quit()
    
    def _open_camera(self):
        """Open the camera"""
        if self.cap is not None:
            self.cap.release()
        
        camera_idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")
        
        # Set resolution (try high res first)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Read actual resolution achieved
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If Full HD not supported, try HD
        if actual_width < 1920:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
        self.sequence_buffer.reset("Camera changed")
        self.in_cooldown = False
        self.cooldown_frames = 0
    
    def _draw_landmarks(self, landmarks, frame):
        """Draw hand landmarks on frame"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        for connection in connections:
            x0, y0 = landmarks[connection[0]][:2]
            x1, y1 = landmarks[connection[1]][:2]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        # Draw points
        for i, point in enumerate(landmarks):
            x, y = point[:2]
            color = (0, 255, 0) if i in [0, 1, 5, 9, 13, 17] else (0, 200, 255)
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
    
    def _draw_overlay(self, frame, status, prediction_result=None, has_movement=True, last_prediction=None):
        """Draw compact overlay information at bottom of frame"""
        h, w = frame.shape[:2]
        overlay_height = 60
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - overlay_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        bar_width = 200
        bar_height = 15
        bar_x = 350
        bar_y = h - overlay_height + 15
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        if status['in_reset_cooldown']:
            progress = 1.0 - (status['reset_cooldown_remaining'] / RESET_COOLDOWN_DURATION)
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 165, 255), -1)
        else:
            progress = status['frame_count'] / self.classifier.sequence_length
            fill_width = int(bar_width * progress)
            
            if status['is_ready'] and status['has_significant_movement']:
                color = (0, 255, 0)  # Green
            elif status['is_ready']:
                color = (0, 165, 255)  # Orange
            else:
                color = (100, 100, 100)  # Gray
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        
        buffer_text_y = bar_y + bar_height + 18
        buffer_text = f"{status['frame_count']}/{self.classifier.sequence_length}"
        cv2.putText(frame, buffer_text, (bar_x, buffer_text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        text_x = bar_x + bar_width + 40
        text_y_line1 = h - overlay_height + 25
        text_y_line2 = h - overlay_height + 45
        
        if status['in_reset_cooldown']:
            if last_prediction is not None:
                if not last_prediction['has_movement']:
                    cv2.putText(frame, "Last: NO MOVEMENT DETECTED", (text_x, text_y_line1), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                elif last_prediction['is_unknown']:
                    result_text = f"Last: UNKNOWN (C:{last_prediction['max_confidence']:.0%} E:{last_prediction['entropy']:.2f})"
                    cv2.putText(frame, result_text, (text_x, text_y_line1), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    label = GESTURE_NAMES.get(last_prediction['predicted_class'], 
                                             f"Class {last_prediction['predicted_class']}")
                    result_text = f"Last: {label} (C:{last_prediction['max_confidence']:.0%} E:{last_prediction['entropy']:.2f})"
                    cv2.putText(frame, result_text, (text_x, text_y_line1), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
            cooldown_text = f"COOLDOWN: {status['reset_cooldown_remaining']:.1f}s - {status['reset_reason']}"
            cv2.putText(frame, cooldown_text, (text_x, text_y_line2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif prediction_result is not None and status['is_ready']:
            if not has_movement:
                cv2.putText(frame, "NO MOVEMENT DETECTED", (text_x, text_y_line1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif prediction_result['is_unknown']:
                result_text = f"UNKNOWN GESTURE (C:{prediction_result['max_confidence']:.0%} E:{prediction_result['entropy']:.2f})"
                cv2.putText(frame, result_text, (text_x, text_y_line1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                label = GESTURE_NAMES.get(prediction_result['predicted_class'], 
                                         f"Class {prediction_result['predicted_class']}")
                conf_text = f"{label} (C:{prediction_result['max_confidence']:.0%} E:{prediction_result['entropy']:.2f})"
                cv2.putText(frame, conf_text, (text_x, text_y_line1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif status['consecutive_nulls'] > 0:
            warning_text = f"Missing detection ({status['consecutive_nulls']}/{self.sequence_buffer.max_consecutive_nulls})"
            cv2.putText(frame, warning_text, (text_x, text_y_line1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        else:
            collecting_text = "Collecting frames..."
            cv2.putText(frame, collecting_text, (text_x, text_y_line1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    def _update_frame(self):
        """Update video frame with recognition"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            current_prediction = None
            has_movement = True
            
            if self.in_cooldown:
                self.cooldown_frames -= 1
                if self.cooldown_frames <= 0:
                    self.sequence_buffer.reset("Prediction completed")
                    self.in_cooldown = False
            
            status = self.sequence_buffer.get_status()
            
            if not self.in_cooldown and not status['in_reset_cooldown']:
                detections = self.tracker(frame)
                
                if detections:
                    for landmarks, hand_bbox, handedness in detections:
                        self._draw_landmarks(landmarks, frame)
                        self.sequence_buffer.add_frame(landmarks)
                else:
                    self.sequence_buffer.add_null_frame()
                
                status = self.sequence_buffer.get_status()
                
                if status['is_ready'] and not self.in_cooldown and not status['in_reset_cooldown']:
                    has_movement = status['has_significant_movement']
                    sequence = self.sequence_buffer.get_sequence()
                    
                    if sequence is not None:
                        current_prediction = self.classifier(sequence)
                        
                        self.last_prediction_result = {
                            'predicted_class': current_prediction['predicted_class'],
                            'max_confidence': current_prediction['max_confidence'],
                            'entropy': current_prediction['entropy'],
                            'is_unknown': current_prediction['is_unknown'],
                            'has_movement': has_movement
                        }
                        
                        self.in_cooldown = True
                        self.cooldown_frames = self.cooldown_duration
            
            self._draw_overlay(frame, status, current_prediction, has_movement, self.last_prediction_result)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            parent_width = self.video_label.master.winfo_width()
            parent_height = self.video_label.master.winfo_height()
            
            if parent_width > 1 and parent_height > 1:
                img_ratio = img.width / img.height
                parent_ratio = parent_width / parent_height
                
                if img_ratio > parent_ratio:
                    new_width = parent_width - 20
                    new_height = int(new_width / img_ratio)
                else:
                    new_height = parent_height - 20
                    new_width = int(new_height * img_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self._update_frame)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionApp(root)
    
    def on_closing():
        app._cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()