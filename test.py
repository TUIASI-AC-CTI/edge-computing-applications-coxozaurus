import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from data_collector.hand_tracker import HandTracker

path = os.getcwd()

PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
ANCHORS = path + "/models/anchors.csv"

GESTURE_MODEL = path + "/trained_models/gesture_model_quant.tflite"
LABELS_FILE = path + "/trained_models/gesture_labels.txt"

UNKNOWN_CONFIDENCE_THRESHOLD = 0.9
UNKNOWN_ENTROPY_THRESHOLD = 0.5


class GestureClassifier:
    
    def __init__(self, model_path, labels_path=None, 
                 unknown_threshold=UNKNOWN_CONFIDENCE_THRESHOLD, 
                 entropy_threshold=UNKNOWN_ENTROPY_THRESHOLD):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Thresholds for unknown detection
        self.unknown_threshold = unknown_threshold
        self.entropy_threshold = entropy_threshold
        
        # Load labels
        self.labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        print(f"Loaded gesture classifier with {len(self.labels)} classes")
        print(f"Unknown detection: confidence < {unknown_threshold:.2f} or entropy > {entropy_threshold:.2f}")
        
    def preprocess_landmarks(self, landmarks):
        """Preprocess landmarks for model input"""
        # Normalize relative to wrist (landmark 0)
        normalized = landmarks - landmarks[0]
        features = normalized.flatten()
        
        # Scale by max absolute value
        max_val = np.max(np.abs(features))
        if max_val > 0:
            features = features / max_val
        
        return features.astype(np.float32)
    
    def __call__(self, landmarks):
        """Classify gesture from landmarks with unknown detection"""

        features = self.preprocess_landmarks(landmarks)
        input_data = np.expand_dims(features, axis=0)
        
        # Handle quantized input
        if self.input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.uint8)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle quantized output
        if self.output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predictions = output_data[0]
        
        # Calculate confidence metrics
        max_confidence = np.max(predictions)
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        
        is_unknown = (max_confidence < self.unknown_threshold or 
                     entropy > self.entropy_threshold)
        
        return {
            'predictions': predictions,
            'max_confidence': max_confidence,
            'entropy': entropy,
            'is_unknown': is_unknown,
            'predicted_class': np.argmax(predictions)
        }


class GestureRecognitionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition")
        self.root.geometry("1000x800")
        
        self.cap = None
        self.tracker = None
        self.classifier = None
        self.current_frame = None
        self.is_running = False
        
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_models()
        
    def _detect_cameras(self):
        """Detect available cameras"""
        available = []
        for i in range(10):  # Check first 10 indices
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
        
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=1, column=0, pady=5)
        
        camera_frame = ttk.LabelFrame(control_frame, text="Camera", padding="10")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=5)
        
        ttk.Label(camera_frame, text="Select:").grid(row=0, column=0, sticky=tk.W, pady=2)
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_index, 
                                     values=self.available_cameras, state="readonly", width=8)
        camera_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self._change_camera)
        
        prediction_frame = ttk.LabelFrame(control_frame, text="Predicted Gesture", padding="10")
        prediction_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.prediction_label = ttk.Label(prediction_frame, text="No detection", 
                                          justify=tk.CENTER, font=('Arial', 16, 'bold'), width=20)
        self.prediction_label.pack(expand=True, fill=tk.BOTH)
        
    def _initialize_models(self):
        """Initialize hand tracker and gesture classifier"""
        try:
            self.tracker = HandTracker(
                palm_detection_model=PALM_MODEL,
                hand_landmark_model=LANDMARK_MODEL,
                anchors=ANCHORS,
                num_hands=1
            )
            
            self.classifier = GestureClassifier(
                GESTURE_MODEL,
                LABELS_FILE,
                unknown_threshold=UNKNOWN_CONFIDENCE_THRESHOLD,
                entropy_threshold=UNKNOWN_ENTROPY_THRESHOLD
            )
            
            self._open_camera()
            self.is_running = True
            self._update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize models:\n{str(e)}\n\nPlease check file paths in script.")
            self.root.quit()
    
    def _open_camera(self):
        """Open the camera"""
        if self.cap is not None:
            self.cap.release()
        
        camera_idx = self.camera_index.get()
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_idx}")
        
        # Try to set higher resolution (1920x1080 or 1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Read actual resolution achieved
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If Full HD not supported, try HD
        if actual_width < 1920:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print(f"Camera resolution: {actual_width}x{actual_height}")
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
    
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
        
        for connection in connections:
            x0, y0 = landmarks[connection[0]]
            x1, y1 = landmarks[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        for i, (x, y) in enumerate(landmarks):
            color = (0, 255, 0) if i in [0, 1, 5, 9, 13, 17] else (0, 200, 255)
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
    
    def _update_prediction_display(self, result):
        """Update the prediction label at the bottom"""
        is_unknown = result['is_unknown']
        max_confidence = result['max_confidence']
        predicted_class = result['predicted_class']
        
        if is_unknown:
            self.prediction_label.config(
                text="UNKNOWN GESTURE", 
                foreground="red"
            )
        else:
            label = self.classifier.labels[predicted_class] if predicted_class < len(self.classifier.labels) else f"Gesture {predicted_class}"
            self.prediction_label.config(
                text=f"{label} ({max_confidence:.1%})", 
                foreground="green"
            )
    
    def _update_frame(self):
        """Update video frame with recognition"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            # Detect hands and landmarks
            detections = self.tracker(frame)
            
            if detections:
                for landmarks, hand_bbox, handedness in detections:
                    self._draw_landmarks(landmarks, frame)
                    result = self.classifier(landmarks)
                    self._update_prediction_display(result)
                    
                    # Display hand type on frame
                    hand_type = "Right" if handedness > 0.5 else "Left"
                    cv2.putText(frame, f"Hand: {hand_type}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.prediction_label.config(text="No detection", foreground="black")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            parent_width = self.video_label.master.winfo_width()
            parent_height = self.video_label.master.winfo_height()
            
            if parent_width > 1 and parent_height > 1:
                # Calculate scale to fit within parent while maintaining aspect ratio
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
    
    # Handle window close
    def on_closing():
        app._cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()