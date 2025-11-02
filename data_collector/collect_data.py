import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from hand_tracker import HandTracker
from dataset_manager import DatasetManager


path = os.getcwd()

class GestureDataCollector:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Collector")
        self.root.geometry("1000x800")
        
        self.PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
        self.LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
        self.ANCHORS = path + "/models/anchors.csv"
        
        self.cap = None
        self.tracker = None
        self.dataset_manager = DatasetManager()
        self.current_frame = None
        self.is_running = False
        
        self.current_gesture_id = tk.StringVar(value="0")
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_tracker()
        
    def _detect_cameras(self):
        """Detect available cameras on Windows"""
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
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)
        
        input_frame = ttk.LabelFrame(control_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Label(input_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, pady=2)
        camera_combo = ttk.Combobox(input_frame, textvariable=self.camera_index, 
                                     values=self.available_cameras, state="readonly", width=8)
        camera_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        camera_combo.bind('<<ComboboxSelected>>', self._change_camera)
        
        ttk.Label(input_frame, text="Gesture ID:").grid(row=1, column=0, sticky=tk.W, pady=2)
        gesture_entry = ttk.Entry(input_frame, textvariable=self.current_gesture_id, width=8)
        gesture_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        
        input_frame.columnconfigure(1, weight=1)
        
        button_frame = ttk.LabelFrame(control_frame, text="Actions", padding="10")
        button_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Button(button_frame, text="Capture Landmarks", 
                  command=self._capture_landmarks).grid(row=0, column=0, 
                                                        sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(button_frame, text="Save Dataset & Exit", 
                  command=self._save_and_exit).grid(row=1, column=0, 
                                                    sticky=(tk.W, tk.E), pady=2)
        
        button_frame.columnconfigure(0, weight=1)
        
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=2, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.stats_label = ttk.Label(stats_frame, text="Total samples: 0\nGesture classes: 0", 
                                     justify=tk.LEFT, anchor=tk.W)
        self.stats_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        stats_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
    def _initialize_tracker(self):
        """Initialize the hand tracker"""
        try:
            self.tracker = HandTracker(
                palm_detection_model=self.PALM_MODEL,
                hand_landmark_model=self.LANDMARK_MODEL,
                anchors=self.ANCHORS,
                num_hands=1
            )
            self._open_camera()
            self.is_running = True
            self._update_frame()
            self.status_var.set("Tracker initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize tracker: {str(e)}")
            self.root.quit()
    
    def _open_camera(self):
        """Open the camera with high resolution"""
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
        
        self.status_var.set(f"Camera resolution: {actual_width}x{actual_height}")
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
        self.status_var.set(f"Switched to camera {self.camera_index.get()}")
    
    def _update_frame(self):
        """Update video frame"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            # Process frame with hand tracker
            detections = self.tracker(frame)
            
            if detections:
                for landmarks, hand_bbox, handedness in detections:
                    self._draw_landmarks(landmarks, frame)
                    self._draw_bbox(hand_bbox, frame)
                    hand_type = "Right" if handedness > 0.5 else "Left"
                    cv2.putText(frame, f"Hand: {hand_type}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to PhotoImage and display - scale to fit while maintaining aspect ratio
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
            x0, y0 = landmarks[connection[0]]
            x1, y1 = landmarks[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        # Draw points
        for i, (x, y) in enumerate(landmarks):
            color = (0, 255, 0) if i in [0, 1, 5, 9, 13, 17] else (0, 200, 255)
            cv2.circle(frame, (int(x), int(y)), 8, color, -1)
    
    def _draw_bbox(self, hand_bbox, frame):
        """Draw bounding box"""
        center = hand_bbox.center
        dims = hand_bbox.dims
        rotation = hand_bbox.rotation * 180 / np.pi
        
        rect = (center, dims, rotation)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    
    def _capture_landmarks(self):
        """Capture landmarks from current frame"""
        try:
            gesture_id = int(self.current_gesture_id.get())
            if gesture_id < 0:
                raise ValueError("Gesture ID must be non-negative")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid gesture ID: {str(e)}")
            return
        
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return
        
        # Process current frame
        detections = self.tracker(self.current_frame)
        
        if not detections:
            messagebox.showwarning("Warning", "No hand detected in frame")
            return
        
        # Take first detection
        landmarks, _, handedness = detections[0]
        
        # Normalize landmarks (relative to wrist)
        normalized_landmarks = landmarks - landmarks[0]
        normalized_landmarks = normalized_landmarks.flatten()
        max_val = np.max(np.abs(normalized_landmarks))
        if max_val > 0:
            normalized_landmarks = normalized_landmarks / max_val
        
        self.dataset_manager.add_sample(normalized_landmarks, gesture_id)
        self._update_statistics()
        
        self.status_var.set(f"Captured gesture {gesture_id}. Total samples: {len(self.dataset_manager.data)}")
    
    def _update_statistics(self):
        """Update statistics display"""
        stats = self.dataset_manager.get_statistics()
        stats_text = f"Total samples: {stats['total_samples']}\n"
        stats_text += f"Gesture classes: {stats['num_classes']}\n\n"
        stats_text += "Samples per class:\n"
        for class_id, count in sorted(stats['samples_per_class'].items()):
            stats_text += f"  Gesture {class_id}: {count}\n"
        
        self.stats_label.config(text=stats_text)
    
    def _save_and_exit(self):
        """Save dataset and exit application"""
        if len(self.dataset_manager.data) == 0:
            result = messagebox.askyesno("Warning", 
                                         "No data collected. Exit anyway?")
            if result:
                self._cleanup()
                self.root.quit()
            return
        
        filename = f"gesture_dataset.h5"
        
        try:
            self.dataset_manager.save_to_h5(filename)
            messagebox.showinfo("Success", 
                               f"Dataset saved to {filename}\n"
                               f"Total samples: {len(self.dataset_manager.data)}")
            self._cleanup()
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
    
    def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureDataCollector(root)
    
    # Handle window close
    def on_closing():
        app._cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()