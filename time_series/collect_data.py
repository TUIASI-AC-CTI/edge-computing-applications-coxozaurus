import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from hand_tracker import HandTracker
from dataset_manager import DatasetManager
from dataset_selector import show_dataset_selector


path = os.getcwd()

class GestureDataCollector:
    
    def __init__(self, root, config):
        self.root = root
        self.root.title("Gesture Dataset Collector - Sequence Mode")
        self.root.geometry("1000x800")
        
        self.PALM_MODEL = path + "/models/palm_detection_full_quant.tflite"
        self.LANDMARK_MODEL = path + "/models/hand_landmark_full_quant.tflite"
        self.ANCHORS = path + "/models/anchors.csv"
        
        self.cap = None
        self.tracker = None
        
        self.config = config
        self.sequence_length = config['sequence_length']
        self.dataset_path = config.get('dataset_path', 'gesture_dataset.h5')
        
        self.dataset_manager = DatasetManager(sequence_length=self.sequence_length)
        
        if config['mode'] == 'load' and config['dataset_path']:
            try:
                self.dataset_manager.load_from_h5(config['dataset_path'])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
        
        self.current_frame = None
        self.is_running = False
        
        # Sequence recording state
        self.is_recording = False
        self.current_sequence = []
        self.frames_recorded = 0
        self.sequence_reference_wrist = None
        self.sequence_reference_scale = None
        
        # Tracking for consecutive null detections and interpolation
        self.consecutive_nulls = 0
        self.max_consecutive_nulls = 2  # Allow skipping 2 frames
        self.last_valid_landmarks = None  # Store last valid landmarks
        self.pending_interpolations = []  # Store frame indices that need interpolation
        
        self.current_gesture_id = tk.StringVar(value=str(config['selected_class']))
        self.camera_index = tk.IntVar(value=0)
        self.available_cameras = self._detect_cameras()
        
        self._setup_ui()
        self._initialize_tracker()
        self.root.bind('<space>', self._toggle_recording)
        
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
        
        # Recording frame
        record_frame = ttk.LabelFrame(control_frame, text="Recording", padding="10")
        record_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.record_button = ttk.Button(record_frame, text="Start Recording (Space)", 
                  command=self._toggle_recording)
        self.record_button.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(record_frame, variable=self.progress_var,
                                           maximum=self.sequence_length, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.frames_label = ttk.Label(record_frame, text=f"0 / {self.sequence_length} frames", 
                                     font=('Arial', 10, 'bold'))
        self.frames_label.grid(row=2, column=0, pady=2)
        
        self.skip_warning_label = ttk.Label(record_frame, text="", 
                                           foreground='orange', font=('Arial', 8))
        self.skip_warning_label.grid(row=3, column=0, pady=2)
        
        record_frame.columnconfigure(0, weight=1)
        
        button_frame = ttk.LabelFrame(control_frame, text="Actions", padding="10")
        button_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Button(button_frame, text="Save & Exit", 
                  command=self._save_and_exit).grid(row=0, column=0, 
                                                   sticky=(tk.W, tk.E), pady=2)
        
        button_frame.columnconfigure(0, weight=1)
        
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.stats_label = ttk.Label(stats_frame, text="Total sequences: 0\nGesture classes: 0", 
                                     justify=tk.LEFT, anchor=tk.W)
        self.stats_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        stats_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready - Press Space or button to start recording")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self._update_statistics()
        
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
            self.status_var.set("Tracker initialized - Ready to record")
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
        
        print(f"Camera resolution: {actual_width}x{actual_height}")
    
    def _change_camera(self, event=None):
        """Change camera source"""
        self._open_camera()
        self.status_var.set(f"Switched to camera {self.camera_index.get()}")
    
    def _toggle_recording(self, event=None):
        """Toggle recording state"""
        if not self.is_recording:
            try:
                gesture_id = int(self.current_gesture_id.get())
                if gesture_id < 0:
                    raise ValueError("Gesture ID must be non-negative")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid gesture ID: {str(e)}")
                return
            
            self.is_recording = True
            self.current_sequence = []
            self.frames_recorded = 0
            self.sequence_reference_wrist = None
            self.sequence_reference_scale = None
            self.consecutive_nulls = 0
            self.last_valid_landmarks = None
            self.pending_interpolations = []
            self.progress_var.set(0)
            self.record_button.config(text="Stop Recording (Space)", style="Recording.TButton")
            self.status_var.set(f"Recording sequence for Gesture {gesture_id}...")
            self.skip_warning_label.config(text="")
        else:
            self._stop_recording()
    
    def _stop_recording(self):
        """Stop recording and save sequence"""
        self.is_recording = False
        self.record_button.config(text="Start Recording (Space)")
        
        if len(self.current_sequence) == self.sequence_length:
            try:
                gesture_id = int(self.current_gesture_id.get())
                self.dataset_manager.add_sequence(self.current_sequence, gesture_id)
                self._update_statistics()
                self.status_var.set(f"Sequence saved! Total sequences: {len(self.dataset_manager.data)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save sequence: {str(e)}")
        else:
            self.status_var.set(f"Recording stopped - incomplete sequence ({len(self.current_sequence)}/{self.sequence_length} frames)")
        
        self.current_sequence = []
        self.frames_recorded = 0
        self.consecutive_nulls = 0
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        self.progress_var.set(0)
        self.skip_warning_label.config(text="")
    
    def _discard_sequence(self, reason):
        """Discard the current sequence being recorded"""
        self.is_recording = False
        self.record_button.config(text="Start Recording (Space)")
        self.current_sequence = []
        self.frames_recorded = 0
        self.consecutive_nulls = 0
        self.last_valid_landmarks = None
        self.pending_interpolations = []
        self.progress_var.set(0)
        self.skip_warning_label.config(text="")
        
        self.status_var.set(f"Sequence DISCARDED: {reason}")
        messagebox.showwarning("Sequence Discarded", 
                              f"The current sequence was discarded.\n\nReason: {reason}\n\n"
                              f"Please try recording again with steadier hand visibility.")
    
    def _perform_linear_interpolation(self, start_landmarks, end_landmarks, num_steps):
        """Perform linear interpolation between two landmark sets."""
        interpolated = []
        for i in range(1, num_steps + 1):
            t = i / (num_steps + 1)
            interpolated_frame = start_landmarks + t * (end_landmarks - start_landmarks)
            interpolated.append(interpolated_frame)
        
        return interpolated
    
    def _update_frame(self):
        """Update video frame"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            # Process frame with hand tracker
            detections = self.tracker(frame)
            
            has_hand = False
            if detections:
                for landmarks, hand_bbox, handedness in detections:
                    has_hand = True
                    self._draw_landmarks(landmarks, frame)
                    self._draw_bbox(hand_bbox, frame)
                    hand_type = "Right" if handedness > 0.5 else "Left"
                    cv2.putText(frame, f"Hand: {hand_type}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.is_recording and len(self.current_sequence) < self.sequence_length:
                        if self.pending_interpolations and self.last_valid_landmarks is not None:
                            # Normalize current landmarks first
                            current_normalized = self._normalize_landmarks_for_sequence(landmarks)
                            
                            # Perform linear interpolation for missing frames
                            num_missing = len(self.pending_interpolations)
                            interpolated_frames = self._perform_linear_interpolation(
                                self.last_valid_landmarks,
                                current_normalized,
                                num_missing
                            )
                            
                            # Insert interpolated frames at their pending positions
                            for idx, interp_frame in zip(self.pending_interpolations, interpolated_frames):
                                self.current_sequence[idx] = interp_frame
                            
                            print(f"Interpolated {num_missing} missing frame(s)")
                            self.pending_interpolations = []
                        
                        self.consecutive_nulls = 0
                        self.skip_warning_label.config(text="")
                        
                        normalized_landmarks = self._normalize_landmarks_for_sequence(landmarks)
                        self.current_sequence.append(normalized_landmarks)
                        self.last_valid_landmarks = normalized_landmarks
                        
                        self.frames_recorded = len(self.current_sequence)
                        self.progress_var.set(self.frames_recorded)
                        self.frames_label.config(text=f"{self.frames_recorded} / {self.sequence_length} frames")
                        
                        if len(self.current_sequence) >= self.sequence_length:
                            self._stop_recording()
            
            # Handle missing detection during recording
            if self.is_recording and not has_hand and len(self.current_sequence) < self.sequence_length:
                self.consecutive_nulls += 1
                
                # Update warning display
                if self.consecutive_nulls == 1:
                    self.skip_warning_label.config(text="Missing detection (1)")
                elif self.consecutive_nulls == 2:
                    self.skip_warning_label.config(text="Missing detection (2)")
                
                # Check if we've exceeded the threshold
                if self.consecutive_nulls > self.max_consecutive_nulls:
                    self._discard_sequence(f"Too many consecutive missing detections ({self.consecutive_nulls})")
                else:
                    if self.last_valid_landmarks is not None:
                        placeholder = self.last_valid_landmarks.copy()
                        self.current_sequence.append(placeholder)
                        
                        # Mark this frame index for interpolation
                        self.pending_interpolations.append(len(self.current_sequence) - 1)
                        
                        self.frames_recorded = len(self.current_sequence)
                        self.progress_var.set(self.frames_recorded)
                        self.frames_label.config(text=f"{self.frames_recorded} / {self.sequence_length} frames (pending interp)")
                        
                        if len(self.current_sequence) >= self.sequence_length:
                            self._stop_recording()
            
            # Show recording indicator
            if self.is_recording:
                if has_hand:
                    color = (0, 0, 255)
                    text = "RECORDING"
                elif self.consecutive_nulls > 0:
                    color = (0, 165, 255)
                    text = f"WAITING FOR HAND ({self.consecutive_nulls}/{self.max_consecutive_nulls})"
                else:
                    color = (0, 165, 255)
                    text = "WAITING FOR HAND"
                
                cv2.circle(frame, (30, 70), 15, color, -1)
                cv2.putText(frame, text, (55, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Frame: {len(self.current_sequence)}/{self.sequence_length}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert to PhotoImage and display
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
    
    def _normalize_landmarks_for_sequence(self, landmarks):
        """Normalize landmarks while preserving relative motion"""
        # Store the first frame's wrist position if this is the first frame
        if len(self.current_sequence) == 0:
            self.sequence_reference_wrist = landmarks[0].copy()
            
            # Calculate scale based on hand span (wrist to middle fingertip)
            hand_span = np.linalg.norm(landmarks[12] - landmarks[0])
            
            # Fallback to max finger spread if hand_span is too small
            if hand_span < 0.01:
                hand_span = np.max(np.linalg.norm(landmarks - landmarks[0], axis=1))
            
            if hand_span == 0:
                hand_span = 1.0
                
            self.sequence_reference_scale = hand_span
            
            print(f"Sequence reference scale (hand span): {hand_span:.4f}")
        
        normalized = landmarks - self.sequence_reference_wrist
        
        normalized = normalized / self.sequence_reference_scale
        
        wrist_pos = normalized[0]
        if len(self.current_sequence) % 3 == 0:
            print(f"Frame {len(self.current_sequence)}: Wrist position = ({wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f})")
        
        return normalized.flatten().astype(np.float32)
    
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
    
    def _update_statistics(self):
        """Update statistics display"""
        stats = self.dataset_manager.get_statistics()
        counts = [str(count) for _, count in sorted(stats['sequences_per_class'].items())]
        counts_str = ', '.join(counts) if counts else '0'
        
        stats_text = f"Total sequences: {stats['total_sequences']}\n"
        stats_text += f"Gesture classes: {stats['num_classes']}\n"
        stats_text += f"Sequences per class:\n{counts_str}"
        
        self.stats_label.config(text=stats_text)
    
    def _save_dataset(self):
        """Save dataset"""
        if len(self.dataset_manager.data) == 0:
            messagebox.showwarning("Warning", "No data collected yet")
            return False
        
        filename = self.dataset_path if self.config['mode'] == 'load' else "gesture_dataset.h5"
        
        try:
            self.dataset_manager.save_to_h5(filename)
            messagebox.showinfo("Success", 
                               f"Dataset saved to {filename}\n"
                               f"Total sequences: {len(self.dataset_manager.data)}")
            self.status_var.set(f"Dataset saved to {filename}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
            return False
    
    def _save_and_exit(self):
        """Save dataset and exit"""
        if len(self.dataset_manager.data) == 0:
            result = messagebox.askyesno("Exit", "No data saved. Exit anyway?")
            if result:
                self._cleanup()
                self.root.quit()
        else:
            if self._save_dataset():
                self._cleanup()
                self.root.quit()
    
    def _exit_app(self):
        """Exit application"""
        if len(self.dataset_manager.data) == 0:
            result = messagebox.askyesno("Exit", "No data saved. Exit anyway?")
            if not result:
                return
        else:
            result = messagebox.askyesnocancel("Exit", 
                                               "Do you want to save before exiting?")
            if result is None:
                return
            elif result:
                self._save_dataset()
        
        self._cleanup()
        self.root.quit()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__": 
    config = show_dataset_selector()
    
    if config['mode'] is None:
        exit()
    
    root = tk.Tk()
    app = GestureDataCollector(root, config)
    
    # Handle window close
    def on_closing():
        app._exit_app()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()