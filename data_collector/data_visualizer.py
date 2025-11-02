import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import h5py
from collections import defaultdict


class DatasetVisualizer:

    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dataset Visualizer")
        self.root.geometry("1000x700")
        
        self.dataset_path = None
        self.landmarks_data = None
        self.labels_data = None
        self.metadata = {}
        self.class_samples = defaultdict(list)  # {class_id: [sample_indices]}
        self.available_classes = []
        
        self.current_class = tk.IntVar(value=0)
        self.current_sample_idx = 0
        self.canvas_size = 600
        
        self._setup_ui()
        self._bind_keys()
        
    def _setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0, minsize=300)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10", width=300)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.grid_propagate(False)
        
        ttk.Button(control_frame, text="Load Dataset", 
                  command=self._load_dataset).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.file_label = ttk.Label(control_frame, text="No dataset loaded", wraplength=280)
        self.file_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 20))
        
        ttk.Label(control_frame, text="Select Gesture Class:").grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        self.class_combo = ttk.Combobox(control_frame, textvariable=self.current_class,
                                        state="readonly", width=30)
        self.class_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        self.class_combo.bind('<<ComboboxSelected>>', self._on_class_changed)
        
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(nav_frame, text="Previous", 
                  command=self._previous_sample).pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="Next", 
                  command=self._next_sample).pack(fill=tk.X, pady=2)
        
        info_frame = ttk.LabelFrame(control_frame, text="Sample Info", padding="10")
        info_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=20)
        
        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT, wraplength=260)
        self.info_label.pack(fill=tk.BOTH, expand=True)
        
        stats_frame = ttk.LabelFrame(control_frame, text="Dataset Statistics", padding="10")
        stats_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="", justify=tk.LEFT, wraplength=260)
        self.stats_label.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Landmark Visualization", padding="10")
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, 
                               height=self.canvas_size, bg='black')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        self.status_var = tk.StringVar(value="Ready - Load a dataset to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def _bind_keys(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self._previous_sample())
        self.root.bind('<Right>', lambda e: self._next_sample())
        self.root.bind('a', lambda e: self._previous_sample())
        self.root.bind('A', lambda e: self._previous_sample())
        self.root.bind('d', lambda e: self._next_sample())
        self.root.bind('D', lambda e: self._next_sample())
        
    def _load_dataset(self):
        """Load dataset from H5 file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with h5py.File(filename, 'r') as hf:
                self.landmarks_data = hf['landmarks'][:]
                self.labels_data = hf['labels'][:]
                
                self.metadata = {
                    'total_samples': hf.attrs.get('total_samples', len(self.landmarks_data)),
                    'num_classes': hf.attrs.get('num_classes', 0),
                    'num_landmarks': hf.attrs.get('num_landmarks', 21),
                    'landmark_dims': hf.attrs.get('landmark_dims', 2),
                    'feature_size': hf.attrs.get('feature_size', 42)
                }
            
            self.dataset_path = filename
            self._process_dataset()
            self._update_statistics()
            self._populate_class_selector()
            self._on_class_changed()
            
            self.file_label.config(text=f"Loaded: {filename.split('/')[-1]}")
            self.status_var.set(f"Dataset loaded successfully - {self.metadata['total_samples']} samples")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    
    def _process_dataset(self):
        """Process dataset to organize samples by class"""
        self.class_samples.clear()
        
        for idx, label in enumerate(self.labels_data):
            self.class_samples[int(label)].append(idx)
        
        self.available_classes = sorted(self.class_samples.keys())
    
    def _populate_class_selector(self):
        """Populate the class selection combobox"""
        if not self.available_classes:
            self.class_combo['values'] = []
            return
        
        class_labels = [f"Gesture {cls} ({len(self.class_samples[cls])} samples)" 
                       for cls in self.available_classes]
        self.class_combo['values'] = class_labels
        
        if self.available_classes:
            self.class_combo.current(0)
            self.current_class.set(self.available_classes[0])
    
    def _update_statistics(self):
        """Update statistics display"""
        if self.landmarks_data is None:
            return
        
        stats_text = f"Total Samples: {self.metadata['total_samples']}\n"
        stats_text += f"Number of Classes: {self.metadata['num_classes']}\n"
        stats_text += f"Landmarks per Sample: {self.metadata['num_landmarks']}\n\n"
        stats_text += "Samples per Class:\n"
        
        for class_id in self.available_classes:
            count = len(self.class_samples[class_id])
            stats_text += f"  Gesture {class_id}: {count}\n"
        
        self.stats_label.config(text=stats_text)
    
    def _on_class_changed(self, event=None):
        """Handle class selection change"""
        if not self.available_classes:
            return
        
        # Get selected class from combobox text
        combo_text = self.class_combo.get()
        if combo_text:
            try:
                class_id = int(combo_text.split()[1])
                self.current_class.set(class_id)
            except:
                pass
        
        self.current_sample_idx = 0
        self._visualize_current_sample()
    
    def _next_sample(self):
        """Navigate to next sample"""
        if not self.available_classes:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_samples:
            return
        
        num_samples = len(self.class_samples[current_class])
        self.current_sample_idx = (self.current_sample_idx + 1) % num_samples
        self._visualize_current_sample()
    
    def _previous_sample(self):
        """Navigate to previous sample"""
        if not self.available_classes:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_samples:
            return
        
        num_samples = len(self.class_samples[current_class])
        self.current_sample_idx = (self.current_sample_idx - 1) % num_samples
        self._visualize_current_sample()
    
    def _visualize_current_sample(self):
        """Visualize the current sample on canvas"""
        if self.landmarks_data is None:
            return
        
        current_class = self.current_class.get()
        if current_class not in self.class_samples:
            return
        
        # Get the actual sample index from the dataset
        sample_indices = self.class_samples[current_class]
        if not sample_indices:
            return
        
        dataset_idx = sample_indices[self.current_sample_idx]
        landmarks = self.landmarks_data[dataset_idx]
        
        # Update info display
        num_samples = len(sample_indices)
        info_text = f"Gesture Class: {current_class}\n"
        info_text += f"Sample: {self.current_sample_idx + 1} / {num_samples}\n"
        info_text += f"Dataset Index: {dataset_idx}"
        self.info_label.config(text=info_text)
        
        self._draw_landmarks(landmarks)
        self.status_var.set(f"Viewing Gesture {current_class} - Sample {self.current_sample_idx + 1}/{num_samples}")
    
    def _draw_landmarks(self, landmarks):
        """Draw hand landmarks on the canvas"""
        self.canvas.delete("all")
        
        # Reshape landmarks from (42,) to (21, 2)
        landmarks = landmarks.reshape(21, 2)
        
        # Denormalize landmarks (normalized relative to wrist)
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        
        # Add padding
        padding = 50
        range_x = max_x - min_x
        range_y = max_y - min_y
        max_range = max(range_x, range_y)
        
        if max_range == 0:
            max_range = 1
        
        # Scale to canvas size with padding
        scale = (self.canvas_size - 2 * padding) / max_range
        
        # Center the hand
        center_x = self.canvas_size / 2
        center_y = self.canvas_size / 2
        
        # Transform landmarks to canvas coordinates
        scaled_landmarks = []
        for x, y in landmarks:
            canvas_x = center_x + (x - (min_x + max_x) / 2) * scale
            canvas_y = center_y + (y - (min_y + max_y) / 2) * scale
            scaled_landmarks.append((canvas_x, canvas_y))
        
        # Hand skeleton connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        for idx1, idx2 in connections:
            x1, y1 = scaled_landmarks[idx1]
            x2, y2 = scaled_landmarks[idx2]
            self.canvas.create_line(x1, y1, x2, y2, fill='#1E90FF', width=3)
        
        # Draw landmarks (points)
        palm_joints = [0, 1, 2, 5, 9, 13, 17]
        for idx, (x, y) in enumerate(scaled_landmarks):
            if idx in palm_joints:
                # Palm joints in orange
                color = '#FFA500'
                size = 8
            else:
                # Other joints in green
                color = '#00FF00'
                size = 6
            
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=color, outline='white', width=2
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetVisualizer(root)
    
    # Handle window close
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()