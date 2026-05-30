"""
main.py - The Hub
Main graphical window for the collaborative robotic arm system.
Displays the live camera feed with bounding boxes around detected Lego pieces.
Houses mode-selection buttons and routes commands between vision and control scripts.
"""

import sys
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from object_detection import LegoDetector
from hand_control import RobotController
from hand_safety import SafetyMonitor
from camera_overlay_selection import ClickHandler


class VideoWorker(QThread):
    """Worker thread for camera feed processing"""
    frame_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, detector, click_handler):
        super().__init__()
        self.detector = detector
        self.click_handler = click_handler
        self.running = True
        
    def run(self):
        """Continuously process camera frames"""
        import pyarabic.open3d  # Camera library - adjust based on your camera module
        # TODO: Initialize camera based on your camera type
        cap = cv2.VideoCapture(0)  # Default camera
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect Lego pieces
            detections = self.detector.detect(frame)
            
            # Draw bounding boxes
            annotated_frame = self.detector.draw_detections(frame, detections)
            
            # Store current detections for click handler
            self.click_handler.set_detections(detections, annotated_frame.copy())
            
            # Emit frame for display
            self.frame_updated.emit(annotated_frame)
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collaborative Robot Arm - Lego Collection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.detector = LegoDetector()
        self.robot_controller = RobotController()
        self.safety_monitor = SafetyMonitor(self.robot_controller)
        self.click_handler = ClickHandler(self.on_lego_selected)
        
        # Start safety monitor in background
        self.safety_thread = threading.Thread(target=self.safety_monitor.run, daemon=True)
        self.safety_thread.start()
        
        # Setup UI
        self.setup_ui()
        
        # Start video worker
        self.video_worker = VideoWorker(self.detector, self.click_handler)
        self.video_worker.frame_updated.connect(self.update_frame)
        self.video_worker.start()
        
        self.current_mode = "Manual"
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left side - Video feed
        left_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.mousePressEvent = self.on_video_click
        left_layout.addWidget(self.video_label)
        
        # Right side - Controls
        right_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Control Panel")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        right_layout.addWidget(title)
        
        # Mode selection
        mode_label = QLabel("Operating Mode:")
        mode_font = QFont()
        mode_font.setPointSize(11)
        mode_label.setFont(mode_font)
        right_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Custom", "Automatic"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        right_layout.addWidget(self.mode_combo)
        
        right_layout.addSpacing(20)
        
        # Status display
        status_label = QLabel("Status:")
        status_label.setFont(mode_font)
        right_layout.addWidget(status_label)
        
        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        right_layout.addWidget(self.status_text)
        
        right_layout.addSpacing(20)
        
        # Control buttons
        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.on_start_clicked)
        right_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Emergency Stop")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold;")
        self.stop_button.clicked.connect(self.on_emergency_stop)
        right_layout.addWidget(self.stop_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setMinimumHeight(40)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        right_layout.addWidget(self.pause_button)
        
        right_layout.addSpacing(20)
        
        # Safety status
        safety_label = QLabel("Safety Status:")
        safety_label.setFont(mode_font)
        right_layout.addWidget(safety_label)
        
        self.safety_status = QLabel("✓ Safe Zone")
        self.safety_status.setStyleSheet("background-color: #90EE90; padding: 10px; border-radius: 5px; font-weight: bold;")
        right_layout.addWidget(self.safety_status)
        
        right_layout.addStretch()
        
        # Assemble main layout
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        central_widget.setLayout(main_layout)
        
    def update_frame(self, frame):
        """Update the video display"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaledToWidth(self.video_label.width())
        self.video_label.setPixmap(scaled_pixmap)
        
    def on_video_click(self, event):
        """Handle clicks on the video feed"""
        x = event.x()
        y = event.y()
        self.click_handler.handle_click(x, y, self.video_label.width(), self.video_label.height())
        
    def on_lego_selected(self, lego_id, location):
        """Callback when a Lego piece is selected"""
        self.status_text.setText(f"Target: Lego {lego_id} at {location}")
        if self.current_mode == "Manual":
            self.robot_controller.move_to_and_pick(location)
            
    def on_mode_changed(self, mode):
        """Handle mode selection change"""
        self.current_mode = mode
        self.status_text.setText(f"Mode: {mode}")
        if mode == "Automatic":
            self.on_start_clicked()
            
    def on_start_clicked(self):
        """Start the operation"""
        self.status_text.setText(f"Running in {self.current_mode} mode...")
        if self.current_mode == "Automatic":
            self.robot_controller.start_automatic_mode()
            
    def on_pause_clicked(self):
        """Pause the operation"""
        self.robot_controller.pause()
        self.status_text.setText("Paused")
        
    def on_emergency_stop(self):
        """Emergency stop - robot retracts immediately"""
        self.robot_controller.emergency_stop()
        self.status_text.setText("⚠ EMERGENCY STOP ACTIVATED")
        
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.video_worker.stop()
        self.safety_monitor.stop()
        self.robot_controller.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
