"""
main_enterprise.py - Enterprise Andon Dashboard
Industrial-grade PyQt interface for multi-robot assembly line management.
Features real-time system status, Andon board alerts, and individual arm status indicators.
"""

import sys
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QFrame, QScrollArea, QGridLayout,
    QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from object_detection import LegoDetector
from hand_control_fleet import FleetController
from hand_safety import SafetyMonitor
from camera_overlay_selection import ClickHandler


class RobotStatusWidget(QWidget):
    """Widget displaying individual robot arm status"""
    
    def __init__(self, arm_id, com_port):
        super().__init__()
        self.arm_id = arm_id
        self.com_port = com_port
        self.status = "Idle"
        self.task = "Waiting"
        self.connection_status = "Connected"
        
        layout = QVBoxLayout()
        
        # Arm title
        title = QLabel(f"Arm {arm_id} ({com_port})")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Status display
        self.status_label = QLabel(f"Status: {self.status}")
        self.status_label.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 3px;")
        layout.addWidget(self.status_label)
        
        # Task display
        self.task_label = QLabel(f"Task: {self.task}")
        self.task_label.setStyleSheet("background-color: #2196F3; color: white; padding: 5px; border-radius: 3px;")
        layout.addWidget(self.task_label)
        
        # Connection status
        self.connection_label = QLabel(f"Connection: {self.connection_status}")
        self.connection_label.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 3px;")
        layout.addWidget(self.connection_label)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #444; padding: 10px; border-radius: 5px;")
    
    def update_status(self, status, task, connection):
        """Update robot status display"""
        self.status = status
        self.task = task
        self.connection_status = connection
        
        # Color code status
        status_colors = {
            "Idle": "#4CAF50",
            "Moving": "#2196F3",
            "Picking": "#FF9800",
            "Error": "#F44336",
            "Paused": "#FFC107"
        }
        
        color = status_colors.get(status, "#757575")
        self.status_label.setText(f"Status: {status}")
        self.status_label.setStyleSheet(f"background-color: {color}; color: white; padding: 5px; border-radius: 3px;")
        
        self.task_label.setText(f"Task: {task}")
        self.connection_label.setText(f"Connection: {connection}")
        
        conn_color = "#4CAF50" if connection == "Connected" else "#F44336"
        self.connection_label.setStyleSheet(f"background-color: {conn_color}; color: white; padding: 5px; border-radius: 3px;")


class AndonBoard(QWidget):
    """Andon board displaying system health and safety status"""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        self.system_status = QLabel("● SYSTEM: GREEN - Running")
        self.system_status.setFont(QFont("Arial", 16, QFont.Bold))
        self.system_status.setStyleSheet("color: #4CAF50; padding: 10px;")
        layout.addWidget(self.system_status)
        
        self.safety_status = QLabel("● SAFETY: CLEAR - No hands detected")
        self.safety_status.setFont(QFont("Arial", 16, QFont.Bold))
        self.safety_status.setStyleSheet("color: #4CAF50; padding: 10px;")
        layout.addWidget(self.safety_status)
        
        self.mode_status = QLabel("● MODE: Manual")
        self.mode_status.setFont(QFont("Arial", 16, QFont.Bold))
        self.mode_status.setStyleSheet("color: #2196F3; padding: 10px;")
        layout.addWidget(self.mode_status)
        
        self.active_arms = QLabel("● ACTIVE ARMS: 0")
        self.active_arms.setFont(QFont("Arial", 14, QFont.Bold))
        self.active_arms.setStyleSheet("color: #FF9800; padding: 10px;")
        layout.addWidget(self.active_arms)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;")
    
    def set_system_status(self, status, color):
        """Update system status"""
        color_hex = {"GREEN": "#4CAF50", "YELLOW": "#FFC107", "RED": "#F44336"}.get(color, "#757575")
        self.system_status.setText(f"● SYSTEM: {status}")
        self.system_status.setStyleSheet(f"color: {color_hex}; padding: 10px;")
    
    def set_safety_status(self, status, color):
        """Update safety status"""
        color_hex = {"GREEN": "#4CAF50", "YELLOW": "#FFC107", "RED": "#F44336"}.get(color, "#757575")
        self.safety_status.setText(f"● SAFETY: {status}")
        self.safety_status.setStyleSheet(f"color: {color_hex}; padding: 10px;")
    
    def set_mode(self, mode):
        """Update operation mode display"""
        self.mode_status.setText(f"● MODE: {mode}")
    
    def set_active_arms(self, count):
        """Update active arms count"""
        self.active_arms.setText(f"● ACTIVE ARMS: {count}")


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
        cap = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            detections = self.detector.detect(frame)
            annotated_frame = self.detector.draw_detections(frame, detections)
            self.click_handler.set_detections(detections, annotated_frame.copy())
            self.frame_updated.emit(annotated_frame)
    
    def stop(self):
        self.running = False


class EnterpriseMainWindow(QMainWindow):
    """Enterprise Andon Dashboard - Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toyota Innovation Challenge - Robot Assembly Line Control System")
        self.setGeometry(0, 0, 1920, 1080)
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        
        # Initialize components
        self.detector = LegoDetector()
        self.fleet_controller = FleetController()
        self.safety_monitor = SafetyMonitor(self.fleet_controller)
        self.click_handler = ClickHandler(self.on_lego_selected)
        
        # Track robot status
        self.robot_statuses = {}
        
        # Start safety monitor
        self.safety_thread = threading.Thread(target=self.safety_monitor.run, daemon=True)
        self.safety_thread.start()
        
        # Setup UI
        self.setup_ui()
        
        # Start video worker
        self.video_worker = VideoWorker(self.detector, self.click_handler)
        self.video_worker.frame_updated.connect(self.update_frame)
        self.video_worker.start()
        
        self.current_mode = "Manual"
        
        # Update timer for status refresh
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_robot_status)
        self.update_timer.start(500)  # Update every 500ms
    
    def setup_ui(self):
        """Setup the enterprise user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # --- Left section - Video feed takes entire left side ---
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(1000, 600)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter) 
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #444;")
        self.video_label.mousePressEvent = self.on_video_click
        left_layout.addWidget(self.video_label)
        
        # --- Right section - Controls, Robot Status, and Andon Board ---
        right_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Fleet Control Panel")
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
        self.mode_combo.setStyleSheet("background-color: #2a2a2a; color: white; padding: 5px;")
        self.mode_combo.addItems(["Manual", "Custom", "Automatic"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        right_layout.addWidget(self.mode_combo)
        
        right_layout.addSpacing(15)
        
        # Control buttons
        self.start_button = QPushButton("Start")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            font-size: 12pt;
            border-radius: 5px;
            padding: 10px;
        """)
        self.start_button.clicked.connect(self.on_start_clicked)
        right_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("EMERGENCY STOP")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setStyleSheet("""
            background-color: #F44336;
            color: white;
            font-weight: bold;
            font-size: 12pt;
            border-radius: 5px;
            padding: 10px;
        """)
        self.stop_button.clicked.connect(self.on_emergency_stop)
        right_layout.addWidget(self.stop_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setMinimumHeight(40)
        self.pause_button.setStyleSheet("""
            background-color: #FFC107;
            color: black;
            font-weight: bold;
            font-size: 11pt;
            border-radius: 5px;
            padding: 8px;
        """)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        right_layout.addWidget(self.pause_button)
        
        right_layout.addSpacing(20)
        
        # Robot status section
        robot_label = QLabel("Connected Robot Arms:")
        robot_label.setFont(mode_font)
        right_layout.addWidget(robot_label)
        
        # Scroll area for robot status
        scroll = QScrollArea()
        scroll.setStyleSheet("background-color: #2a2a2a; border: 1px solid #444;")
        scroll.setWidgetResizable(True)
        
        self.robot_status_container = QWidget()
        self.robot_status_layout = QVBoxLayout()
        self.robot_status_container.setLayout(self.robot_status_layout)
        scroll.setWidget(self.robot_status_container)
        
        right_layout.addWidget(scroll)
        
        right_layout.addSpacing(20)

        # Andon Board moved to the bottom of the right column
        self.andon_board = AndonBoard()
        right_layout.addWidget(self.andon_board)
        
        # Assemble layout
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
        # Scaled to maintain aspect ratio and fit properly
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def on_video_click(self, event):
        """Handle clicks on the video feed"""
        x = event.x()
        y = event.y()
        self.click_handler.handle_click(x, y, self.video_label.width(), self.video_label.height())
    
    def on_lego_selected(self, lego_id, location):
        """Callback when a Lego piece is selected"""
        if self.current_mode == "Manual":
            # Find closest available robot and dispatch
            available_arm = self.fleet_controller.get_closest_available_arm(location)
            if available_arm:
                self.fleet_controller.dispatch_task(available_arm, location)
                print(f"[DASHBOARD] Dispatched Arm {available_arm} to pick Lego {lego_id}")
    
    def on_mode_changed(self, mode):
        """Handle mode selection change"""
        self.current_mode = mode
        self.andon_board.set_mode(mode)
        print(f"[DASHBOARD] Mode changed to {mode}")
    
    def on_start_clicked(self):
        """Start the operation"""
        if self.current_mode == "Automatic":
            self.fleet_controller.start_automatic_mode()
        print(f"[DASHBOARD] Started in {self.current_mode} mode")
    
    def on_pause_clicked(self):
        """Pause all robot operations"""
        self.fleet_controller.pause_all()
        self.andon_board.set_system_status("PAUSED", "YELLOW")
        print("[DASHBOARD] All operations paused")
    
    def on_emergency_stop(self):
        """Emergency stop - all robots retract immediately"""
        self.fleet_controller.emergency_stop_all()
        self.andon_board.set_safety_status("EMERGENCY STOP ACTIVATED", "RED")
        print("[DASHBOARD] EMERGENCY STOP - All robots retracting")
    
    def refresh_robot_status(self):
        """Refresh robot status display"""
        # Get current fleet status
        fleet_status = self.fleet_controller.get_fleet_status()
        
        # Clear old widgets
        for i in reversed(range(self.robot_status_layout.count())): 
            self.robot_status_layout.itemAt(i).widget().setParent(None)
        
        # Add status widgets for each arm
        active_count = 0
        for arm_id, status in fleet_status.items():
            widget = RobotStatusWidget(arm_id, status["com_port"])
            widget.update_status(status["state"], status["task"], status["connection"])
            self.robot_status_layout.addWidget(widget)
            if status["connection"] == "Connected":
                active_count += 1
        
        # Update Andon board
        self.andon_board.set_active_arms(active_count)
        
        # Update safety status
        if self.safety_monitor.hand_detected:
            self.andon_board.set_safety_status("HAND DETECTED - Emergency Stop Active", "RED")
        else:
            self.andon_board.set_safety_status("CLEAR - No hands detected", "GREEN")
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.video_worker.stop()
        self.safety_monitor.stop()
        self.fleet_controller.shutdown()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EnterpriseMainWindow()
    window.show()
    sys.exit(app.exec_())