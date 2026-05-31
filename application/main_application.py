"""
main_enterprise.py - Enterprise Andon Dashboard (SolidWorks UI Style)
Industrial-grade PyQt interface for multi-robot assembly line management.
"""

import sys
import threading
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QScrollArea, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
import serial.tools.list_ports

# =====================================================================
# BACKEND CLASSES
# =====================================================================
from application.object_detection import LegoDetector
from application.camera_overlay_selection import CameraOverlay
from application.hand_control_fleet import FleetController

class SafetyMonitor:
    def __init__(self, fleet):
        self.hand_detected = False
        self.running = True
    def run(self):
        while self.running: time.sleep(0.1)
    def stop(self): self.running = False
# =====================================================================

# --- SOLIDWORKS-STYLE CAD CSS ---
BTN_STYLE_FLAT_GREY = """
    QPushButton {
        background-color: #404040;
        color: #E0E0E0;
        border: 1px solid #555555;
        border-radius: 1px;
        font-weight: bold;
        font-family: 'Segoe UI', Arial;
        padding: 8px;
    }
    QPushButton:hover {
        background-color: #4d4d4d;
        border: 1px solid #666666;
    }
    QPushButton:pressed {
        background-color: #2b2b2b;
        border: 1px solid #005A9E; /* SolidWorks Active Blue */
    }
"""

BTN_STYLE_ESTOP_FLAT = """
    QPushButton {
        background-color: #B71C1C;
        color: white;
        border: 2px solid #D32F2F;
        border-radius: 1px;
        font-weight: bold;
        font-family: 'Segoe UI', Arial;
        font-size: 14pt;
        padding: 12px;
        letter-spacing: 2px;
    }
    QPushButton:hover {
        background-color: #D32F2F;
    }
    QPushButton:pressed {
        background-color: #7F0000;
        border: 2px solid #B71C1C;
    }
"""


class RobotStatusWidget(QWidget):
    """Widget displaying individual robot arm status"""
    def __init__(self, arm_id, com_port):
        super().__init__()
        self.arm_id = arm_id
        self.com_port = com_port
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        # Arm title (CAD style header)
        title = QLabel(f"ARM {arm_id} [{com_port}]")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; background-color: #2b2b2b; padding: 4px; border: 1px solid #444;")
        layout.addWidget(title)
        
        # Status Badges (Flat square design)
        badge_style = "background-color: #383838; color: #D0D0D0; padding: 6px; border: 1px solid #444; border-radius: 1px; font-family: 'Segoe UI';"
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet(badge_style)
        layout.addWidget(self.status_label)
        
        self.task_label = QLabel("Task: Waiting")
        self.task_label.setStyleSheet(badge_style)
        layout.addWidget(self.task_label)
        
        self.connection_label = QLabel("Connection: Connected")
        self.connection_label.setStyleSheet(badge_style)
        layout.addWidget(self.connection_label)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #303030; border: 1px solid #222; padding: 6px; border-radius: 1px;")
    
    def update_status(self, status, task, connection):
        self.status_label.setText(f"Status: {status}")
        self.task_label.setText(f"Task: {task}")
        self.connection_label.setText(f"Connection: {connection}")


class AndonBoard(QWidget):
    """Andon board displaying system health and safety status"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        font = QFont("Segoe UI", 12, QFont.Bold)
        self.base_style = "color: #D0D0D0; padding: 6px; background: transparent;"
        
        self.system_status = QLabel("■ SYSTEM: Running")
        self.system_status.setFont(font)
        layout.addWidget(self.system_status)
        
        self.safety_status = QLabel("■ SAFETY: CLEAR")
        self.safety_status.setFont(font)
        layout.addWidget(self.safety_status)
        
        self.mode_status = QLabel("■ MODE: Manual")
        self.mode_status.setFont(font)
        layout.addWidget(self.mode_status)
        
        self.active_arms = QLabel("■ ACTIVE ARMS: 0")
        self.active_arms.setFont(font)
        layout.addWidget(self.active_arms)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            background-color: #2b2b2b; 
            border: 1px solid #111; 
            border-top: 4px solid #555; 
            padding: 10px; 
            border-radius: 1px;
        """)
        self.set_system_status("Running", "GREEN")
        self.set_safety_status("CLEAR - No hands detected", "GREEN")
    
    def set_system_status(self, status, color):
        dot_color = {"GREEN": "#4CAF50", "YELLOW": "#FFC107", "RED": "#F44336"}.get(color, "#757575")
        self.system_status.setText(f"<span style='color:{dot_color};'>■</span> SYSTEM: {status}")
        self.system_status.setStyleSheet(self.base_style)
    
    def set_safety_status(self, status, color):
        dot_color = {"GREEN": "#4CAF50", "YELLOW": "#FFC107", "RED": "#F44336"}.get(color, "#757575")
        self.safety_status.setText(f"<span style='color:{dot_color};'>■</span> SAFETY: {status}")
        self.safety_status.setStyleSheet(self.base_style)
    
    def set_mode(self, mode):
        self.mode_status.setText(f"<span style='color:#005A9E;'>■</span> MODE: {mode}")
        self.mode_status.setStyleSheet(self.base_style)
    
    def set_active_arms(self, count):
        self.active_arms.setText(f"<span style='color:#D0D0D0;'>■</span> ACTIVE ARMS: {count}")
        self.active_arms.setStyleSheet(self.base_style)


class VideoWorker(QThread):
    """Worker thread for camera feed processing"""
    frame_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, detector, overlay, camera_index=0):
        super().__init__()
        self.detector = detector
        self.overlay = overlay
        self.camera_index = camera_index
        self.running = True
        self.cap = None
    
    def run(self):
        if self.camera_index is None:
            print("[VIDEO] No camera selected")
            while self.running:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA SELECTED", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                self.frame_updated.emit(frame)
                time.sleep(0.5)
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[VIDEO] Camera {self.camera_index} not available")
            while self.running:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO VIDEO FEED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                self.frame_updated.emit(frame)
                time.sleep(0.5)
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO VIDEO FEED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                self.frame_updated.emit(frame)
                time.sleep(0.1)
                continue
            
            detections = self.detector.detect(frame)
            annotated_frame = self.detector.get_annotated_frame(frame)
            self.overlay.update_detections(detections, annotated_frame)
            rendered_frame = self.overlay.render_frame(annotated_frame, detections)
            self.frame_updated.emit(rendered_frame)
            time.sleep(0.01)
    
    def stop(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class EnterpriseMainWindow(QMainWindow):
    """Enterprise Andon Dashboard - Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TMMC Assembly Control System - Collaborative Robotics")
        self.setGeometry(100, 100, 1600, 900)
        
        # SolidWorks / CAD Deep Grey styling
        self.setStyleSheet("""
            QMainWindow { background-color: #282828; }
            QWidget { color: #E0E0E0; font-family: 'Segoe UI', Arial; }
            QLabel { background: transparent; }
            
            /* COMBOBOX STYLING FIX - Forces Dropdown to be dark */
            QComboBox { 
                background-color: #383838; 
                color: #E0E0E0;
                border: 1px solid #555; 
                padding: 6px; 
                border-radius: 1px; 
            }
            QComboBox::drop-down { border: none; background-color: #383838; }
            QComboBox QAbstractItemView {
                background-color: #383838;
                color: #E0E0E0;
                selection-background-color: #005A9E; /* SW Active Blue */
                selection-color: #ffffff;
                border: 1px solid #555;
            }
        """)
        
        self.current_mode = "Manual"
        self.detector = LegoDetector()
        self.overlay = CameraOverlay(self.on_lego_selected)
        self.overlay.set_mode(self.current_mode)
        self.fleet_controller = FleetController()
        self.safety_monitor = SafetyMonitor(self.fleet_controller)
        self.selected_target = None
        
        self.safety_thread = threading.Thread(target=self.safety_monitor.run, daemon=True)
        self.safety_thread.start()
        
        self.setup_ui()
        
        selected_camera = self.get_selected_camera_index()
        self.video_worker = VideoWorker(self.detector, self.overlay, camera_index=selected_camera)
        self.video_worker.frame_updated.connect(self.update_frame)
        self.video_worker.start()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_robot_status)
        self.update_timer.start(500) 
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # --- Left section - Video feed ---
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(1000, 600)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter) 
        # CAD-style viewport border
        self.video_label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #444; border-radius: 1px;")
        self.video_label.mousePressEvent = self.on_video_click
        left_layout.addWidget(self.video_label)
        
        # --- Right section - Controls ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)
        
        title = QLabel("FLEET CONTROL PANEL")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; border-bottom: 2px solid #555; padding-bottom: 4px;")
        right_layout.addWidget(title)
        
        mode_layout = QHBoxLayout()
        mode_label = QLabel("OPERATING MODE:")
        mode_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Custom", "Automatic"])
        self.mode_combo.setFont(QFont("Segoe UI", 10))
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo, 1)
        right_layout.addLayout(mode_layout)
        
        self.start_button = QPushButton("START SEQUENCE")
        self.start_button.setMinimumHeight(40)
        self.start_button.setStyleSheet(BTN_STYLE_FLAT_GREY)
        self.start_button.clicked.connect(self.on_start_clicked)
        right_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("PAUSE / HOLD")
        self.pause_button.setMinimumHeight(40)
        self.pause_button.setStyleSheet(BTN_STYLE_FLAT_GREY)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        right_layout.addWidget(self.pause_button)
        
        camera_label = QLabel("CAMERA SOURCE:")
        camera_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        camera_label.setStyleSheet("color: #B0B0B0;")
        right_layout.addWidget(camera_label)

        camera_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setFont(QFont("Segoe UI", 10))
        camera_layout.addWidget(self.camera_combo, stretch=2)

        camera_refresh_btn = QPushButton("↻")
        camera_refresh_btn.setFixedSize(32, 32)
        camera_refresh_btn.setStyleSheet(BTN_STYLE_FLAT_GREY)
        camera_refresh_btn.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(camera_refresh_btn)

        right_layout.addLayout(camera_layout)
        self.refresh_cameras()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        
        self.stop_button = QPushButton("EMERGENCY STOP")
        self.stop_button.setMinimumHeight(55)
        self.stop_button.setStyleSheet(BTN_STYLE_ESTOP_FLAT)
        self.stop_button.clicked.connect(self.on_emergency_stop)
        right_layout.addWidget(self.stop_button)
        
        right_layout.addSpacing(10)
        
        com_label = QLabel("HARDWARE CONNECTION:")
        com_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        com_label.setStyleSheet("color: #B0B0B0;")
        right_layout.addWidget(com_label)

        port_layout = QHBoxLayout()
        self.com_combo = QComboBox()
        self.com_combo.setFont(QFont("Segoe UI", 10))
        self.refresh_ports()
        
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedSize(32, 32)
        refresh_btn.setStyleSheet(BTN_STYLE_FLAT_GREY)
        refresh_btn.clicked.connect(self.refresh_ports)
        
        connect_btn = QPushButton("Connect")
        connect_btn.setFixedHeight(32)
        connect_btn.setStyleSheet(BTN_STYLE_FLAT_GREY)
        connect_btn.clicked.connect(self.on_connect_arm)
        
        port_layout.addWidget(self.com_combo, stretch=2)
        port_layout.addWidget(refresh_btn)
        port_layout.addWidget(connect_btn, stretch=1)
        right_layout.addLayout(port_layout)
        
        right_layout.addSpacing(5)
        
        robot_label = QLabel("CONNECTED FLEET:")
        robot_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        robot_label.setStyleSheet("color: #B0B0B0; border-bottom: 1px solid #444; padding-bottom: 2px;")
        right_layout.addWidget(robot_label)
        
        scroll = QScrollArea()
        scroll.setStyleSheet("""
            QScrollArea { background-color: #252526; border: 1px solid #111; }
            QScrollBar:vertical { background: #2a2a2a; width: 14px; }
            QScrollBar::handle:vertical { background: #555; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        scroll.setWidgetResizable(True)
        
        self.robot_status_container = QWidget()
        self.robot_status_container.setStyleSheet("background-color: #252526;")
        self.robot_status_layout = QVBoxLayout()
        self.robot_status_container.setLayout(self.robot_status_layout)
        scroll.setWidget(self.robot_status_container)
        
        right_layout.addWidget(scroll, stretch=1)
        right_layout.addSpacing(10)

        self.andon_board = AndonBoard()
        right_layout.addWidget(self.andon_board)
        
        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)
        central_widget.setLayout(main_layout)
    
    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def on_video_click(self, event):
        x, y = event.x(), event.y()
        self.overlay.handle_click(x, y, self.video_label.width(), self.video_label.height())
    
    def on_lego_selected(self, lego_id, location):
        self.selected_target = {"id": lego_id, "location": location}
        self.andon_board.set_system_status(f"TARGET SELECTED: {lego_id}", "YELLOW")
        print(f"[DASHBOARD] Selected target {lego_id} at {location}")
    
    def on_mode_changed(self, mode):
        self.current_mode = mode
        self.andon_board.set_mode(mode)
        self.overlay.set_mode(mode)
    
    def on_start_clicked(self):
        if self.current_mode == "Manual":
            if not self.selected_target:
                print("[DASHBOARD] No target selected")
                self.andon_board.set_system_status("NO TARGET SELECTED", "RED")
                return
            available_arm = self.fleet_controller.get_closest_available_arm(self.selected_target["location"])
            if available_arm:
                self.fleet_controller.dispatch_task(available_arm, self.selected_target["location"])
                self.andon_board.set_system_status("ACTIVE", "GREEN")
                print(f"[DASHBOARD] Dispatched Arm {available_arm} to pick target {self.selected_target['id']}")
                self.selected_target = None
            else:
                print("[DASHBOARD] No available arm to dispatch")
                self.andon_board.set_system_status("NO AVAILABLE ARM", "RED")
        elif self.current_mode == "Automatic":
            self.fleet_controller.start_automatic_mode()
            self.andon_board.set_system_status("ACTIVE", "GREEN")
    
    def on_pause_clicked(self):
        self.fleet_controller.pause_all()
        self.andon_board.set_system_status("PAUSED", "YELLOW")
    
    def on_emergency_stop(self):
        self.fleet_controller.emergency_stop_all()
        self.andon_board.set_safety_status("EMERGENCY STOP ACTIVATED", "RED")
    
    def refresh_robot_status(self):
        fleet_status = self.fleet_controller.get_fleet_status()
        
        for i in reversed(range(self.robot_status_layout.count())): 
            widget_to_remove = self.robot_status_layout.itemAt(i).widget()
            if widget_to_remove: widget_to_remove.setParent(None)
        
        active_count = 0
        for arm_id, status in fleet_status.items():
            widget = RobotStatusWidget(arm_id, status["com_port"])
            widget.update_status(status["state"], status["task"], status["connection"])
            self.robot_status_layout.addWidget(widget)
            if status["connection"] == "Connected": active_count += 1
                
        self.robot_status_layout.addStretch() 
        self.andon_board.set_active_arms(active_count)
        
        if self.safety_monitor.hand_detected:
            self.andon_board.set_safety_status("HAND DETECTED - Safety Override", "RED")
        else:
            self.andon_board.set_safety_status("CLEAR", "GREEN")
    
    def refresh_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        available_cameras = []
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(index)
                cap.release()
        
        if not available_cameras:
            self.camera_combo.addItem("No Cameras Found")
        else:
            for index in available_cameras:
                self.camera_combo.addItem(f"Camera {index}", index)
            self.camera_combo.setCurrentIndex(0)
        self.camera_combo.blockSignals(False)

    def refresh_ports(self):
        self.com_combo.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.com_combo.addItem("No Ports Found")
        else:
            for port, desc, hwid in sorted(ports):
                self.com_combo.addItem(f"{port}")
                
    def on_connect_arm(self):
        selected_port = self.com_combo.currentText()
        if selected_port != "No Ports Found" and selected_port != "":
            print(f"[DASHBOARD] Connecting to {selected_port}...")
            # THIS NOW WORKS AND ADDS TO THE UI
            self.fleet_controller.add_arm(selected_port)
            self.andon_board.set_system_status(f"CONNECTING {selected_port}...", "YELLOW")

    def get_selected_camera_index(self):
        if self.camera_combo.currentText() == "No Cameras Found":
            return None
        data = self.camera_combo.currentData()
        return data if data is not None else None

    def on_camera_changed(self, index):
        if self.camera_combo.currentText() == "No Cameras Found":
            return
        selected_index = self.camera_combo.currentData()
        if selected_index is None:
            return
        self.restart_video_worker(selected_index)

    def restart_video_worker(self, camera_index):
        if hasattr(self, 'video_worker') and self.video_worker is not None:
            self.video_worker.stop()
            self.video_worker.wait(1000)
        self.video_worker = VideoWorker(self.detector, self.overlay, camera_index=camera_index)
        self.video_worker.frame_updated.connect(self.update_frame)
        self.video_worker.start()
            
    def closeEvent(self, event):
        self.video_worker.stop()
        self.safety_monitor.stop()
        self.fleet_controller.shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Required on Windows to ensure CSS drop-down overrides work properly
    window = EnterpriseMainWindow()
    window.show()
    sys.exit(app.exec_())