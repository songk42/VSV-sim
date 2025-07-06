import os
import csv
import numpy as np
from typing import List, Tuple
from enum import Enum

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QSlider, QFrame,
                               QMessageBox)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QFont, QKeySequence, QShortcut

import simulation as sim


class AnimationState(Enum):
    """Enumeration for animation states"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    RECORDING = "recording"


class SimulationWorker(QThread):
    """Worker thread for running the simulation calculations"""
    simulation_complete = Signal(list, int)
    progress_update = Signal(int, int)  # current, total
    error_occurred = Signal(str)  # error message

    def __init__(self, config: sim.SimulationConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            coords = []
            max_n_steps = 0
            i = 0
            n_exited = 0
            exit_times = []

            while i < self.config.n_particles:
                # Emit progress update
                self.progress_update.emit(i, self.config.n_particles)

                try:
                    x_coords, y_coords, throw_out, exit_time, _, _, _, _ = sim.move(
                        self.config.total_time,
                        self.config.p_driv,
                        self.config.trap_dist,
                        self.config.trap_std,
                        self.config.time_between,
                        2*np.pi*i/self.config.n_particles,
                        self.config.dt,
                    )
                except Exception as e:
                    self.error_occurred.emit(f"Error simulating particle {i}: {str(e)}")
                    return

                max_n_steps = max(len(x_coords), max_n_steps)
                if throw_out:
                    print(f"Discarded particle {i} (entered nucleus)")
                    continue
                else:
                    if exit_time != -1:
                        n_exited += 1
                        exit_times.append(exit_time)
                    coords.append([x_coords, y_coords])
                    print(f"Particle {i} completed successfully")
                    i += 1

            # Final progress update
            self.progress_update.emit(self.config.n_particles, self.config.n_particles)

            print(f"Simulation complete: {n_exited}/{self.config.n_particles} particles exited")
            if exit_times:
                print(f"Average exit time: {np.mean(exit_times):.2e} s")

            self.simulation_complete.emit(coords, max_n_steps)

        except Exception as e:
            self.error_occurred.emit(f"Simulation failed: {str(e)}")


class SimulationCanvas(QWidget):
    """Custom widget for drawing the simulation"""

    # Constants for rendering
    SCALE_FACTOR = 1e7
    BACKGROUND_COLOR = QColor(0, 0, 0)
    CELL_COLOR = QColor(255, 255, 0)  # Yellow
    NUCLEUS_COLOR = QColor(255, 255, 255)  # White
    TRAIL_COLOR = QColor(255, 255, 255)  # White
    FALLBACK_PARTICLE_SIZE = 8

    def __init__(self, width: int = 600, height: int = 600):
        super().__init__()
        self.width = width
        self.height = height
        self.radius_x = self.width / 2
        self.radius_y = self.height / 2

        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")

        # Particle data
        self.coords: List[List[List[float]]] = []
        self.particle_positions: List[List[float]] = []
        self.trails: List[List[Tuple[float, float, float, float]]] = []
        self.show_trails = True

        # Load particle graphics
        self.particle_pixmap = self._create_particle_pixmap()

    def _create_particle_pixmap(self) -> QPixmap:
        """Load particle image from file with fallback"""
        try:
            pixmap = QPixmap("dot-2.png")
            if not pixmap.isNull():
                return pixmap
            raise FileNotFoundError("dot-2.png not found or invalid")
        except Exception as e:
            print(f"Loading dot-2.png failed: {e}. Using fallback graphics.")
            return self._create_fallback_particle()

    def _create_fallback_particle(self) -> QPixmap:
        """Create a fallback particle graphic"""
        size = self.FALLBACK_PARTICLE_SIZE
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self.TRAIL_COLOR))
        painter.setPen(QPen(self.TRAIL_COLOR))
        painter.drawEllipse(0, 0, size, size)
        painter.end()

        return pixmap

    def set_coords(self, coords: List[List[List[float]]]):
        """Set particle coordinates"""
        self.coords = coords
        self.particle_positions = [[0.0, 0.0] for _ in range(len(coords))]
        self.trails = [[] for _ in range(len(coords))]

    def toggle_trails(self, show: bool):
        """Toggle trail visibility"""
        self.show_trails = show
        self.update()

    def reset_animation(self):
        """Reset animation to beginning"""
        self.particle_positions = [[0.0, 0.0] for _ in range(len(self.coords))]
        self.trails = [[] for _ in range(len(self.coords))]
        self.update()

    def update_frame(self, frame_number: int):
        """Update particle positions for current frame"""
        if not self.coords:
            return

        # Clear previous trails
        self.trails = [[] for _ in range(len(self.coords))]

        for i, coord in enumerate(self.coords):
            if frame_number < len(coord[0]):
                # Build complete trail up to current frame
                for f in range(1, frame_number + 1):
                    if f < len(coord[0]):
                        prev_x = self.radius_x + coord[0][f-1] * self.SCALE_FACTOR
                        prev_y = self.radius_y + coord[1][f-1] * self.SCALE_FACTOR
                        curr_x = self.radius_x + coord[0][f] * self.SCALE_FACTOR
                        curr_y = self.radius_y + coord[1][f] * self.SCALE_FACTOR
                        
                        self.trails[i].append((prev_x, prev_y, curr_x, curr_y))

                # Update current particle position
                self.particle_positions[i] = [
                    self.radius_x + coord[0][frame_number] * self.SCALE_FACTOR,
                    self.radius_y + coord[1][frame_number] * self.SCALE_FACTOR
                ]

        self.update()

    def clear_trails(self):
        """Clear all particle trails"""
        self.trails = [[] for _ in range(len(self.coords))]
        self.update()

    def paintEvent(self, event):
        """Paint the simulation canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), self.BACKGROUND_COLOR)

        # Draw cell boundary
        painter.setPen(QPen(self.CELL_COLOR, 2))
        cell_diameter = sim.CELL_RADIUS * self.SCALE_FACTOR * 2
        painter.drawEllipse(
            self.radius_x - sim.CELL_RADIUS * self.SCALE_FACTOR,
            self.radius_y - sim.CELL_RADIUS * self.SCALE_FACTOR,
            cell_diameter,
            cell_diameter
        )

        # Draw nucleus boundary
        painter.setPen(QPen(self.NUCLEUS_COLOR, 2))
        nucleus_diameter = sim.NUCLEUS_RADIUS * self.SCALE_FACTOR * 2
        painter.drawEllipse(
            self.radius_x - sim.NUCLEUS_RADIUS * self.SCALE_FACTOR,
            self.radius_y - sim.NUCLEUS_RADIUS * self.SCALE_FACTOR,
            nucleus_diameter,
            nucleus_diameter
        )

        # Draw trails if enabled
        if self.show_trails:
            painter.setPen(QPen(self.TRAIL_COLOR, 1))
            for particle_trails in self.trails:
                for trail_segment in particle_trails:
                    painter.drawLine(*trail_segment)

        # Draw particles
        particle_width = self.particle_pixmap.width()
        particle_height = self.particle_pixmap.height()

        for pos in self.particle_positions:
            if len(pos) >= 2:
                painter.drawPixmap(
                    int(pos[0] - particle_width // 2), 
                    int(pos[1] - particle_height // 2), 
                    self.particle_pixmap
                )


class SimulationVis(QMainWindow):
    """Main simulation visualization window"""

    # UI Constants
    WINDOW_TITLE = "VSV Particle Simulation Visualizer"
    DEFAULT_FONT_SIZE = 12
    TIME_FONT_SIZE = 16
    ANIMATION_INTERVAL_MS = 10

    def __init__(self, config: sim.SimulationConfig):
        super().__init__()

        self.config = config
        self.state = AnimationState.STOPPED

        # Animation state
        self.coords: List[List[List[float]]] = []
        self.max_n_steps = 0
        self.frame_number = 0
        self.simulation_ready = False
        self.slider_being_dragged = False
        self.trails_state_before_recording = True
        self.default_status_text = "Preparing simulation..."

        # Create output directory
        self._ensure_output_directory()

        # Initialize UI components
        self._init_ui()
        self._setup_timer()
        self._setup_hover_hints()
        self._setup_keyboard_shortcuts()

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        try:
            os.makedirs(self.config.dirname, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create directory {self.config.dirname}: {e}")

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        shortcuts = [
            (Qt.Key_Space, self.toggle_play, "Spacebar"),
            ("R", self.reset_animation, "R"),
            ("T", self.toggle_trails, "T"),
            (Qt.Key_Left, self.step_backward, "Left Arrow"),
            (Qt.Key_Right, self.step_forward, "Right Arrow"),
        ]

        for key, method, name in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(method)

    def step_backward(self):
        """Step one frame backward"""
        if not self._can_navigate():
            return
        if self.frame_number > 0:
            self._goto_frame(self.frame_number - 1)

    def step_forward(self):
        """Step one frame forward"""
        if not self._can_navigate():
            return
        if self.frame_number < self.max_n_steps - 1:
            self._goto_frame(self.frame_number + 1)

    def _can_navigate(self) -> bool:
        """Check if navigation is allowed"""
        return self.simulation_ready and self.state != AnimationState.RECORDING

    def _goto_frame(self, frame_number: int):
        """Navigate to specific frame"""
        self.frame_number = frame_number
        self.canvas.update_frame(self.frame_number)
        self.time_label.setText(f"Time: {self.frame_number * self.config.dt:7.2f} s")
        self.frame_slider.setValue(self.frame_number)

    def _setup_hover_hints(self):
        """Setup hover hints for all controls"""
        widgets_and_hints = [
            (self.play_button, "Start/stop automatic animation playback (Spacebar)"),
            (self.reset_button, "Reset animation to beginning and clear all trails (R)"),
            (self.trails_button, "Toggle visibility of particle movement trails (T)"),
            (self.export_button, "Export particle coordinates to CSV file"),
            (self.record_button, "Record animation frames as PNG files for video creation"),
            (self.frame_slider, "Drag to navigate to any frame in the animation (Left/Right arrows)"),
        ]

        self.hints = {}
        for widget, hint in widgets_and_hints:
            widget.installEventFilter(self)
            self.hints[widget] = hint

    def eventFilter(self, obj, event):
        """Handle hover events for controls"""
        from PySide6.QtCore import QEvent

        if obj in self.hints:
            if event.type() == QEvent.Enter:
                self._show_hint(self.hints[obj])
                return True
            elif event.type() == QEvent.Leave:
                self._clear_hint()
                return True

        return super().eventFilter(obj, event)

    def _show_hint(self, text: str):
        """Show hint text in status bar"""
        self.status_label.setText(text)

    def _clear_hint(self):
        """Clear hint and restore default status text"""
        self.status_label.setText(self.default_status_text)

    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setFixedSize(self.config.width, self.config.height + 200)

        # Create main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Add components
        self._create_canvas(layout)
        self._create_time_display(layout)
        self._add_separator(layout)
        self._create_control_buttons(layout)
        self._add_separator(layout)
        self._create_frame_slider(layout)
        self._add_separator(layout)
        self._create_status_bar(layout)

    def _create_canvas(self, layout: QVBoxLayout):
        """Create simulation canvas"""
        self.canvas = SimulationCanvas(self.config.width, self.config.height)
        layout.addWidget(self.canvas, 0, Qt.AlignCenter)

    def _create_time_display(self, layout: QVBoxLayout):
        """Create time display label"""
        self.time_label = QLabel("Time: 0.00 s")
        self.time_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(self.TIME_FONT_SIZE)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("padding: 10px;")
        layout.addWidget(self.time_label)

    def _add_separator(self, layout: QVBoxLayout):
        """Add horizontal separator line"""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #666666;")
        layout.addWidget(separator)

    def _create_control_buttons(self, layout: QVBoxLayout):
        """Create control button layout"""
        button_layout = QHBoxLayout()

        # Create buttons
        buttons = [
            ("Play", self.toggle_play, "play_button"),
            ("Reset", self.reset_animation, "reset_button"),
            ("Hide Trails", self.toggle_trails, "trails_button"),
            ("Export to CSV", self.export_to_csv, "export_button"),
            ("Record Frames", self.toggle_recording, "record_button"),
        ]

        for text, method, attr_name in buttons:
            button = QPushButton(text)
            button.setFont(QFont("Arial", self.DEFAULT_FONT_SIZE))
            button.clicked.connect(method)
            button.setEnabled(False)  # Disabled until simulation ready
            button_layout.addWidget(button)
            setattr(self, attr_name, button)

        layout.addLayout(button_layout)

    def _create_frame_slider(self, layout: QVBoxLayout):
        """Create frame control slider"""
        slider_layout = QVBoxLayout()

        # Label
        slider_label = QLabel("Frame Control:")
        slider_label.setAlignment(Qt.AlignLeft)
        slider_label.setStyleSheet("padding: 5px;")
        slider_layout.addWidget(slider_label)

        # Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)

        # Connect signals
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)

        # Styling
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d4d4d4, stop:1 #afafaf);
            }
        """)

        slider_layout.addWidget(self.frame_slider)
        layout.addLayout(slider_layout)

    def _create_status_bar(self, layout: QVBoxLayout):
        """Create status label"""
        self.status_label = QLabel("Preparing simulation...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 5px;")
        layout.addWidget(self.status_label)

    def _setup_timer(self):
        """Setup animation timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.setInterval(self.ANIMATION_INTERVAL_MS)

    def run_simulation(self):
        """Start the simulation calculation in a separate thread"""
        self._update_status("Running simulation calculations...")
        
        self.worker = SimulationWorker(self.config)
        self.worker.simulation_complete.connect(self._on_simulation_complete)
        self.worker.error_occurred.connect(self._on_simulation_error)
        self.worker.start()

    def _on_simulation_complete(self, coords: List[List[List[float]]], max_n_steps: int):
        """Handle simulation completion"""
        self.coords = coords
        self.max_n_steps = max_n_steps
        self.canvas.set_coords(self.coords)

        # Enable controls
        for button in [self.play_button, self.reset_button, self.trails_button, 
                      self.export_button, self.record_button]:
            button.setEnabled(True)

        self.frame_slider.setEnabled(True)
        self.frame_slider.setMaximum(max_n_steps - 1)
        self.simulation_ready = True

        # Initialize display
        self._update_status("Simulation ready - Press 'Play' or use controls")
        self._goto_frame(0)

    def _on_simulation_error(self, error_message: str):
        """Handle simulation errors"""
        QMessageBox.critical(self, "Simulation Error", error_message)
        self._update_status("Simulation failed - check console for details")

    def _update_status(self, message: str):
        """Update status message"""
        self.default_status_text = message
        self.status_label.setText(message)

    def toggle_play(self):
        """Toggle play/pause state"""
        if not self.simulation_ready:
            return

        if self.state in [AnimationState.STOPPED, AnimationState.PAUSED]:
            self._start_playback()
        else:
            self._pause_playback()

    def _start_playback(self):
        """Start animation playback"""
        if self.state != AnimationState.RECORDING:
            self.state = AnimationState.PLAYING
        self.play_button.setText("Pause")
        self.timer.start()
        self._update_status("Playing animation...")

    def _pause_playback(self):
        """Pause animation playback"""
        self.state = AnimationState.PAUSED
        self.play_button.setText("Play")
        self.timer.stop()
        self._update_status("Animation paused - Use controls to navigate")

    def toggle_trails(self):
        """Toggle trail visibility"""
        if not self.simulation_ready or self.state == AnimationState.RECORDING:
            return

        new_state = not self.canvas.show_trails
        self.canvas.toggle_trails(new_state)
        self.trails_button.setText("Hide Trails" if new_state else "Show Trails")

    def toggle_recording(self):
        """Toggle frame recording mode"""
        if not self.simulation_ready:
            return

        if self.state != AnimationState.RECORDING:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Start frame recording mode"""
        # Confirm with user
        reply = QMessageBox.question(
            self, "Record Frames", 
            f"Record all {self.max_n_steps} frames as PNG files?\n"
            f"Directory: {self.config.dirname}/",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.state = AnimationState.RECORDING
        self.config.record_frames = True

        # Save current state and setup for recording
        self.trails_state_before_recording = self.canvas.show_trails
        if self.canvas.show_trails:
            self.canvas.toggle_trails(False)
            self.trails_button.setText("Show Trails")

        # Disable controls
        for button in [self.play_button, self.reset_button, self.trails_button, 
                      self.export_button, self.frame_slider]:
            button.setEnabled(False)

        self.record_button.setText("Stop Recording")
        self._update_status("Recording frames - Animation playing automatically")

        # Reset and start
        self._goto_frame(0)
        self.canvas.clear_trails()
        self._start_playback()

        print(f"Started recording frames to {self.config.dirname}/")

    def _stop_recording(self):
        """Stop frame recording mode"""
        self.state = AnimationState.STOPPED
        self.config.record_frames = False

        # Stop playback
        self.timer.stop()
        self.play_button.setText("Play")

        # Restore trails state
        if self.trails_state_before_recording != self.canvas.show_trails:
            self.canvas.toggle_trails(self.trails_state_before_recording)
            self.trails_button.setText("Hide Trails" if self.trails_state_before_recording else "Show Trails")

        # Re-enable controls
        for button in [self.play_button, self.reset_button, self.trails_button, 
                      self.export_button, self.frame_slider]:
            button.setEnabled(True)

        self.record_button.setText("Record Frames")
        self._update_status(f"Recording complete - Frames saved to {self.config.dirname}/")

        QMessageBox.information(self, "Recording Complete", 
                              f"Frames saved to: {self.config.dirname}/")
        print(f"Recording complete - Frames saved to {self.config.dirname}/")

    def _on_slider_pressed(self):
        """Handle slider press"""
        self.slider_being_dragged = True
        if self.state == AnimationState.PLAYING:
            self.timer.stop()

    def _on_slider_released(self):
        """Handle slider release"""
        self.slider_being_dragged = False
        if self.state == AnimationState.PLAYING:
            self.timer.start()

    def _on_slider_value_changed(self, value: int):
        """Handle slider value changes"""
        if not self.simulation_ready or value == self.frame_number:
            return

        self._goto_frame(value)

        if self.config.record_frames:
            self._save_frame()

    def _update_frame(self):
        """Update animation frame (timer callback)"""
        if self.frame_number >= self.max_n_steps - 1:
            self.timer.stop()

            if self.state == AnimationState.RECORDING:
                self._stop_recording()
            else:
                self.state = AnimationState.STOPPED
                self.play_button.setText("Play")
                self._update_status("Animation complete")
            return

        if not self.slider_being_dragged:
            self._goto_frame(self.frame_number + 1)

            # Update slider (avoid feedback loop)
            if self.state != AnimationState.RECORDING:
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.frame_number)
                self.frame_slider.blockSignals(False)

            if self.config.record_frames:
                self._save_frame()

    def _save_frame(self):
        """Save current frame as image"""
        pixmap = self.canvas.grab()
        filename = os.path.join(self.config.dirname, f"scene-{self.frame_number:03d}.png")
        pixmap.save(filename)

    def reset_animation(self):
        """Reset animation to beginning"""
        if not self.simulation_ready:
            return

        # Stop any playback
        self.timer.stop()
        self.state = AnimationState.STOPPED
        self.play_button.setText("Play")

        # Reset display
        self._goto_frame(0)
        self.canvas.clear_trails()
        self._update_status("Animation reset - Press 'Play' or use controls")

    def export_to_csv(self):
        """Export coordinates to CSV file"""
        if not self.simulation_ready:
            return

        filename = os.path.join(self.config.dirname, 'coords.csv')
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'particle', 'x', 'y'])
                
                for i, coord in enumerate(self.coords):
                    for j in range(len(coord[0])):
                        writer.writerow([j, i, coord[0][j], coord[1][j]])
            
            self._update_status(f"Exported to {filename}")
            QMessageBox.information(self, "Export Complete", f"Data exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")

    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()
