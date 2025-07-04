import os
import sys

import numpy as np
import csv
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, QFrame)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QFont

import simulation as sim


class SimulationWorker(QThread):
    """Worker thread for running the simulation calculations"""
    simulation_complete = Signal(list, list, int, list)
    
    def __init__(self, n_particles, pDriv, trap_dist, time_between, total_time):
        super().__init__()
        self.n_particles = n_particles
        self.pDriv = pDriv
        self.trap_dist = trap_dist
        self.time_between = time_between
        self.total_time = total_time
        
    def run(self):
        coords = []
        max_n_steps = 0
        i = 0
        n_exited = 0
        exit_times = []
        
        while i < self.n_particles:
            x_coords, y_coords, throw_out, exit_time, _, _, _, _ = sim.move(
                self.total_time,
                self.pDriv,
                self.trap_dist,
                self.time_between,
                theta=2*np.pi*i/self.n_particles
            )
            max_n_steps = max(len(x_coords), max_n_steps)
            if throw_out:
                print(f"throw out particle {i}")
                continue
            else:
                if exit_time != -1:
                    n_exited += 1
                    exit_times.append(exit_time)
                coords.append([x_coords, y_coords])
                print(f"particle {i} done")
                i += 1
        
        print("{} out of {} exit the cell".format(n_exited, self.n_particles))
        if exit_times:
            print("Avg exit time: {:.2e}".format(np.mean(exit_times)))
        
        self.simulation_complete.emit(coords, exit_times, max_n_steps, [])


class SimulationCanvas(QWidget):
    """Custom widget for drawing the simulation"""
    
    def __init__(self, width=600, height=600):
        super().__init__()
        self.width = width
        self.height = height
        self.radius_x = self.width / 2
        self.radius_y = self.height / 2
        self.scale_factor = 1e7
        
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")
        
        # Particle data
        self.coords = []
        self.particle_positions = []
        self.trails = []  # Store trail lines
        self.show_trails = True  # Toggle for showing trails
        
        # Create particle pixmap
        self.particle_pixmap = self.create_particle_pixmap()
        
    def create_particle_pixmap(self):
        """Load particle image from file"""
        try:
            pixmap = QPixmap("dot-2.png")
            if pixmap.isNull():
                raise Exception("Pixmap is null")
            return pixmap
        except Exception as e:
            print(f"Error loading dot-2.png: {e}, using fallback dot")
            # Fallback to programmatic dot
            pixmap = QPixmap(8, 8)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawEllipse(0, 0, 8, 8)
            painter.end()
            return pixmap
    
    def set_coords(self, coords):
        """Set particle coordinates"""
        self.coords = coords
        self.particle_positions = [[0] * len(coords) for _ in range(len(coords))]
        self.trails = [[] for _ in range(len(coords))]
        
    def toggle_trails(self, show):
        """Toggle trail visibility"""
        self.show_trails = show
        self.update()
        
    def reset_animation(self):
        """Reset animation to beginning"""
        self.particle_positions = [[0] * len(self.coords) for _ in range(len(self.coords))]
        self.trails = [[] for _ in range(len(self.coords))]
        self.update()
        
    def update_frame(self, frame_number):
        """Update particle positions for current frame"""
        if not self.coords:
            return
            
        for i, coord in enumerate(self.coords):
            if frame_number < len(coord[0]):
                # Store previous position for trail
                if frame_number > 0:
                    prev_x = self.radius_x + coord[0][frame_number-1] * self.scale_factor
                    prev_y = self.radius_y + coord[1][frame_number-1] * self.scale_factor
                    curr_x = self.radius_x + coord[0][frame_number] * self.scale_factor
                    curr_y = self.radius_y + coord[1][frame_number] * self.scale_factor
                    
                    # Add to trail
                    self.trails[i].append((prev_x, prev_y, curr_x, curr_y))
                
                self.particle_positions[i] = [
                    self.radius_x + coord[0][frame_number] * self.scale_factor,
                    self.radius_y + coord[1][frame_number] * self.scale_factor
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
        
        # Draw black background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Draw cell boundary
        painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow
        cell_diameter = sim.CELL_RADIUS * self.scale_factor * 2
        painter.drawEllipse(
            self.radius_x - sim.CELL_RADIUS * self.scale_factor,
            self.radius_y - sim.CELL_RADIUS * self.scale_factor,
            cell_diameter,
            cell_diameter
        )
        
        # Draw nucleus boundary
        painter.setPen(QPen(QColor(255, 255, 255), 2))  # White
        nucleus_diameter = sim.NUCLEUS_RADIUS * self.scale_factor * 2
        painter.drawEllipse(
            self.radius_x - sim.NUCLEUS_RADIUS * self.scale_factor,
            self.radius_y - sim.NUCLEUS_RADIUS * self.scale_factor,
            nucleus_diameter,
            nucleus_diameter
        )
        
        # Draw trails only if enabled
        if self.show_trails:
            painter.setPen(QPen(QColor(255, 255, 255), 1))  # White trails
            for particle_trails in self.trails:
                for trail_segment in particle_trails:
                    painter.drawLine(trail_segment[0], trail_segment[1], trail_segment[2], trail_segment[3])
        
        # Draw particles
        for pos in self.particle_positions:
            if len(pos) >= 2:
                # Center the particle image on the position
                particle_width = self.particle_pixmap.width()
                particle_height = self.particle_pixmap.height()
                painter.drawPixmap(
                    pos[0] - particle_width // 2, 
                    pos[1] - particle_height // 2, 
                    self.particle_pixmap
                )


class Simulation(QMainWindow):
    def __init__(
        self,
        total_time,
        pDriv,
        trap_dist,
        time_between,
        n_particles=1,
        dt=0.01,
        dirname="sim",
        width=600,
        height=600,
        write_to_ps=False,
    ):
        super().__init__()
        
        # Simulation parameters
        self.total_time = total_time
        self.n_particles = n_particles
        self.pDriv = pDriv
        self.trap_dist = trap_dist
        self.time_between = time_between
        self.dt = dt
        self.dirname = dirname
        self.write_to_ps = write_to_ps
        
        # Animation state
        self.coords = []
        self.max_n_steps = 0
        self.frame_number = 0
        self.is_playing = False
        self.simulation_ready = False
        self.slider_being_dragged = False
        self.is_recording = False
        self.trails_state_before_recording = True
        self.default_status_text = "Preparing simulation..."
        
        # Create directory for output
        try:
            os.makedirs(self.dirname, exist_ok=True)
        except:
            pass
        
        self.init_ui(width, height)
        self.setup_timer()
        self.setup_hover_hints()
        
    def setup_hover_hints(self):
        """Setup hover hints for all controls"""
        # Install event filters for buttons
        self.play_button.installEventFilter(self)
        self.reset_button.installEventFilter(self)
        self.trails_button.installEventFilter(self)
        self.export_button.installEventFilter(self)
        self.record_button.installEventFilter(self)
        self.frame_slider.installEventFilter(self)
        
        # Store hint messages
        self.hints = {
            self.play_button: "Start/stop automatic animation playback",
            self.reset_button: "Reset animation to beginning and clear all trails",
            self.trails_button: "Toggle visibility of particle movement trails",
            self.export_button: "Export particle coordinates to CSV file",
            self.record_button: "Record animation frames as PNG files for video creation",
            self.frame_slider: "Drag to navigate to any frame in the animation"
        }
    
    def eventFilter(self, obj, event):
        """Handle hover events for controls"""
        from PySide6.QtCore import QEvent
        
        if obj in self.hints:
            if event.type() == QEvent.Enter:
                self.show_hint(self.hints[obj])
                return True
            elif event.type() == QEvent.Leave:
                self.clear_hint()
                return True
        
        return super().eventFilter(obj, event)
        
    def show_hint(self, text):
        """Show hint text in status bar"""
        self.status_label.setText(text)
        
    def clear_hint(self):
        """Clear hint and restore default status text"""
        self.status_label.setText(self.default_status_text)
        
    def init_ui(self, width, height):
        """Initialize the user interface"""
        self.setWindowTitle("Particle Simulation")
        self.setFixedSize(width, height + 200)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        layout.setSpacing(5)  # Set consistent spacing between elements
        
        # Canvas
        self.canvas = SimulationCanvas(width, height)
        layout.addWidget(self.canvas, 0, Qt.AlignCenter)  # Center the canvas
        
        # Time label
        self.time_label = QLabel("Time: 0.00 s")
        self.time_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("padding: 10px;")
        layout.addWidget(self.time_label)
        
        # Separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("color: #666666;")
        layout.addWidget(separator1)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.setFont(QFont("Arial", 12))
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)  # Disabled until simulation is ready
        button_layout.addWidget(self.play_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFont(QFont("Arial", 12))
        self.reset_button.clicked.connect(self.reset_animation)
        self.reset_button.setEnabled(False)  # Disabled until simulation is ready
        button_layout.addWidget(self.reset_button)
        
        # Trails toggle button
        self.trails_button = QPushButton("Hide Trails")
        self.trails_button.setFont(QFont("Arial", 12))
        self.trails_button.clicked.connect(self.toggle_trails)
        self.trails_button.setEnabled(False)  # Disabled until simulation is ready
        button_layout.addWidget(self.trails_button)
        
        # Export button
        self.export_button = QPushButton("Export to CSV")
        self.export_button.setFont(QFont("Arial", 12))
        self.export_button.clicked.connect(self.export_to_csv)
        self.export_button.setEnabled(False)  # Disabled until simulation is ready
        button_layout.addWidget(self.export_button)
        
        # Record frames button
        self.record_button = QPushButton("Record Frames")
        self.record_button.setFont(QFont("Arial", 12))
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)  # Disabled until simulation is ready
        button_layout.addWidget(self.record_button)
        
        layout.addLayout(button_layout)
        
        # Separator line
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("color: #666666;")
        layout.addWidget(separator2)
        
        # Frame control slider
        slider_layout = QVBoxLayout()
        
        slider_label = QLabel("Frame Control:")
        slider_label.setAlignment(Qt.AlignLeft)
        slider_label.setStyleSheet("padding: 5px;")
        slider_layout.addWidget(slider_label)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Will be updated when simulation is ready
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)  # Disabled until simulation is ready
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)
        self.frame_slider.sliderPressed.connect(self.on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_slider_released)
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
        
        # Separator line
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        separator3.setStyleSheet("color: #666666;")
        layout.addWidget(separator3)
        
        # Status label
        self.status_label = QLabel("Preparing simulation...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("padding: 5px;")
        layout.addWidget(self.status_label)
        
    def setup_timer(self):
        """Setup animation timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(int(self.dt * 1000))  # Convert to milliseconds
        
    def run_simulation(self):
        """Start the simulation calculation in a separate thread"""
        self.status_label.setText("Running simulation calculations...")
        self.worker = SimulationWorker(
            self.n_particles, self.pDriv, self.trap_dist, 
            self.time_between, self.total_time
        )
        self.worker.simulation_complete.connect(self.on_simulation_complete)
        self.worker.start()
        
    def on_simulation_complete(self, coords, exit_times, max_n_steps, vsv):
        """Handle simulation completion"""
        # Convert coordinates to screen coordinates
        self.coords = []
        for coord in coords:
            x_coords = [x for x in coord[0]]
            y_coords = [y for y in coord[1]]
            self.coords.append([x_coords, y_coords])
        
        self.max_n_steps = max_n_steps
        self.canvas.set_coords(self.coords)
        
        # Enable controls
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.trails_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.frame_slider.setEnabled(True)
        self.frame_slider.setMaximum(max_n_steps - 1)
        self.simulation_ready = True
        
        # Update status
        self.default_status_text = "Simulation ready - Press 'Play' or use slider to control"
        self.status_label.setText(self.default_status_text)
        
        # Initialize first frame
        self.frame_number = 0
        self.canvas.update_frame(self.frame_number)
        self.frame_slider.setValue(0)
        
    def toggle_play(self):
        """Toggle play/pause state"""
        if not self.simulation_ready:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText("Pause")
            self.timer.start()
            self.default_status_text = "Playing animation..."
            self.status_label.setText(self.default_status_text)
        else:
            self.play_button.setText("Play")
            self.timer.stop()
            self.default_status_text = "Animation paused - Use slider to navigate"
            self.status_label.setText(self.default_status_text)
    
    def toggle_trails(self):
        """Toggle trail visibility"""
        if not self.simulation_ready or self.is_recording:
            return
            
        current_state = self.canvas.show_trails
        new_state = not current_state
        
        self.canvas.toggle_trails(new_state)
        
        # Update button text
        if new_state:
            self.trails_button.setText("Hide Trails")
        else:
            self.trails_button.setText("Show Trails")
    
    def toggle_recording(self):
        """Toggle frame recording mode"""
        if not self.simulation_ready:
            return
            
        if not self.is_recording:
            # Start recording
            self.start_recording()
        else:
            # Stop recording
            self.stop_recording()
    
    def start_recording(self):
        """Start frame recording mode"""
        self.is_recording = True
        self.write_to_ps = True
        
        # Store current trails state and hide trails
        self.trails_state_before_recording = self.canvas.show_trails
        if self.canvas.show_trails:
            self.canvas.toggle_trails(False)
            self.trails_button.setText("Show Trails")
        
        # Disable all other controls
        self.play_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.trails_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.frame_slider.setEnabled(False)
        
        # Update button text and status
        self.record_button.setText("Stop Recording")
        self.default_status_text = "Recording frames - Animation will play automatically"
        self.status_label.setText(self.default_status_text)
        
        # Reset to beginning and start playing
        self.frame_number = 0
        self.canvas.reset_animation()
        self.canvas.clear_trails()
        self.time_label.setText("Time: 0.00 s")
        self.frame_slider.setValue(0)
        
        # Start playing automatically
        self.is_playing = True
        self.play_button.setText("Pause")
        self.timer.start()
        
        print(f"Started recording frames to {self.dirname}/")
    
    def stop_recording(self):
        """Stop frame recording mode"""
        self.is_recording = False
        self.write_to_ps = False
        
        # Stop animation
        self.timer.stop()
        self.is_playing = False
        self.play_button.setText("Play")
        
        # Restore trails state
        if self.trails_state_before_recording != self.canvas.show_trails:
            self.canvas.toggle_trails(self.trails_state_before_recording)
            if self.trails_state_before_recording:
                self.trails_button.setText("Hide Trails")
            else:
                self.trails_button.setText("Show Trails")
        
        # Re-enable all controls
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.trails_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.frame_slider.setEnabled(True)
        
        # Update button text and status
        self.record_button.setText("Record Frames")
        self.default_status_text = f"Recording complete - Frames saved to {self.dirname}/"
        self.status_label.setText(self.default_status_text)
        
        print(f"Recording complete - Frames saved to {self.dirname}/")
    
    def on_slider_pressed(self):
        """Handle when user starts dragging the slider"""
        self.slider_being_dragged = True
        if self.is_playing:
            self.timer.stop()  # Pause animation while dragging
    
    def on_slider_released(self):
        """Handle when user stops dragging the slider"""
        self.slider_being_dragged = False
        if self.is_playing:
            self.timer.start()  # Resume animation if auto-play is enabled
    
    def on_slider_value_changed(self, value):
        """Handle slider value changes (manual frame control)"""
        if not self.simulation_ready or value == self.frame_number:
            return
        
        # Update to the frame specified by slider
        self.frame_number = value
        self.canvas.update_frame(self.frame_number)
        self.time_label.setText(f"Time: {self.frame_number * self.dt:7.2f} s")
        
        # Save frame if requested
        if self.write_to_ps:
            self.save_frame()
            
    def update_frame(self):
        """Update animation frame (called by timer during auto-play)"""
        if self.frame_number >= self.max_n_steps - 1:
            self.timer.stop()
            self.is_playing = False
            self.play_button.setText("Play")
            
            # If we were recording, stop recording automatically
            if self.is_recording:
                self.stop_recording()
            else:
                self.default_status_text = "Animation complete"
                self.status_label.setText(self.default_status_text)
            return
        
        if not self.slider_being_dragged:
            self.frame_number += 1
            self.canvas.update_frame(self.frame_number)
            self.time_label.setText(f"Time: {self.frame_number * self.dt:7.2f} s")
            
            # Update slider position (but only if user isn't dragging it and not recording)
            if not self.is_recording:
                self.frame_slider.blockSignals(True)  # Prevent recursive calls
                self.frame_slider.setValue(self.frame_number)
                self.frame_slider.blockSignals(False)
            
            # Save frame if requested
            if self.write_to_ps:
                self.save_frame()
        
    def save_frame(self):
        """Save current frame as image"""
        pixmap = self.canvas.grab()
        filename = os.path.join(self.dirname, f"scene-{self.frame_number:03d}.png")
        pixmap.save(filename)
        
    def reset_animation(self):
        """Reset animation to beginning"""
        if not self.simulation_ready:
            return
            
        self.timer.stop()
        self.is_playing = False
        self.play_button.setText("Play")
        self.frame_number = 0
        self.canvas.reset_animation()
        self.canvas.clear_trails()
        self.time_label.setText("Time: 0.00 s")
        self.frame_slider.setValue(0)
        self.default_status_text = "Animation reset - Press 'Play' or use slider"
        self.status_label.setText(self.default_status_text)
        
    def export_to_csv(self):
        """Export coordinates to CSV file"""
        if not self.simulation_ready:
            return
            
        filename = os.path.join(self.dirname, 'coords.csv')
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'particle', 'x', 'y'])  # Header
            for i, coord in enumerate(self.coords):
                for j in range(len(coord[0])):
                    writer.writerow([j, i, coord[0][j], coord[1][j]])
        
        self.default_status_text = f"Exported to {filename}"
        self.status_label.setText(self.default_status_text)
        
    def closeEvent(self, event):
        """Handle window close event"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Create and show simulation
    simulation = Simulation(
        total_time=2000,
        n_particles=1,
        pDriv=0.03,
        trap_dist=sim.TRAP_DIST,
        time_between=sim.TIME_BETWEEN_STATES,
        dt=0.01,
        dirname="sim",
        width=600,
        height=600,
        write_to_ps=False,
    )
    
    simulation.show()
    
    # Start simulation calculations
    simulation.run_simulation()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()