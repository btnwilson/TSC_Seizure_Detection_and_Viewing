import sys
import numpy as np
import os

# PyQt6 Imports
from PyQt6.QtCore import QTimer, QDateTime, Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDateTimeEdit, \
    QComboBox, QPushButton, QGridLayout, QProgressBar, QDial, QSlider, QTextEdit, QMainWindow, QSpacerItem, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
# Matplotlib Imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import pickle
from datetime import datetime, timedelta
import cv2
import time

class MatplotlibWidget(FigureCanvas):
    def __init__(self):#, width=11, height=3, dpi=100):
        self.fig = plt.Figure(figsize=(11, 3), dpi=100)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("EEG Trace", fontsize=10)
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
        self.ax.set_xlabel("Time (seconds)", fontsize=9)
        self.ax.set_ylabel("Normalized EEG", fontsize=9)

        super(MatplotlibWidget, self).__init__(self.fig)

    def updateplot(self, x, y):  # Renamed from plot to updateplot
        self.ax.clear()  # Clear previous plot
        self.ax.plot(x, y)  # Plot new data
        self.ax.set_title("EEG Trace")
        self.draw()  # Use self.draw() instead of self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Structure of the window
        self.setWindowTitle("TSC Candidate Seizure Event UI")

        # Main widget and layout
        central_widget = QWidget(self)
        main_layout = QVBoxLayout(central_widget)

        # Row 1: Dropdowns with descriptions
        row1_layout = QHBoxLayout()
        self.mice_dropdown = QComboBox()
        self.mice_dropdown.setMinimumWidth(200)
        self.file_dropdown = QComboBox()
        self.file_dropdown.setMinimumWidth(200)
        self.video_dropdown = QComboBox()
        self.video_dropdown.setMinimumWidth(200)

        # Labels for dropdowns
        label1 = QLabel("Selected Mouse")
        label2 = QLabel("Selected File")
        label3 = QLabel("Selected Video")

        # Add the dropdowns and labels to row1_layout
        row1_layout.addWidget(label1)
        row1_layout.addWidget(self.mice_dropdown)
        row1_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row1_layout.addWidget(label2)
        row1_layout.addWidget(self.file_dropdown)
        row1_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row1_layout.addWidget(label3)
        row1_layout.addWidget(self.video_dropdown)

        # Row 2: Video window placeholder
        self.video_widget = QLabel(self)  # Placeholder for video window
        self.video_widget.setStyleSheet("background-color: black;")  # Black background for video
        self.video_widget.setMinimumHeight(500)
        self.video_widget.setMinimumWidth(500)
        self.video_widget.setScaledContents(True)

        # Row 3: Video control with 5 buttons and a slider
        row3_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play/Pause")
        self.previous_button = QPushButton("Previous Event")
        self.next_button = QPushButton("Next Event")
        self.jump_back_button = QPushButton("Jump -5s")
        self.jump_forward_button = QPushButton("Jump +15s")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 6)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setMaximumWidth(200)
        self.slider.setValue(2)

        # Label for slider
        slider_label = QLabel("Playback Speed")

        # Add buttons and spacer between them
        row3_layout.addWidget(self.play_pause_button)
        row3_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row3_layout.addWidget(self.previous_button)
        row3_layout.addWidget(self.next_button)
        row3_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row3_layout.addWidget(self.jump_back_button)
        row3_layout.addWidget(self.jump_forward_button)
        row3_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row3_layout.addWidget(slider_label)
        row3_layout.addWidget(self.slider)

        # Row 4: Matplotlib plot
        self.plot_widget = MatplotlibWidget()
        self.plot_widget.setMinimumHeight(200)
        self.plot_widget.setMinimumWidth(500)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.plot_widget.updateplot(x, y)

        # Row 5: Text box at the bottom (non-editable)
        self.text_box = QTextEdit(self)
        self.text_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_box.setFixedHeight(50)
        self.text_box.setPlainText("Event Information Will Be Displayed Here")
        self.text_box.setReadOnly(True)  # Make the text box non-editable

        # Adding all rows to main layout
        main_layout.addLayout(row1_layout)
        main_layout.addWidget(self.video_widget)
        main_layout.addLayout(row3_layout)
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.text_box)

        # Set central widget
        self.setCentralWidget(central_widget)
        # ---------------------------------------------------------

        #initialization of video playing
        data_location = "C:/Users/bentn/OneDrive/Desktop/Work/New data"
        os.chdir(data_location)
        self.mice_dropdown.addItems(os.listdir("Events/"))
        selected_mouse = self.mice_dropdown.currentText()
        selected_mouse = "f87tsc1l3_180"
        self.file_dropdown.addItems(os.listdir(os.path.join("Events", selected_mouse)))
        selected_file = self.file_dropdown.currentText()
        selected_file = "110623_174017_16613.pkl"
        self.pickle_location = os.path.join("Events", selected_mouse, selected_file)
        print(self.pickle_location)
        self.data = None
        self.open_pickle()

        self.video_path = "D:/TSC Cage Videos"
        self.all_videos = os.listdir(self.video_path)
        self.video_start_times = [datetime.strptime(name.replace(".mp4", "").replace(".mkv", ""), "%Y-%m-%d %H-%M-%S") for name in self.all_videos]


        self.current_event_index = 0
        self.video_fs = 30
        self.playback_speed = self.get_playback_speed()
        self.milliseconds_per_frame = int(1000 /(self.video_fs * self.playback_speed))
        self.isplaying = False
        self.cap = None
        self.start_fram_index = None
        self.last_frame = None

        sorted_indices = sorted(range(len(self.video_start_times)), key=lambda i: self.video_start_times[i])
        self.sorted_video_start_times = np.array([self.video_start_times[i] for i in sorted_indices])
        self.sorted_video_names = np.array([self.all_videos[i] for i in sorted_indices])

        self.eeg_data = self.data[f"Event {self.current_event_index}"]["EEG Signal"]
        self.event_start_time = datetime.strptime(self.data[f"Event {self.current_event_index}"]["Start Time"], "%m-%d-%Y %H:%M:%S")

        self.max_spikes = self.data[f"Event {self.current_event_index}"]["Max Spikes"]
        self.max_arclength = self.data[f"Event {self.current_event_index}"]["Max Arclength"]
        self.event_duration = self.data[f"Event {self.current_event_index}"]["Duration"]
        self.cage_num = self.data[f"Event {self.current_event_index}"]["Cage Number"]
        self.text_box.setPlainText(f"Max Spikes: {self.max_spikes} \t Max Linelength: {self.max_arclength} \t Event Duration: {self.event_duration}s \t Cage Number: {self.cage_num}")
        self.plot_widget.updateplot(np.arange(0,len(self.eeg_data))/256, self.eeg_data)

        self.selected_video = None
        self.selected_video_start_time = None
        self.identify_videos()

        self.cap = cv2.VideoCapture(os.path.join(self.video_path, self.selected_video))
        time_difference = self.event_start_time - self.selected_video_start_time
        time_from_start = time_difference.total_seconds()
        start_frame_index = int(time_from_start * 30 - 5 * 30)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)


        # button functionality
        self.mice_dropdown.currentIndexChanged.connect(self.mouse_changed)
        self.file_dropdown.currentIndexChanged.connect(self.file_changed)
        self.video_dropdown.currentIndexChanged.connect(self.video_changed)
        self.play_pause_button.clicked.connect(self.play_pause)
        self.previous_button.clicked.connect(self.go_to_previous_event)
        self.next_button.clicked.connect(self.go_to_next_event)
        self.jump_back_button.clicked.connect(self.jump_back)
        self.jump_forward_button.clicked.connect(self.jump_forward)
        self.slider.valueChanged.connect(self.change_playback_speed)

        self.video_timer = QTimer()
        self.video_timer.setInterval(self.milliseconds_per_frame)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_timer.start()
    def update_video_frame(self):
        if self.isplaying:
            # Read a frame from the video capture object

            ret, frame = self.cap.read()
            if ret:
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert the frame to QImage
                height, width, channels = frame_rgb.shape
                bytes_per_line = 3 * width
                qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

                # Convert QImage to QPixmap
                pixmap = QPixmap.fromImage(qimage)

                # Set the pixmap to the video widget
                self.video_widget.setPixmap(pixmap)

    def change_video(self):
        if self.cap is not None:
            self.cap.release()

        self.eeg_data = self.data[f"Event {self.current_event_index}"]["EEG Signal"]
        self.event_start_time = datetime.strptime(self.data[f"Event {self.current_event_index}"]["Start Time"], "%m-%d-%Y %H:%M:%S")

        self.cap = cv2.VideoCapture(os.path.join(self.video_path, self.selected_video))

        if not self.cap.isOpened():
            print(f"Error: Unable to open video file {self.selected_video}")
            return
        try:
            ret, frame = self.cap.read()

            time_difference = self.event_start_time - self.selected_video_start_time
            time_from_start = time_difference.total_seconds()
            start_frame_index = int(time_from_start * 30 - 5 * 30)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

            self.max_spikes = self.data[f"Event {self.current_event_index}"]["Max Spikes"]
            self.max_arclength = self.data[f"Event {self.current_event_index}"]["Max Arclength"]
            self.event_duration = self.data[f"Event {self.current_event_index}"]["Duration"]
            self.cage_num = self.data[f"Event {self.current_event_index}"]["Cage Number"]

            self.text_box.setPlainText(f"Max Spikes: {self.max_spikes} \t Max Linelength: {self.max_arclength} \t Event Duration: {self.event_duration}s \t Cage Number: {self.cage_num}")

            self.plot_widget.updateplot(np.arange(0, len(self.eeg_data))/256, self.eeg_data)

            self.video_timer.start()
        except Exception as e:
            print(f"Error in change video: {e}")
    def change_playback_speed(self):
        self.video_timer.stop()
        self.playback_speed = self.get_playback_speed()
        self.milliseconds_per_frame = int(1000 /(self.video_fs * self.playback_speed))
        self.video_timer.setInterval(self.milliseconds_per_frame)
        print("Playback speed changed")
        self.video_timer.start()

    def jump_forward(self):
        if self.cap is not None:
            try:
                self.video_timer.stop()
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame + 15 * 30))
                self.video_timer.start()
            except Exception as e:
                print(f"Error in jump_backward: {e}")

    def jump_back(self):
        if self.cap is not None:
            try:
                self.video_timer.stop()
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 5 * 30))
                self.video_timer.start()
            except Exception as e:
                print(f"Error in jump_backward: {e}")

    def go_to_next_event(self):
        if self.current_event_index < len(self.data) - 1:
            self.current_event_index += 1
            QTimer.singleShot(500, self.process_next_event)

        else:
            self.text_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.text_box.setPlainText("All Events Viewed")

    def process_next_event(self):
        try:
            self.video_timer.stop()
            # This logic will run after the delay
            self.identify_videos()

            self.change_video()
        except Exception as e:
            print(f"Error in process_next_event: {e}")

    def play_pause(self):
        if self.isplaying:
            self.isplaying = False
            self.video_timer.stop()
        else:
            self.isplaying = True
            self.video_timer.start()

    def go_to_previous_event(self):
        if self.current_event_index > 0:
            self.current_event_index -= 1
            QTimer.singleShot(500, self.process_next_event)

        else:
            self.text_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.text_box.setPlainText("No Previous Events")
        print("Going to previous event")

    def mouse_changed(self):
        print("option changed")
        self.mice_dropdown.blockSignals(True)
        selected_mouse = self.mice_dropdown.currentText()
        print(selected_mouse)
        self.file_dropdown.clear()
        files = os.listdir(os.path.join("Events", selected_mouse))
        print(files)
        self.file_dropdown.addItems(files)
        print("dropdown updated")
        selected_file = self.file_dropdown.currentText()
        self.pickle_location = os.path.join("Events", selected_mouse, selected_file)
        print(self.pickle_location)
        self.open_pickle()
        self.change_video()
        print("Mouse changed")
        self.mice_dropdown.blockSignals(False)

    def file_changed(self):
        selected_mouse = self.mice_dropdown.currentText()
        selected_file = self.file_dropdown.currentText()
        self.pickle_location = os.path.join("Events", selected_mouse, selected_file)
        print(self.pickle_location)
        self.open_pickle()
        self.change_video()
        print("File changed")

    def video_changed(self):
        self.selected_video = self.video_dropdown.currentText()
        self.selected_video_start_time = datetime.strptime(self.selected_video.replace(".mp4", "").replace(".mkv", ""),"%Y-%m-%d %H-%M-%S")
        self.change_video()
        print("Video changed")

    def open_pickle(self):
        with open(self.pickle_location, 'rb') as f:
            self.data = pickle.load(f)

    def get_playback_speed(self):
        slider_value = self.slider.value()
        return slider_value * .25 + .5

    def identify_videos(self):
        #self.video_dropdown.clear()
        selected_videos = []
        selected_video_start_time = None
        time_window = timedelta(minutes=60)
        for video_number in range(len(self.sorted_video_start_times) - 1):
            if self.sorted_video_start_times[video_number] <= self.event_start_time <= self.sorted_video_start_times[video_number + 1]:
                potential_videos = (self.sorted_video_start_times >= self.sorted_video_start_times[video_number] - time_window) & (self.sorted_video_start_times <= self.sorted_video_start_times[video_number])
                if np.sum(potential_videos) > 0:
                    selected_videos = list(self.sorted_video_names[potential_videos])
                else:
                    selected_videos = list(self.sorted_video_names[video_number])
                break

        self.video_dropdown.addItems(selected_videos)
        self.selected_video = self.video_dropdown.currentText()
        self.selected_video_start_time = datetime.strptime(self.selected_video.replace(".mp4", "").replace(".mkv", ""),"%Y-%m-%d %H-%M-%S")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
