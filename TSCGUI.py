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
import matplotlib.dates as mdates
import time

class MatplotlibWidget(FigureCanvas):
    def __init__(self):#, width=11, height=3, dpi=100):
        self.fig = plt.Figure(figsize=(11, 3), dpi=100)
        super(MatplotlibWidget, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("EEG Trace", fontsize=10)
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
        self.ax.set_xlabel("Time (seconds)", fontsize=9)
        self.ax.set_ylabel("Normalized EEG", fontsize=9)

        #super(MatplotlibWidget, self).__init__(self.fig)

    def updateplot(self, y, fs, event_start_time, current_event_index, selected_file):  # Renamed from plot to updateplot
        interval = timedelta(seconds=1 / fs)
        time_array = [event_start_time + i * interval for i in range(len(y))]
        time_array = np.array(time_array)
        self.ax.clear()  # Clear previous plot
        self.ax.plot(time_array, y)  # Plot new data
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M:%S'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.set_title(f"Event {current_event_index + 1} from {selected_file}")
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
        self.mouse_selection_button = QPushButton("Select Mouse")
        self.file_selection_button = QPushButton("Select File")
        self.video_selection_button = QPushButton("Select Video")

        # Add the dropdowns and labels to row1_layout

        row1_layout.addWidget(self.mice_dropdown)
        row1_layout.addWidget(self.mouse_selection_button)
        row1_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        row1_layout.addWidget(self.file_dropdown)
        row1_layout.addWidget(self.file_selection_button)
        row1_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        row1_layout.addWidget(self.video_dropdown)
        row1_layout.addWidget(self.video_selection_button)

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
        self.slider.setRange(0, 8)
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

        # Row 5: Text box at the bottom (non-editable)
        self.text_box = QTextEdit(self)
        self.text_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_box.setFixedHeight(50)
        self.text_box.setPlainText("Event Information Will Be Displayed Here")
        self.text_box.setReadOnly(True)  

        # Adding all rows to main layout
        main_layout.addLayout(row1_layout)
        main_layout.addWidget(self.video_widget)
        main_layout.addLayout(row3_layout)
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.text_box)

        # Set central widget
        self.setCentralWidget(central_widget)
        # ---------------------------------------------------------

        #initialization of video playing and plot
        #data_location = "C:/Users/bentn/OneDrive - University of Vermont/Desktop/Work/New data"
        data_location = "Z:\Emily\TSCproject\Event Identification Events\eventid"
        os.chdir(data_location)
        mice = [f for f in os.listdir("Events/") if ".DS_Store" not in f]
        self.mice_dropdown.addItems(mice)
        selected_mouse = self.mice_dropdown.currentText()
        self.file_dropdown.addItems(os.listdir(os.path.join("Events", selected_mouse)))
        selected_file = self.file_dropdown.currentText()
        self.pickle_location = os.path.join("Events", selected_mouse, selected_file)
        self.data = None
        self.current_event_index = 0
        self.open_pickle()

        self.video_path = "Z:\Emily TSC Video Capturing"
        self.all_videos = os.listdir(self.video_path)
        self.video_start_times = [datetime.strptime(name.replace(".mp4", "").replace(".mkv", "").replace("._", ""), "%Y-%m-%d %H-%M-%S") for name in self.all_videos if name.endswith((".mp4", ".mkv"))]

        self.fs = 256
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

        self.plot_widget.updateplot(self.eeg_data, self.fs, self.event_start_time, self.current_event_index, self.file_dropdown.currentText())

        self.selected_video = None
        self.selected_video_start_time = None
        self.identify_videos()

        if self.selected_video != None:
            self.cap = cv2.VideoCapture(os.path.join(self.video_path, self.selected_video))
            self.video_fs = self.cap.get(cv2.CAP_PROP_FPS)
            time_difference = self.event_start_time - self.selected_video_start_time
            time_from_start = time_difference.total_seconds()
            start_frame_index = int(time_from_start * self.video_fs - 5 * self.video_fs)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)


        # button functionality
        self.mouse_selection_button.clicked.connect(self.mouse_changed)
        self.file_selection_button.clicked.connect(self.file_changed)
        self.video_selection_button.clicked.connect(self.video_changed)
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
        if self.isplaying and self.selected_video != None:
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
        if self.selected_video != None:
            if self.cap is not None:
                self.cap.release()

            self.event_start_time = datetime.strptime(self.data[f"Event {self.current_event_index}"]["Start Time"],
                                                      "%m-%d-%Y %H:%M:%S")
            print(f"Print Event Start Time: {self.event_start_time}")
            
            self.cap = cv2.VideoCapture(os.path.join(self.video_path, self.selected_video))
            self.video_fs = self.cap.get(cv2.CAP_PROP_FPS)
            if not self.cap.isOpened():
                print(f"Error: Unable to open video file {self.selected_video}")
                return
            try:
                ret, frame = self.cap.read()

                print(f"Video Start Time: {self.selected_video_start_time}")
                time_difference = self.event_start_time - self.selected_video_start_time
                print(f"Time difference seconds {time_difference}")
                
                time_from_start = time_difference.total_seconds()
                start_frame_index = int(time_from_start * self.video_fs - 5 * self.video_fs)
                print(f"Start frame index: {start_frame_index}")
                
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Total Frames in Video: {total_frames}")
                
                if start_frame_index < total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
                    self.video_widget.setStyleSheet(
                        "background-color: black; color: white; font-size: 18px; font-weight: bold;")
                    self.video_widget.setText(
                        "Video Available Press Play/Pause")  # Display text when no video is loaded
                    self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

                else:
                    self.isplaying = False
                    self.video_widget.setStyleSheet("background-color: black; color: white; font-size: 18px; font-weight: bold;")
                    self.video_widget.setText("No Video Available")
                    self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)


                self.max_spikes = self.data[f"Event {self.current_event_index}"]["Max Spikes"]
                self.max_arclength = self.data[f"Event {self.current_event_index}"]["Max Arclength"]
                self.event_duration = self.data[f"Event {self.current_event_index}"]["Duration"]
                self.cage_num = self.data[f"Event {self.current_event_index}"]["Cage Number"]


                self.text_box.setPlainText(f"Max Spikes: {self.max_spikes} \t Max Linelength: {self.max_arclength} \t Event Duration: {self.event_duration}s \t Cage Number: {self.cage_num} \t Start Time: {self.event_start_time}")

                self.plot_widget.updateplot(self.eeg_data, self.fs, self.event_start_time, self.current_event_index, self.file_dropdown.currentText())


                self.video_timer.start()
            except Exception as e:
                print(f"Error in change video: {e}")

        self.eeg_data = self.data[f"Event {self.current_event_index}"]["EEG Signal"]
        self.event_start_time = datetime.strptime(self.data[f"Event {self.current_event_index}"]["Start Time"],
                                                  "%m-%d-%Y %H:%M:%S")
        self.max_spikes = self.data[f"Event {self.current_event_index}"]["Max Spikes"]
        self.max_arclength = self.data[f"Event {self.current_event_index}"]["Max Arclength"]
        self.event_duration = self.data[f"Event {self.current_event_index}"]["Duration"]
        self.cage_num = self.data[f"Event {self.current_event_index}"]["Cage Number"]

        self.text_box.setPlainText(
            f"Max Spikes: {self.max_spikes} \t Max Linelength: {self.max_arclength} \t Event Duration: {self.event_duration}s \t Cage Number: {self.cage_num}  \t Start Time: {self.event_start_time}")

        self.plot_widget.updateplot(self.eeg_data, self.fs, self.event_start_time, self.current_event_index, self.file_dropdown.currentText())

    def change_playback_speed(self):
        self.video_timer.stop()
        self.playback_speed = self.get_playback_speed()
        self.milliseconds_per_frame = int(1000 /(self.video_fs * self.playback_speed))
        self.video_timer.setInterval(self.milliseconds_per_frame)
        self.video_timer.start()

    def jump_forward(self):
        if self.cap is not None:
            try:
                self.video_timer.stop()
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

                jump_frame = current_frame + 15 * self.video_fs
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if jump_frame < total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, jump_frame))
                else:
                    self.isplaying = False
                    self.video_widget.setStyleSheet(
                        "background-color: black; color: white; font-size: 18px; font-weight: bold;")
                    self.video_widget.setText("No Video Available")
                    self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

                self.video_timer.start()
            except Exception as e:
                print(f"Error in jump_backward: {e}")

    def jump_back(self):
        if self.cap is not None:
            try:
                self.video_timer.stop()
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 5 * self.video_fs))
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

            self.eeg_data = self.data[f"Event {self.current_event_index}"]["EEG Signal"]
            self.event_start_time = datetime.strptime(self.data[f"Event {self.current_event_index}"]["Start Time"],
                                                      "%m-%d-%Y %H:%M:%S")

            self.max_spikes = self.data[f"Event {self.current_event_index}"]["Max Spikes"]
            self.max_arclength = self.data[f"Event {self.current_event_index}"]["Max Arclength"]
            self.event_duration = self.data[f"Event {self.current_event_index}"]["Duration"]
            self.cage_num = self.data[f"Event {self.current_event_index}"]["Cage Number"]

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


    def mouse_changed(self):
        self.current_event_index = 0
        selected_mouse = self.mice_dropdown.currentText()
        self.file_dropdown.clear()
        files = os.listdir(os.path.join("Events", selected_mouse))
        self.file_dropdown.addItems(files)
        selected_file = self.file_dropdown.currentText()
        self.pickle_location = os.path.join("Events", selected_mouse, selected_file)
        self.open_pickle()
        self.identify_videos()
        self.change_video()

    def file_changed(self):
        self.current_event_index = 0
        selected_mouse = self.mice_dropdown.currentText()
        selected_file = self.file_dropdown.currentText()
        self.pickle_location = os.path.join("Events", selected_mouse, selected_file)
        self.open_pickle()
        self.identify_videos()
        self.change_video()

    def video_changed(self):
        self.selected_video = self.video_dropdown.currentText()
        self.selected_video_start_time = datetime.strptime(self.selected_video.replace(".mp4", "").replace(".mkv", ""),"%Y-%m-%d %H-%M-%S")
        self.change_video()

    def open_pickle(self):
        with open(self.pickle_location, 'rb') as f:
            self.data = pickle.load(f)

        self.eeg_data = self.data[f"Event {self.current_event_index}"]["EEG Signal"]
        self.event_start_time = datetime.strptime(self.data[f"Event {self.current_event_index}"]["Start Time"],
                                                  "%m-%d-%Y %H:%M:%S")

        self.max_spikes = self.data[f"Event {self.current_event_index}"]["Max Spikes"]
        self.max_arclength = self.data[f"Event {self.current_event_index}"]["Max Arclength"]
        self.event_duration = self.data[f"Event {self.current_event_index}"]["Duration"]
        self.cage_num = self.data[f"Event {self.current_event_index}"]["Cage Number"]
        self.text_box.setPlainText(f"Max Spikes: {self.max_spikes} \t Max Linelength: {self.max_arclength} \t Event Duration: {self.event_duration}s \t Cage Number: {self.cage_num}  \t Start Time: {self.event_start_time}")

    def get_playback_speed(self):
        slider_value = self.slider.value()
        return slider_value * .25 + .5

    def identify_videos(self):
        selected_videos = []
        time_window = timedelta(minutes=60)  # 60-minute search window
        max_days = timedelta(days=5)
        
        print(f"Selected Videos Before: {selected_videos}")
        for video_number in range(len(self.sorted_video_start_times) - 1):
            if self.sorted_video_start_times[video_number] <= self.event_start_time <= self.sorted_video_start_times[video_number + 1]:
                potential_videos = (self.sorted_video_start_times >= self.sorted_video_start_times[video_number] - time_window) & (self.sorted_video_start_times <= self.sorted_video_start_times[video_number])

                filtered_videos = []
                for i, valid in enumerate(potential_videos):
                    if valid:
                        video_time = self.sorted_video_start_times[i]
                        time_difference = abs(self.event_start_time - video_time)

                        if time_difference <= max_days:
                            filtered_videos.append(self.sorted_video_names[i])

                if filtered_videos:
                    selected_videos = filtered_videos
                else:
                    selected_videos = []
                break
        print(f"Selected Videos After: {selected_videos}")
        self.video_dropdown.blockSignals(True)
        self.video_dropdown.clear()

        if selected_videos:
            self.video_dropdown.addItems(selected_videos)
            self.video_dropdown.setCurrentIndex(0)
            self.selected_video = self.video_dropdown.currentText()
            self.selected_video_start_time = datetime.strptime(
                self.selected_video.replace(".mp4", "").replace(".mkv", ""), "%Y-%m-%d %H-%M-%S")

        else:
            self.isplaying = False
            self.selected_video = None
            self.selected_video_start_time = None
            self.video_widget.setStyleSheet("background-color: black; color: white; font-size: 18px; font-weight: bold;")
            self.video_widget.setText("No Video Available")  # Display text when no video is loaded
            self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.video_dropdown.blockSignals(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())








