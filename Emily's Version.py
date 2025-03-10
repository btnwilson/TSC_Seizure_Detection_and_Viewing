import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from PIL import Image, ImageTk
import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates
import threading
import queue
import time

plt.ioff()
mice = os.listdir("Z:\Emily\TSCproject\Event Identification Events\eventid\Events")
selected_mouse = "cmvcre3_136"
file_path = os.path.join("Z:\Emily\TSCproject\Event Identification Events\eventid\Events", selected_mouse)
files = os.listdir(file_path)
selected_file = files[0]
 
video_path = "Z:\Emily TSC Video Capturing" #Change to correct path to video location for your device
videos = os.listdir(video_path)
video_start_times = [datetime.strptime(name.replace(".mp4", "").replace(".mkv", "").replace("._", ""), "%Y-%m-%d %H-%M-%S") for name in videos if name.endswith((".mp4", ".mkv"))]
 
with open(os.path.join(file_path, selected_file), 'rb') as f:
    data = pickle.load(f)
 
# Global variables
current_event_index = 0
fs = 30
playback_speed = 1
is_playing = False
cap = None
frame_queue = queue.Queue()
last_frame = None
time_marker_ax = None
start_frame_index = None

sorted_indices = sorted(range(len(video_start_times)), key=lambda i: video_start_times[i])
sorted_video_start_times = np.array([video_start_times[i] for i in sorted_indices])
sorted_video_names = np.array([videos[i] for i in sorted_indices])

eeg_data = data[f"Event {current_event_index}"]["EEG Signal"]
event_start_time = datetime.strptime(data[f"Event {current_event_index}"]["Start Time"], "%m-%d-%Y %H:%M:%S")

max_spikes = data[f"Event {current_event_index}"]["Max Spikes"]
max_arclength = data[f"Event {current_event_index}"]["Max Arclength"]
event_duration = data[f"Event {current_event_index}"]["Duration"]
cage_num = data[f"Event {current_event_index}"]["Cage Number"]

selected_videos = []
selected_video_start_time = None
time_window = timedelta(minutes=60)
for video_number in range(len(sorted_video_start_times) - 1):
    if sorted_video_start_times[video_number] <= event_start_time <= sorted_video_start_times[video_number + 1]:
        potential_videos = (sorted_video_start_times >= sorted_video_start_times[video_number] - time_window) & (sorted_video_start_times <= sorted_video_start_times[video_number])
        if np.sum(potential_videos) > 0:
            selected_videos = list(sorted_video_names[potential_videos])
        else:
            selected_videos = list(sorted_video_names[video_number])
        break
selected_video = selected_videos[0]
selected_video_start_time = datetime.strptime(selected_video.replace(".mp4", "").replace(".mkv", ""), "%Y-%m-%d %H-%M-%S")

def create_plot(frame):
    global time_marker_ax
    try:
        for widget in frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 3))  
        eeg_data = data[f"Event {current_event_index}"]["EEG Signal"]
        event_start_time = datetime.strptime(data[f"Event {current_event_index}"]["Start Time"], "%m-%d-%Y %H:%M:%S")
        
        interval = timedelta(seconds=1 / fs)
        time_array = [event_start_time + i * interval for i in range(len(eeg_data))]
        time_array = np.array(time_array)
        
        ax.plot(time_array, eeg_data)
        ax.set_title(f"Event {current_event_index + 1} from {selected_file}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized EEG")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.xticks(rotation=45)
        plt.tight_layout() 
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    except Exception as e:
        print(f"Error in create_plot: {e}")

def resize_frame(frame, width):
    height, original_width = frame.shape[:2]
    aspect_ratio = original_width / height
    new_height = int(width / aspect_ratio)
    return cv2.resize(frame, (width, new_height))

def play_video():
    global cap, is_playing, last_frame
    while is_playing and cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                is_playing = False
                break
            
            frame = resize_frame(frame, 1800)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            frame_queue.put(img)
        except Exception as e:
            print(f"Error in play_video: {e}")
            is_playing = False

def update_frame():
    global time_marker_ax, event_start_time, start_frame_index
    try:
        if not frame_queue.empty():
            img = frame_queue.get_nowait()
            video_label.imgtk = img
            video_label.configure(image=img)
            
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_dif = max(0, current_frame - start_frame_index)
            current_frame_sec = int(frame_dif / 30) # 30 is sampling frequency
            current_frame_time = event_start_time + timedelta(seconds=current_frame_sec)
            
            
        video_label.after(5, update_frame)
    except Exception as e:
        print(f"Error in update_frame: {e}")
        video_label.after(5, update_frame)

def play_pause_video():
    global is_playing
    if not is_playing:
        is_playing = True
        threading.Thread(target=play_video, daemon=True).start()
    else:
        is_playing = False

def next_event():
    global current_event_index
    if current_event_index < len(data) - 1:
        current_event_index += 1
        time.sleep(0.5)
        
        find_possible_videos()
        
        update_plots()
        
    else:
        event_textbox.delete(1.0, tk.END)
        event_textbox.insert(tk.END, "All Events Viewed")

def previous_event():
    global current_event_index
    if current_event_index > 0:
        current_event_index -= 1
        time.sleep(0.5)
        
        find_possible_videos()
        
        update_plots()
    else:
        event_textbox.delete(1.0, tk.END)
        event_textbox.insert(tk.END, "No Prior Events")

def jump_forward():
    global cap, is_playing
    if cap is not None:
        try:
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 15 * 30)
            if is_playing:
                update_frame()
        except Exception as e:
            print(f"Error in jump_forward: {e}")

def jump_backward():
    global cap, is_playing
    if cap is not None:
        try:
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 5 * 30))
            if is_playing:
                update_frame()
        except Exception as e:
            print(f"Error in jump_backward: {e}")

def update_plots():
    global cap, selected_video, selected_video_start_time, last_frame, start_frame_index, selected_videos
    
    if cap is not None:
        cap.release()
    
    eeg_data = data[f"Event {current_event_index}"]["EEG Signal"]
    event_start_time = datetime.strptime(data[f"Event {current_event_index}"]["Start Time"], "%m-%d-%Y %H:%M:%S")
    
    cap = cv2.VideoCapture(os.path.join(video_path, selected_video))
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {selected_video}")
        return
        
    try:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = ImageTk.PhotoImage(Image.fromarray(frame))
            #video_label.configure(image=last_frame)
        
        time_difference = event_start_time - selected_video_start_time
        time_from_start = time_difference.total_seconds()
        start_frame_index = int(time_from_start * 30 - 5 * 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        
        create_plot(graph_frame)
        
        event_textbox.delete(1.0, tk.END)
        max_spikes = data[f"Event {current_event_index}"]["Max Spikes"]
        max_arclength = data[f"Event {current_event_index}"]["Max Arclength"]
        event_duration = data[f"Event {current_event_index}"]["Duration"]
        cage_num = data[f"Event {current_event_index}"]["Cage Number"]
        
        event_textbox.insert(tk.END, f"Max Spikes: {max_spikes} \t Max Linelength: {max_arclength} \t Event Duration: {event_duration}s \t Cage Number: {cage_num}")
        
        if cap.isOpened():
            update_frame()
    except Exception as e:
        print(f"Error in update_plots: {e}")

def set_file():
    global selected_file, current_event_index, data, file_path
    selected_file = file_options.get()
    
    try:
        with open(os.path.join(file_path, selected_file), 'rb') as f:
            data = pickle.load(f)
        
        current_event_index = 0
        update_plots()
    except Exception as e:
        print(f"Error in set_file: {e}")

def set_mouse():
    global selected_mouse, selected_file, files, file_path
    
    selected_mouse = mouse_options.get()
    file_path = os.path.join("Events", selected_mouse)
    files = os.listdir(file_path)  
    file_options['values'] = files
    file_options.set(f"")

def set_video():
    global selected_video_start_time, selected_video
    
    selected_video = video_options.get()
    selected_video_start_time = datetime.strptime(selected_video.replace(".mp4", "").replace(".mkv", ""), "%Y-%m-%d %H-%M-%S")
    update_plots()
    
def find_possible_videos():
    global selected_video_start_time, selected_video, selected_videos, sorted_video_start_times, sorted_video_names
    time_window = timedelta(minutes=60)
    print(event_strart_time)
    
    for video_number in range(len(sorted_video_start_times) - 1):
        if sorted_video_start_times[video_number] <= event_start_time <= sorted_video_start_times[video_number + 1]:
            potential_videos = (sorted_video_start_times >= sorted_video_start_times[video_number] - time_window) & (sorted_video_start_times <= sorted_video_start_times[video_number])
            if np.sum(potential_videos) > 0:
                selected_videos = list(sorted_video_names[potential_videos])
            else:
                selected_videos = list(sorted_video_names[potential_videos])
            break
    print(selected_videos)
    selected_video = selected_videos[0]
    selected_video_start_time = datetime.strptime(selected_video.replace(".mp4", "").replace(".mkv", ""), "%Y-%m-%d %H-%M-%S")
    print(selected_video)
    video_options['values'] = selected_videos
    video_options.set(f"{selected_video}")


root = tk.Tk()
root.title("EEG Event Video Player")
root.geometry("1200x800")  

file_control_frame = ttk.Frame(root, height=50)
file_control_frame.grid(row=0, column=0, columnspan=3, sticky="ew")

video_frame = ttk.Frame(root, width=800, height=400)
video_frame.grid(row=1, column=1, sticky="ns")

graph_frame = ttk.Frame(root, width=1200, height=300)
graph_frame.grid(row=3, column=0, columnspan=3, sticky="n")

video_control_frame = ttk.Frame(root, height=50)
video_control_frame.grid(row=2, column=0, columnspan=3, sticky="ew")

event_info_frame = ttk.Frame(root, width=1200, height=150)
event_info_frame.grid(row=4, column=0, columnspan=3, sticky="n")

root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=2)  
root.grid_rowconfigure(2, weight=0)  
root.grid_rowconfigure(3, weight=0)  
root.grid_rowconfigure(4, weight=1) 

root.grid_columnconfigure(0, weight=1) 
root.grid_columnconfigure(1, weight=2)  
root.grid_columnconfigure(2, weight=1)  

video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0, sticky="nsew")

mouse_options = ttk.Combobox(file_control_frame, values= mice)
mouse_options.set(f"{selected_mouse}")
mouse_options.pack(side=tk.LEFT, padx=5)

mouse_selection_button = tk.Button(file_control_frame, text="Confirm Mouse", command=set_mouse)
mouse_selection_button.pack(side=tk.LEFT, padx=5)

file_options = ttk.Combobox(file_control_frame, values= files)
file_options.set(f"{selected_file}")
file_options.pack(side=tk.LEFT, padx=5)

file_selection_button = tk.Button(file_control_frame, text= "Confirm File", command=set_file)
file_selection_button.pack(side=tk.LEFT, padx=5)

video_options = ttk.Combobox(file_control_frame, values=selected_videos)
video_options.set(f"{selected_videos[0]}")
video_options.pack(side=tk.RIGHT, padx=5)

video_selection_button = tk.Button(file_control_frame, text= "Confirm Video", command=set_video)
video_selection_button.pack(side=tk.RIGHT, padx=5)


play_button = ttk.Button(video_control_frame, text="Play/Pause", command=play_pause_video)
play_button.pack(side=tk.LEFT, padx=5)

prev_button = ttk.Button(video_control_frame, text="Previous Event", command=previous_event)
prev_button.pack(side=tk.LEFT, padx=5)

next_button = ttk.Button(video_control_frame, text="Next Event", command=next_event)
next_button.pack(side=tk.LEFT, padx=5)

jump_backward_button = ttk.Button(video_control_frame, text="Jump -5s", command=jump_backward)
jump_backward_button.pack(side=tk.LEFT, padx=5)

jump_forward_button = ttk.Button(video_control_frame, text="Jump +15s", command=jump_forward)
jump_forward_button.pack(side=tk.LEFT, padx=5)

event_label = ttk.Label(event_info_frame, text="Event Information")
event_label.pack(side=tk.TOP, pady=5)

event_textbox = tk.Text(event_info_frame, height=5, wrap=tk.WORD, width=100)
event_textbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

event_textbox.insert(tk.END, f"Max Spikes: {max_spikes} \t Max Linelength: {max_arclength} \t Event Duration: {event_duration}s \t Cage Number: {cage_num}")

create_plot(graph_frame)

cap = cv2.VideoCapture(os.path.join(video_path, selected_video))
time_difference = event_start_time - selected_video_start_time
time_from_start = time_difference.total_seconds()

start_frame_index = int(time_from_start * 30 - 5 * 30)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

if cap.isOpened():
    update_frame()
else:
    print("Error: Could not open video file.")

root.mainloop()

if cap is not None:
    cap.release()
