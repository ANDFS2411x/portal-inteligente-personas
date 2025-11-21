import tkinter as tk
from tkinter import ttk
import os
import signal
import subprocess

process = None
BASE_PATH = "/home/andfs/portal-inteligente-personas"

# ---------- PROCESS CONTROL ----------

def stop_system():
    global process
    if process is not None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            pass
        process = None
        status_var.set("✔ System stopped.")
        status_label.config(fg="#00ff66")


def run_system():
    global process
    stop_system()

    detector = detector_var.get()
    tracker = tracker_var.get()

    # ---------- ROUTES AND COMMANDS ----------
    if detector == "YOLOv5" and tracker == "SORT":
        folder = "cpp_det_track_count_yolov5-sort"
        command = ["./portal"]

    elif detector == "YOLOv5" and tracker == "ByteTrack":
        folder = "cpp_det_track_count_yolov5-bytetrack"
        command = ["./portal_bytetrack"]

    elif detector == "YOLOv11" and tracker == "ByteTrack":
        folder = "cpp_det_track_count_yolov11-bytetrack"
        command = [
            "./portal_yolov11_bytetrack",
            "/home/andfs/portal-inteligente-personas/modelos/YoloV11s/100_Épocas/best-yolov11-100.engine"
        ]

    elif detector == "YOLOv11" and tracker == "SORT":
        folder = "cpp_det_track_count_yolov11-sort"
        command = [
            "./portal_yolov11_sort",
            "/home/andfs/portal-inteligente-personas/modelos/YoloV11s/100_Épocas_v2/best_yolov11s_100epochs.engine"
        ]

    final_path = f"{BASE_PATH}/{folder}"

    status_var.set("▶ Running system...")  
    status_label.config(fg="#66ff99")

    # Start process in its own group so STOP kills everything
    process = subprocess.Popen(
        command,
        cwd=final_path,
        preexec_fn=os.setsid
    )


# ---------- UI (GREEN PROFESSIONAL STYLE) ----------

root = tk.Tk()
root.title("People Counter - Launcher")
root.geometry("430x320")
root.configure(bg="#0d0d0d")

# Header
header = tk.Label(
    root,
    text="People Counter Launcher",
    bg="#0d0d0d",
    fg="#00ff66",
    font=("Segoe UI", 20, "bold")
)
header.pack(pady=15)

# Frame for menu selection
frame = tk.Frame(root, bg="#0d0d0d")
frame.pack(pady=10)

# Detector
tk.Label(frame, text="Detector", bg="#0d0d0d", fg="white",
         font=("Segoe UI", 12)).grid(row=0, column=0, padx=10, pady=5)

detector_var = tk.StringVar(value="YOLOv5")
detector_menu = ttk.Combobox(frame, textvariable=detector_var,
                             values=["YOLOv5", "YOLOv11"], width=17)
detector_menu.grid(row=0, column=1)

# Tracker
tk.Label(frame, text="Tracker", bg="#0d0d0d", fg="white",
         font=("Segoe UI", 12)).grid(row=1, column=0, padx=10, pady=5)

tracker_var = tk.StringVar(value="SORT")
tracker_menu = ttk.Combobox(frame, textvariable=tracker_var,
                            values=["SORT", "ByteTrack"], width=17)
tracker_menu.grid(row=1, column=1)

# Button styling
def style_button(btn, color):
    btn.configure(
        bg=color,
        fg="black",
        activebackground="#00cc55",
        activeforeground="black",
        relief="flat",
        bd=0,
        font=("Segoe UI", 12, "bold"),
        height=1
    )

# Start Button
btn_start = tk.Button(root, text="START COUNTING", command=run_system, width=28)
style_button(btn_start, "#00ff66")
btn_start.pack(pady=12)

# Stop Button
btn_stop = tk.Button(root, text="STOP SYSTEM", command=stop_system, width=28)
style_button(btn_stop, "#ff6666")
btn_stop.pack()

# Status Label
status_var = tk.StringVar()
status_label = tk.Label(root, textvariable=status_var, bg="#0d0d0d",
                        fg="white", font=("Segoe UI", 11))
status_label.pack(pady=15)

root.mainloop()
