import itertools
import random
import numpy as np
import pandas as pd
import math
import time
import simpy
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from PIL import ImageTk
import random
from datetime import datetime, timedelta
import os
from collections import defaultdict

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import tkinter as tk

from PIL import ImageTk
# -------------------------
#  CONFIGURATION
# -------------------------

RFID_SELECTION_RATE = 0.7

RFID_GATE_LINES = 2
PAPER_GATE_LINES = 2

PAPER_EMPS_PER_LINE = 1
RFID_EMPS_PER_LINE = 1

RFID_SCAN_TIME_MIN = 5
RFID_SCAN_TIME_MAX = 12
PAPER_SCAN_TIME_MIN = 10
PAPER_SCAN_TIME_MAX = 17

JOIN_RATE_HIGH_MEAN = 5
JOIN_RATE_HIGH_STD = 0.05

JOIN_RATE_AVG_MEAN = 8
JOIN_RATE_AVG_STD = 1

JOIN_RATE_LOW_MEAN = 50
JOIN_RATE_LOW_STD = 10

ERROR_RATE_RFID = 0.01
ERROR_RATE_PAPER = 0.035
# Creating the 'output' directory if it doesn't exist
os.makedirs('output', exist_ok=True)


# -------------------------
#  ANALYTICAL GLOBALS
# -------------------------

arrivals = defaultdict(lambda: 0)
total_waits = defaultdict(list)
event_log = []


def get_current_time(elapsed_seconds):
    start_time = datetime.strptime('06:30:00', '%H:%M:%S')
    current_time = start_time + timedelta(seconds=elapsed_seconds)
    formatted_current_time = current_time.strftime('%H:%M:%S')
    return formatted_current_time


def avg_wait(raw_waits):
    waits = [w for i in raw_waits.values() for w in i]
    return round(np.mean(waits), 1) if waits else 0


def register_individual_arrival(time, person_id):
    arrivals[int(time)] += 1
    event_log.append({"event": "INDIVIDUAL_ARRIVAL", "time": get_current_time(time), "personId": person_id})


def queueing_to_scanner(person, card_type, gate_line, traffic_status, queue_begin, queue_end, scan_begin, scan_end,
                        error_appearance, correction_begin, error_correction_time, correction_end):
    wait = queue_end - queue_begin
    total_waits[int(queue_end)].append(wait)
    event_log.append({"event": "WAIT_IN_GATE_LINE", "person": f"id_{person}", "selected line": f"{card_type}_{gate_line}",
                      "traffic status": traffic_status, "begin time": get_current_time(queue_begin),
                      "end time": get_current_time(queue_end), "duration": round(wait, 2)})
    event_log.append({"event": "SCAN TICKET", "person": f"id_{person}", "selected line": f"{card_type}_{gate_line}",
                      "traffic status": traffic_status, "begin time": get_current_time(scan_begin),
                      "end time": get_current_time(scan_end), "duration": round(scan_end - scan_begin, 2)})
    if error_appearance:
        event_log.append({"event": "ERROR OCCURRENCE AND CORRECTION", "person": f"id_{person}",
                          "selected line": f"{card_type}_{gate_line}", "traffic status": traffic_status,
                          "begin time": get_current_time(correction_begin),
                          "end time": get_current_time(correction_end),
                          "duration": round(error_correction_time, 2)})

# -------------------------
#  UI/ANIMATION 
# -------------------------

root = tk.Tk()
# size 
root.geometry('1500x800+10+0')
# chiều ngang, chiều cao, khoảng cách với mép trái, khoảng cách với mép trên
# tilte 
root.title('MÔ PHỎNG VÀ TỐI ƯU HÓA BÃI ĐỖ XE TRƯỜNG ĐẠI HỌC') 
#icon
root.iconbitmap(r'anh\due.ico')
#background
root.configure(bg='#fff')
#label
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, expand = False)
label = Label(root,text="MÔ PHỎNG VÀ TỐI ƯU HÓA BÃI ĐỖ XE TRƯỜNG ĐẠI HỌC", font =('Arial',26),bg='black',fg='white', height=2)
label.pack(side=TOP, fill=X, expand=False)
#canvas
canvas = tk.Canvas(root, width = 1450, height = 270, bg = "white")
canvas.pack(side=tk.TOP, expand = False)

#plot
f = plt.Figure(figsize=(2, 2), dpi=72)
a3 = f.add_subplot(121)
a3.plot()
a1 = f.add_subplot(222)
a1.plot()
a2 = f.add_subplot(224)
a2.plot()
data_plot = FigureCanvasTkAgg(f, master=root)
data_plot.get_tk_widget().config(height = 400)
data_plot.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

class QueueGraphics:
    text_height = 50
    icon_top_margin = 5
    
    def __init__(self, icon_file, icon_width, queue_image_file, num_lines, canvas, x_top, y_top):
        self.icon_file = icon_file
        self.icon_width = icon_width
        self.queue_image_file = queue_image_file
        self.num_lines = num_lines
        self.canvas = canvas
        self.x_top = x_top
        self.y_top = y_top

        # Load queue image
        self.queue_image = tk.PhotoImage(file=self.queue_image_file)
        self.icon_image = tk.PhotoImage(file=self.icon_file)
        self.icons = defaultdict(lambda: [])
        for i in range(num_lines):
            # Create queue image
            canvas.create_image(x_top, y_top + (i * self.text_height), anchor=tk.NW, image=self.queue_image)
        self.canvas.update()

    def add_to_line(self, gate_line):
        count = len(self.icons[gate_line])
        x = self.x_top + 100 + (count * self.icon_width)
        y = self.y_top + ((gate_line - 1) * self.text_height) + self.icon_top_margin
        self.icons[gate_line].append(
                self.canvas.create_image(x, y, anchor=tk.NW, image=self.icon_image)
        )
        self.canvas.update()

    def remove_from_line(self, gate_line):
        if len(self.icons[gate_line]) == 0: return
        to_del = self.icons[gate_line].pop()
        self.canvas.delete(to_del)
        self.canvas.update()

def GATE(canvas, x_top, y_top):
    
    # Position the RFID gate above the PAPER gate
    rfid_gate = QueueGraphics(r"anh\xe xanh.png", 70, r'anh\cong xanh.png', RFID_GATE_LINES, canvas, x_top, y_top)
    paper_gate = QueueGraphics(r"anh\xe cam.png", 70, r'anh\cong cam.png', PAPER_GATE_LINES, canvas, x_top, y_top + 120)

    return rfid_gate, paper_gate


class ClockAndData:
    def __init__(self, canvas, x1, y1, x2, y2, time):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        self.train = canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, fill="#fff")
        self.time = canvas.create_text(self.x1 + 10, self.y1 + 10, text = "Current Time = "+ get_current_time(time), anchor = tk.NW)
        wait_time = avg_wait(total_waits)
        if wait_time >= 60:
            wait_time = round(wait_time / 60, 1)
            wait_time_string = str(wait_time) + 'm'
        else:
            wait_time_string = str(wait_time) + 's'
        self.seller_wait = canvas.create_text(self.x1 + 10, self.y1 + 35, text = "Avg. Wait Time = "+ wait_time_string, anchor = tk.NW)
        self.traffic = canvas.create_text(self.x1 + 10, self.y1 + 50, text = "Traffic Status = "+ check_traffic_status(get_current_time(time)), anchor = tk.NW)
        self.canvas.update()

    def tick(self, time):
        self.canvas.delete(self.time)
        self.canvas.delete(self.seller_wait)
        self.canvas.delete(self.traffic)

        wait_time = avg_wait(total_waits)
        if wait_time >= 60:
            wait_time = round(wait_time / 60, 1)
            wait_time_string = str(wait_time) + 'm'
        else:
            wait_time_string = str(wait_time) + 's'
            
        self.time = canvas.create_text(self.x1 + 10, self.y1 + 10, text = "Current Time = "+ get_current_time(time), anchor = tk.NW)
        self.seller_wait = canvas.create_text(self.x1 + 10, self.y1 + 30, text = "Avg. Wait Time = "+ wait_time_string, anchor = tk.NW)
        self.traffic = canvas.create_text(self.x1 + 10, self.y1 + 50, text = "Traffic Status = "+ check_traffic_status(get_current_time(time)), anchor = tk.NW)
        
        a1.cla()
        a1.set_xlabel("Time")
        a1.set_ylabel("Avg. Wait Time (seconds)")
        a1.step([ t for (t, waits) in total_waits.items() ], [ np.mean(waits) for (t, waits) in total_waits.items() ])
        
        a2.cla()
        a2.set_xlabel("Time")
        a2.set_ylabel("Arrivals")
        a2.bar([ t for (t, a) in arrivals.items() ], [ a for (t, a) in arrivals.items() ])
        
        data_plot.draw()
        self.canvas.update()



# -------------------------
#  SIMULATION
# -------------------------


def pick_shortest(lines):
    """
    Cho một danh sách các tài nguyên trong SimPy, xác định tài nguyên có hàng đợi ngắn nhất.
    
    Lưu ý rằng thứ tự hàng đợi được xáo trộn để không chọn hàng đợi đầu tiên quá nhiều.
    """
    # Tạo list chứa các hàng và index
    shuffled_lines = list(zip(range(len(lines)), lines))
    # Sắp xếp ngẫu nhiên vị trí các cổng trong list
    # Nhằm mô phỏng người tham gia chọn ngẫu nhiên trong số các cổng có cùng chiều dài ngắn nhất
    random.shuffle(shuffled_lines)
    
    # Tạm thời gán hàng chờ có độ dài ngắn nhất là hàng đầu tiên
    
    first_line = shuffled_lines[0]
    firt_line_length = first_line[0]
    idx_of_shortest = firt_line_length
    
    # Duyệt qua list các hàng đã được sắp xếp ngẫu nhiên
    for i, line in shuffled_lines:
        """
        Nếu chiều dài của hàng hiện tại ngắn hơn chiều dài của hàng đang giữ vị trí ngắn nhất
        thì cập nhật index của hàng ngắn nhất là index của hàng hiện tại.
        Dùng .queue để xác định những đối tượng đang request hàng hiện tại, khi mỗi hàng là một resource.
        Những đối tượng đang request resource được hiểu là những người đang xếp hàng tại hàng hiện tại.
        Dùng len() để xác định chiều dài của hàng hiện tại
        """
        if len(line.queue) < len(lines[idx_of_shortest].queue):
            idx_of_shortest = i
    # Trả về hàng ngắn nhất trong số tất cả các hàng được chọn
    return lines[idx_of_shortest], idx_of_shortest + 1
    
def create_clock(env):
    """
        This generator is meant to be used as a SimPy event to update the clock
        and the data in the UI
    """
    
    while True:
        yield env.timeout(50)
        clock.tick(env.now)

def generate_scan_time(card_type):
    if card_type == 'RFID':
        return random.uniform(RFID_SCAN_TIME_MIN, RFID_SCAN_TIME_MAX)
    elif card_type == 'paper':
        return random.uniform(PAPER_SCAN_TIME_MIN, PAPER_SCAN_TIME_MAX)
    
def check_traffic_status(current_time):
    if current_time < "08:30:00":
        return "low"
    elif "08:30:00" <= current_time < "09:30:00":
        return "avg"
    elif "09:30:00" <= current_time < "10:30:00":
        return "high"
    elif "10:30:00" <= current_time < "11:30:00":
        return "avg"
    elif "11:30:00" <= current_time < "12:30:00":
        return "high"
    elif "12:30:00" <= current_time < "14:30:00":
        return "low"
    elif "14:30:00" <= current_time < "15:30:00":
        return "avg"
    elif "15:30:00" <= current_time < "16:30:00":
        return "high"
    elif "16:30:00" <= current_time < "18:30:00":
        return "avg"
    elif "18:30:00" <= current_time < "19:30:00":
        return "high"
    else:
        return "low"

    
def generate_arrival_time(env):
    current_time = get_current_time(env.now)
    traffic_status = check_traffic_status(current_time)
    if traffic_status == 'high':
        arrival_time = max(0, random.normalvariate(JOIN_RATE_HIGH_MEAN, JOIN_RATE_HIGH_MEAN))
    elif traffic_status == 'low':
        arrival_time = max(0, random.normalvariate(JOIN_RATE_LOW_MEAN, JOIN_RATE_LOW_STD))
    else: # NORMAL
        arrival_time = max(0, random.normalvariate(JOIN_RATE_AVG_MEAN, JOIN_RATE_AVG_STD))        
    return arrival_time, traffic_status


def is_error(card_type):
    if card_type == 'RFID':
        return random.random() <= ERROR_RATE_RFID
    elif card_type == 'paper':
        return random.random() <= ERROR_RATE_PAPER
    
def generate_error_correction_time(card_type):
    if card_type == 'RFID':
        if RFID_EMPS_PER_LINE == 1:
            ERROR_CORRECTION_TIME = max(0, random.normalvariate(15, 5))
        elif 0.5 < RFID_EMPS_PER_LINE < 1:
            ERROR_CORRECTION_TIME = max(0, random.normalvariate(20, 5))
        else:
            ERROR_CORRECTION_TIME = max(0, random.normalvariate(30, 5))
        return max(0, random.normalvariate(20, 5))
    elif card_type == 'paper':
        ERROR_CORRECTION_TIME = max(0, random.normalvariate(10, 2))
        
    return ERROR_CORRECTION_TIME
def individual_arrival(env, rfid_gate_lines, paper_gate_lines):
    next_person_id = 0
    while True:
        # Select card type randomly
        card_type = random.choices(['RFID', 'paper'], weights=[RFID_SELECTION_RATE, 1 - RFID_SELECTION_RATE])[0]

        if card_type == 'RFID':
            gate_lines = rfid_gate_lines
        else:
            gate_lines = paper_gate_lines
        
        next_arrival, traffic_status = generate_arrival_time(env)
        yield env.timeout(next_arrival)

        env.process(purchasing_individual(env, next_person_id, gate_lines, card_type, traffic_status))
        next_person_id += 1

def purchasing_individual(env, person_id, gate_lines, card_type, traffic_status):
    # Ensure card_type is valid
    if card_type not in ["RFID", "paper"]:
        raise ValueError("Invalid card type")

    queue_begin = env.now

    # Select the gate line based on card type
    if card_type == "RFID":
        gate_line = 0  # RFID gate line
        shortest_gate_line, gate_line = pick_shortest(gate_lines)
        gates[0].add_to_line(gate_line)  # Add an item to the RFID gate line
    else:
        gate_line = 0  # Paper gate line
        shortest_gate_line, gate_line = pick_shortest(gate_lines)
        gates[1].add_to_line(gate_line)  # Add an item to the paper gate line
    
    with shortest_gate_line.request() as req:
        yield req
        
        # Remove the item from the gate line when processed
        if card_type == "RFID":
            gates[0].remove_from_line(gate_line)  # Remove the item from the RFID gate line
        else:
            gates[1].remove_from_line(gate_line)  # Remove the item from the paper gate line
        
        queue_end = env.now

        # SCANNING
        scan_begin = env.now
        scan_time = generate_scan_time(card_type=card_type)
        yield env.timeout(scan_time)
        scan_end = env.now
        
        # ERROR
        correction_begin = env.now
        error_appearance = is_error(card_type)
        if error_appearance:
            error_correction_time = generate_error_correction_time(card_type)
        else:
            error_correction_time = 0
        
        yield env.timeout(error_correction_time)
        correction_end = env.now
            
        queueing_to_scanner(person_id, card_type, gate_line, traffic_status, queue_begin, queue_end, scan_begin, scan_end, error_appearance, correction_begin, error_correction_time, correction_end)
        # ERROR
        correction_begin = env.now
        error_appearance = is_error(card_type)
        if error_appearance:
            error_correction_time = generate_error_correction_time(card_type)
        else:
            error_correction_time = 0
        
        yield env.timeout(error_correction_time)
        correction_end = env.now
            
        queueing_to_scanner(person_id, card_type, gate_line, traffic_status, queue_begin, queue_end, scan_begin, scan_end, error_appearance, correction_begin, error_correction_time, correction_end)

        # Remove from the queue once processed
        del event_log[-1]


gates = GATE(canvas, 340, 15)
clock = ClockAndData(canvas, 1250, 180, 1440, 260, 0)
env = simpy.Environment()


rfid_gate_lines = [simpy.Resource(env, capacity=RFID_EMPS_PER_LINE) for _ in range(RFID_GATE_LINES)]
paper_gate_lines = [simpy.Resource(env, capacity=PAPER_EMPS_PER_LINE) for _ in range(PAPER_GATE_LINES)]
all_gate_lines = [rfid_gate_lines, paper_gate_lines]


env.process(individual_arrival(env, rfid_gate_lines, paper_gate_lines))
env.process(create_clock(env))
env.run(until=14*60*60)

root.mainloop()
    
# Writing data to a JSON file
with open('output/events.json', 'w') as outfile:
    # input_string = f"""RFID Gates: {RFID_GATE_LINES} | Paper Gates: {PAPER_GATE_LINES} || Paper Employees per Line: {PAPER_EMPS_PER_LINE} | RFID Employees per Line: {RFID_EMPS_PER_LINE}"""
    # config_string = f"""RFID Selection Rate: {RFID_SELECTION_RATE} || RFID Scan Time (Min): {RFID_SCAN_TIME_MIN}| RFID Scan Time (Max): {RFID_SCAN_TIME_MAX} || Paper Scan Time (Min): {PAPER_SCAN_TIME_MIN}| Paper Scan Time (Max): {PAPER_SCAN_TIME_MAX} || Join Rate High Mean: {JOIN_RATE_HIGH_MEAN}| Join Rate High Std: {JOIN_RATE_HIGH_STD} || Join Rate Avg Mean: {JOIN_RATE_AVG_MEAN}| Join Rate Avg Std: {JOIN_RATE_AVG_STD} || Join Rate Low Mean: {JOIN_RATE_LOW_MEAN}| Join Rate Low Std: {JOIN_RATE_LOW_STD} || Error Rate RFID: {ERROR_RATE_RFID}| Error Rate Paper: {ERROR_RATE_PAPER}"""
    json.dump({"RFID GATES": RFID_GATE_LINES, 
               "PAPER GATES": PAPER_GATE_LINES,
               "RFID EMPLOYEES": int(RFID_GATE_LINES * RFID_EMPS_PER_LINE),
               "PAPER EMPLOYEES": PAPER_GATE_LINES * PAPER_EMPS_PER_LINE,
               "events": event_log}, outfile, indent=4)
