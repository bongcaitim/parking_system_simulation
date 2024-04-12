import itertools
import time
import simpy
from tkinter import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tkinter as tk
import json
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk
import os

######################
##### INPUTS #########
######################
RFID_GATE_LINES = 2
PAPER_GATE_LINES = 2

PAPER_EMPS_PER_LINE = 1
RFID_EMPS_PER_LINE = 0.5

######################
### CONFIGURATIONS ###
######################
RFID_SELECTION_RATE = 0.7

RFID_SCAN_TIME_MIN = 9
RFID_SCAN_TIME_MAX = 13
PAPER_SCAN_TIME_MIN = 10
PAPER_SCAN_TIME_MAX = 17

JOIN_RATE_HIGH_MEAN = 3
JOIN_RATE_HIGH_STD = 0.5

JOIN_RATE_AVG_MEAN = 15
JOIN_RATE_AVG_STD = 1

JOIN_RATE_LOW_MEAN = 60
JOIN_RATE_LOW_STD = 10

ERROR_RATE_RFID = 0.01
ERROR_RATE_PAPER = 0.035
# Creating the 'anh' directory if it doesn't exist
os.makedirs('anh', exist_ok=True)


arrivals = defaultdict(lambda: 0)
total_waits = defaultdict(list)
rfid_total_waits = defaultdict(list)
paper_total_waits = defaultdict(list)

def get_current_time(elapsed_seconds):
    start_time = datetime.strptime('06:30:00', '%H:%M:%S')
    current_time = start_time + timedelta(seconds=elapsed_seconds)
    formatted_current_time = current_time.strftime('%H:%M:%S')
    return formatted_current_time

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
    
def generate_scan_time(card_type):
    if card_type == 'RFID':
        return random.uniform(RFID_SCAN_TIME_MIN, RFID_SCAN_TIME_MAX)
    elif card_type == 'paper':
        return random.uniform(PAPER_SCAN_TIME_MIN, PAPER_SCAN_TIME_MAX)
    

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

    
def generate_error_correction_time(card_type):
    if card_type == 'RFID':
        if RFID_EMPS_PER_LINE == 1:
            ERROR_CORRECTION_TIME = max(0, random.normalvariate(15, 5))
        elif 0.5 < RFID_EMPS_PER_LINE < 1:
            ERROR_CORRECTION_TIME = max(0, random.normalvariate(20, 5))
        else:
            ERROR_CORRECTION_TIME = max(0, random.normalvariate(30, 5))
    elif card_type == 'paper':
        ERROR_CORRECTION_TIME = max(0, random.normalvariate(10, 2))
        
    return ERROR_CORRECTION_TIME

event_log = []

# Hàm sinh trường hợp lỗi của mỗi loại thẻ dựa trên xác suất
def is_error(card_type):
    if card_type == 'RFID':
        # Sinh ra một số ngẫu nhiên từ 0-1
        # Nếu số này nhỏ hơn xác suất xuất hiện lỗi thẻ từ, trả về True và đây là trường hợp có phát sinh lỗi
        # Nếu số này lớn hơn xác suất xuất hiện lỗi thẻ từ, trả về False và đây là trường hợp không phát sinh lỗi
        return random.random() <= ERROR_RATE_RFID
    elif card_type == 'paper':
        # Tương tự cho thẻ giấy
        return random.random() <= ERROR_RATE_PAPER

# Hàm chọn hàng có ít người đang xếp hàng nhất
# Input là các hàng có thể chọn dựa trên loại thẻ    
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
    # Trả về hàng ngắn nhất trong số tất cả các hàng được chọn và một index cho phần UI sử dụng
    return lines[idx_of_shortest], idx_of_shortest+1


# Hàm ghi nhận lại các events
def logging_events(person, card_type, gate_line, traffic_status, queue_begin, queue_end, scan_begin, scan_end, error_appearance, correction_begin, error_correction_time, correction_end):
    queue_duration = queue_end - queue_begin
    scan_duration = scan_end - scan_begin
    error_correcting_duration = error_correction_time
    wait = queue_duration + scan_duration + error_correcting_duration
    
    event_log.append({"event": "WAITING TO BE SCANNED", "person": f"id_{person}", "selected line": f"{card_type}_{gate_line}", "traffic status": traffic_status, "begin time": get_current_time(queue_begin), "end time": get_current_time(queue_end), "duration": round(queue_duration, 2)})
    event_log.append({"event": "SCAN TICKET", "person": f"id_{person}", "selected line": f"{card_type}_{gate_line}", "traffic status": traffic_status, "begin time": get_current_time(scan_begin), "end time": get_current_time(scan_end), "duration": round(scan_duration, 2)})
    if error_appearance:
        event_log.append({"event": "ERROR OCCURENCE AND CORRECTION", "person": f"id_{person}", "selected line": f"{card_type}_{gate_line}", "traffic status": traffic_status, "begin time": get_current_time(correction_begin), "end time": get_current_time(correction_end), "duration": round(error_correcting_duration, 2)})        
            
    wait_end_mark = scan_end if error_appearance==False else correction_end
    total_waits[int(wait_end_mark)].append(wait)
    if card_type == 'RFID':
            rfid_total_waits[int(wait_end_mark)].append(wait)
    else:
        paper_total_waits[int(wait_end_mark)].append(wait)

# QUÁ TRÌNH TẠO PHƯƠNG TIỆN ĐẾN
def vehicle_arrival(env, rfid_gate_lines, paper_gate_lines):
    # Khởi tạo id cho xe đầu tiên
    next_person_id = 0
    
    # Bắt đầu vòng lặp
    while True:
        # Xác định loại thẻ xe này sẽ sử dụng
        card_type = random.choices(['RFID', 'paper'], weights=[RFID_SELECTION_RATE, 1 - RFID_SELECTION_RATE])[0]
        
        # Nếu là sử dụng thẻ từ, các cổng sẽ là cổng từ
        if card_type == 'RFID':
            gate_lines = rfid_gate_lines
            
        # Nếu là sử dụng thẻ giấy, các cổng sẽ là cổng giấy
        else:
            gate_lines = paper_gate_lines
            
        # Tạo ra phương tiện với id, lựa chọn loại thẻ và các cổng họ có thể đi như đã khai báo ở trên
        # Xác định thời gian cần chờ để phương tiện này xuất hiện và trạng thái giao thông tương ứng
        next_arrival, traffic_status = generate_arrival_time(env)
        yield env.timeout(next_arrival) # Ghi nhận thời gian đã trôi qua trong giả lập

        # Phương tiện này sau đó sẽ tham gia quá trình sử dụng dịch vụ kiểm vé
        env.process(using_gate(env, next_person_id, gate_lines, card_type, traffic_status))
        
        # Tạo id cho phương tiện tiếp theo
        next_person_id += 1

# QUÁ TRÌNH SỬ DỤNG DỊCH VỤ KIỂM VÉ
def using_gate(env, person_id, gate_lines, card_type, traffic_status):
    
    # Ghi nhận thời điểm xếp hàng
    queue_begin = env.now
    # Chọn cổng có ít người đang xếp hàng ở đó nhất
    gate_line = pick_shortest(gate_lines)[0]
    non_zero_gate_idx = pick_shortest(gate_lines)[1]
    
    # Thêm phương tiện vào cổng chờ trong giao diện
    if card_type == 'RFID':
        graphic_rfid_gates.add_to_line(non_zero_gate_idx)
    else:
        graphic_paper_gates.add_to_line(non_zero_gate_idx)
        
    # Tạo request được xếp hàng đến lượt "sử dụng tài nguyên (resource)"
    # Tài nguyên ở đây là cổng được chọn
    with gate_line.request() as req:
        yield req
        # Đến lượt sử dụng = đã xếp hàng xong
        # Vẫn còn nắm giữ "tài nguyên", tức là vẫn còn ở trong cổng này chưa ra khỏi
        # Ghi nhận thời điểm xếp hàng xong
        queue_end = env.now
        

    # Bắt đầu "sử dụng tài nguyên (resource)"
    ### SCANNING
        # Sau khi xếp hàng xong, thẻ xe của phương tiện sẽ được quét
        # Ghi nhận thời điểm bắt đầu quét
        scan_begin = env.now
        # Sinh thời gian quét thẻ cho phương tiện này
        scan_time = generate_scan_time(card_type=card_type)
        # Ghi nhận thời gian trôi qua trong giả lập
        yield env.timeout(scan_time)
        # Ghi nhận thời điểm quét thẻ xong
        scan_end = env.now
        
        
    ### ERROR
        # Xác định có xảy ra lỗi hay không
        error_appearance = is_error(card_type)
        # Ghi nhận thời điểm bắt đầu kiểm lỗi
        correction_begin = env.now
        # Nếu có lỗi phát sinh
        if error_appearance:
            # Sinh thời gian sửa lỗi
            error_correction_time = generate_error_correction_time(card_type)
        # Nếu không có lỗi phát sinh
        else:
            # Thời gian kiểm lỗi bằng 0
            error_correction_time = 0
        
        # Ghi nhận thời gian trôi qua trong giả lập
        yield env.timeout(error_correction_time)
        # Ghi nhận thời điểm sửa lỗi xong
        correction_end = env.now
        
    # HOÀN THÀNH VÀ XÓA XE KHỎI HÀNG CHỜ TRÊN GIAO DIỆN
        if card_type == "RFID":
            graphic_rfid_gates.remove_from_line(non_zero_gate_idx)
        else:
            graphic_paper_gates.remove_from_line(non_zero_gate_idx)
            
    # LƯU CÁC SỰ KIỆN    
        logging_events(person_id, card_type, non_zero_gate_idx, traffic_status, queue_begin, queue_end, scan_begin, scan_end, error_appearance, correction_begin, error_correction_time, correction_end)
        





def avg_wait(raw_waits):
    waits = [w for i in raw_waits.values() for w in i]
    avg_wait_time = round(np.mean(waits), 1) if waits else 0
    if avg_wait_time >= 60:
            avg_wait_time = round(avg_wait_time / 60, 1)
            wait_time_string = str(avg_wait_time) + 'm'
    else:
        wait_time_string = str(avg_wait_time) + 's'
    return wait_time_string

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
canvas_width = 1450
canvas_height = 400
canvas = tk.Canvas(root, width = canvas_width, height = canvas_height, bg = "white")
canvas.pack(side=tk.TOP, expand = False)

#plot
f = plt.Figure(figsize=(2, 2), dpi=72)
# f.subplots_adjust(hspace=0.5, wspace=0.5)
a1 = f.add_subplot(222)
a1.plot()
a2 = f.add_subplot(224)
a2.plot()
a3 = f.add_subplot(121)
a3.plot()


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

def graphic_gates(canvas, x_top, y_top):
    
    # Position the RFID gate above the PAPER gate
    graphic_rfid_gates = QueueGraphics(r"E:\COLLEGE\DA w Python\parking_system_simulation\anh\xe xanh.png", 70, r'anh\cong xanh.png', RFID_GATE_LINES, canvas, x_top, y_top)
    graphic_paper_gates = QueueGraphics(r"E:\COLLEGE\DA w Python\parking_system_simulation\anh\xe cam.png", 70, r'anh\cong cam.png', PAPER_GATE_LINES, canvas, x_top, y_top + 120)

    return graphic_rfid_gates, graphic_paper_gates


class ClockAndData:
    def __init__(self, canvas, x1, y1, x2, y2, time):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        self.train = canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, fill="#fff")
        self.time = canvas.create_text(self.x1 + 10, self.y1 + 10, text = "Current Time = "+ get_current_time(time), anchor = tk.NW)
        
        wait_time_string = avg_wait(total_waits)
        
            
        self.overall_avg_wait = canvas.create_text(self.x1 + 10, self.y1 + 35, text = "Avg. Wait Time = "+ wait_time_string, anchor = tk.NW)
        self.traffic = canvas.create_text(self.x1 + 10, self.y1 + 50, text = "Traffic Status = "+ check_traffic_status(get_current_time(time)), anchor = tk.NW)
        self.canvas.update()

    def tick(self, time):
        self.canvas.delete(self.time)
        self.canvas.delete(self.overall_avg_wait)
        self.canvas.delete(self.traffic)

        wait_time_string = avg_wait(total_waits)
            
        self.time = canvas.create_text(self.x1 + 10, self.y1 + 10, text = "Current Time = "+ get_current_time(time), anchor = tk.NW)
        self.overall_avg_wait = canvas.create_text(self.x1 + 10, self.y1 + 30, text = "Avg. Wait Time = "+ wait_time_string, anchor = tk.NW)
        self.traffic = canvas.create_text(self.x1 + 10, self.y1 + 50, text = "Traffic Status = "+ check_traffic_status(get_current_time(time)), anchor = tk.NW)
        
        a1.cla()
        a1.set_title(f"RFID | Avg. wait time: {avg_wait(rfid_total_waits)}")
        a1.set_ylabel("Avg. Wait Time (seconds)")
        a1.step([ t for (t, waits) in rfid_total_waits.items() ], [ np.mean(waits) for (t, waits) in rfid_total_waits.items() ])
        
        
        a2.cla()
        a2.set_title(f"PAPER | Avg. wait time: {avg_wait(paper_total_waits)}")
        a2.set_xlabel("Time")
        a2.step([ t for (t, waits) in paper_total_waits.items() ], [ np.mean(waits) for (t, waits) in paper_total_waits.items() ])
        

        from collections import OrderedDict
        def moving_average(totals_dict, step):
            mean_waits_dict = {}
            moving_averages = {}
            for key, waits in totals_dict.items():
                if len(waits) == 0:
                    key_mean = 0
                else:
                    key_mean = np.mean(waits)
                mean_waits_dict[key] = key_mean
            for i, key in enumerate(mean_waits_dict.keys()):
                if i >= step - 1:
                    window = list(mean_waits_dict.values())[i - step + 1:i + 1]
                    moving_averages[key] = np.mean(window)
            for key in totals_dict.keys():
                if key not in moving_averages:
                    moving_averages[key] = 0
            moving_averages_sorted = OrderedDict(sorted(moving_averages.items()))
            return moving_averages_sorted

        step = 10
        moving_average_rfid = moving_average(rfid_total_waits, step=step)
        moving_average_paper = moving_average(paper_total_waits, step=step)

        a3.cla()
        a3.set_title(f"Moving Average Comparison, step={step}")
        a3.set_xlabel("Time")
        a3.set_ylabel("Avg. Wait Time (seconds)")

        a3.step([ t for (t, moving_avg) in moving_average_rfid.items() ], [ moving_avg for (t, moving_avg) in moving_average_rfid.items() ], label='RFID')
        a3.step([ t for (t, moving_avg) in moving_average_paper.items() ], [ moving_avg for (t, moving_avg) in moving_average_paper.items() ], label='PAPER')

        a3.legend()


        data_plot.draw()
        self.canvas.update()
        
graphic_rfid_gates, graphic_paper_gates = graphic_gates(canvas, 340, 15)

clock_and_data_width = 190
clock_and_data_height = 80
x1 = (canvas_width / 2) - (clock_and_data_width / 2)
y1 = canvas_height - clock_and_data_height - 10

clock = ClockAndData(canvas, 
                     x1=x1, y1=y1, 
                     x2=x1+clock_and_data_width, 
                     y2=y1+clock_and_data_height, 
                     time=0)
env = simpy.Environment()

def create_clock(env):
    """
        This generator is meant to be used as a SimPy event to update the clock
        and the data in the UI
    """
    
    while True:
        yield env.timeout(70)
        clock.tick(env.now)
        

        
        
env = simpy.Environment()

rfid_gate_lines = [simpy.Resource(env, capacity=RFID_EMPS_PER_LINE) for _ in range(RFID_GATE_LINES)]
paper_gate_lines = [simpy.Resource(env, capacity=PAPER_EMPS_PER_LINE) for _ in range(PAPER_GATE_LINES)]
all_gate_lines = [rfid_gate_lines, paper_gate_lines]


env.process(vehicle_arrival(env, rfid_gate_lines, paper_gate_lines))
env.process(create_clock(env))

# Từ 6:30 đến 20:30 là 14 tiếng đồng hồ
hours = 14
# Đổi từ giờ sang giây
seconds = hours*60*60
env.run(until=seconds)
root.mainloop()



# Writing data to a JSON file
with open('anh/events.json', 'w') as outfile:
    # input_string = f"""RFID Gates: {RFID_GATE_LINES} | Paper Gates: {PAPER_GATE_LINES} || Paper Employees per Line: {PAPER_EMPS_PER_LINE} | RFID Employees per Line: {RFID_EMPS_PER_LINE}"""
    # config_string = f"""RFID Selection Rate: {RFID_SELECTION_RATE} || RFID Scan Time (Min): {RFID_SCAN_TIME_MIN}| RFID Scan Time (Max): {RFID_SCAN_TIME_MAX} || Paper Scan Time (Min): {PAPER_SCAN_TIME_MIN}| Paper Scan Time (Max): {PAPER_SCAN_TIME_MAX} || Join Rate High Mean: {JOIN_RATE_HIGH_MEAN}| Join Rate High Std: {JOIN_RATE_HIGH_STD} || Join Rate Avg Mean: {JOIN_RATE_AVG_MEAN}| Join Rate Avg Std: {JOIN_RATE_AVG_STD} || Join Rate Low Mean: {JOIN_RATE_LOW_MEAN}| Join Rate Low Std: {JOIN_RATE_LOW_STD} || Error Rate RFID: {ERROR_RATE_RFID}| Error Rate Paper: {ERROR_RATE_PAPER}"""
    json.dump({"RFID GATES": RFID_GATE_LINES, 
               "PAPER GATES": PAPER_GATE_LINES,
               "RFID EMPLOYEES": int(RFID_GATE_LINES * RFID_EMPS_PER_LINE),
               "PAPER EMPLOYEES": PAPER_GATE_LINES * PAPER_EMPS_PER_LINE,
               "events": event_log}, outfile, indent=4)