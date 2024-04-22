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
RFID_GATE_LINES = 3
PAPER_GATE_LINES = 2

PAPER_EMPS_PER_LINE = 1
RFID_EMPS_PER_LINE = 2/3

######################
### CONFIGURATIONS ###
######################
RFID_SELECTION_RATE = 0.6

RFID_SCAN_TIME_MIN = 9
RFID_SCAN_TIME_MAX = 13
PAPER_SCAN_TIME_MIN = 10
PAPER_SCAN_TIME_MAX = 17

JOIN_RATE_HIGH_MEAN = 3
JOIN_RATE_HIGH_STD = 0.5

JOIN_RATE_AVG_MEAN = 30
JOIN_RATE_AVG_STD = 1

JOIN_RATE_LOW_MEAN = 60
JOIN_RATE_LOW_STD = 10

ERROR_RATE_RFID = 0.01
ERROR_RATE_PAPER = 0.035
# Creating the 'images' directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Khởi tạo dictionary chứa list tổng thời gian chờ tại mỗi giây diễn ra trong giả lập của tất cả các loại cổng
total_waits = defaultdict(list)
# Khởi tạo dictionary chứa list tổng thời gian chờ tại mỗi giây diễn ra trong giả lập của cổng thẻ từ
rfid_total_waits = defaultdict(list)
# Khởi tạo dictionary chứa list tổng thời gian chờ tại mỗi giây diễn ra trong giả lập của cổng thẻ giấy
paper_total_waits = defaultdict(list)

# Hàm xác định thời điểm (giờ:phút:giây) hiện tại dựa trên số giây đã trôi qua trong giả lập
def get_current_time(elapsed_seconds):
    start_time = datetime.strptime('06:30:00', '%H:%M:%S')
    current_time = start_time + timedelta(seconds=elapsed_seconds)
    formatted_current_time = current_time.strftime('%H:%M:%S')
    return formatted_current_time

# Xác định trạng thái giao thông dựa trên thời điểm hiện tại
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
    
# Sinh thời gian quét thẻ dựa trên mỗi loại cổng
def generate_scan_time(card_type):
    if card_type == 'RFID':
        return random.uniform(RFID_SCAN_TIME_MIN, RFID_SCAN_TIME_MAX)
    elif card_type == 'paper':
        return random.uniform(PAPER_SCAN_TIME_MIN, PAPER_SCAN_TIME_MAX)
    
# Sinh thời gian xe đến dựa vào tình trạng giao thông hiện tại
def generate_arrival_time(env):
    current_time = get_current_time(env.now)
    traffic_status = check_traffic_status(current_time)
    if traffic_status == 'high':
        arrival_time = max(0, random.normalvariate(JOIN_RATE_HIGH_MEAN, JOIN_RATE_HIGH_STD))
    elif traffic_status == 'low':
        arrival_time = max(0, random.normalvariate(JOIN_RATE_LOW_MEAN, JOIN_RATE_LOW_STD))
    else: # NORMAL
        arrival_time = max(0, random.normalvariate(JOIN_RATE_AVG_MEAN, JOIN_RATE_AVG_STD))        
    return arrival_time, traffic_status

# Sinh thời gian cần thiết để sửa lỗi nếu phát sinh dựa trên loại thẻ
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

# Tạo list rỗng để chứa các ghi nhận event
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
        
# Chuyển đổi giây thành phút với hậu tố là s nếu là giây và m nếu là phút
# Nếu dưới 60s thì không chuyển thành phút và giữ nguyên hậu tố s
# Nếu trên 60s thì chuyển thành phút và chuyển đổi thành hậu tố m
def seconds_to_minutes_string(seconds):
    if math.isnan(seconds):
        return 0
    if seconds >= 60:
        minutes = int(round(seconds / 60, 0))
        time_str = str(minutes) + 'm'
    else:
        seconds = int(round(seconds, 0))
        time_str = str(seconds) + 's'
    return time_str

# Tính toán thời gian chờ trung bình của các phương tiện tại mỗi thời điểm trong giả lập
# Mỗi thời điểm có một list thời gian chờ, do mỗi thời điểm có thể phát sinh nhiều phương tiện chờ
def avg_wait(raw_waits):
    waits = [w for i in raw_waits.values() for w in i]
    avg_wait_time = round(np.mean(waits), 1) if waits else 0
    wait_time_string = seconds_to_minutes_string(avg_wait_time)
    return wait_time_string

root = tk.Tk()
# size 
root.geometry('1500x800+10+0')
# chiều ngang, chiều cao, khoảng cách với mép trái, khoảng cách với mép trên
# tilte 
root.title('MÔ PHỎNG VÀ TỐI ƯU HÓA BÃI ĐỖ XE TRƯỜNG ĐẠI HỌC') 
#icon
root.iconbitmap(r'images\due.ico')
#background
root.configure(bg='#fff')
#label
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, expand = False)
label = Label(root,text="MÔ PHỎNG VÀ PHÂN TÍCH HỆ THỐNG KIỂM VÉ BÃI ĐỖ XE", font =('Arial',26),bg='#223442',fg='white', height=2)
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

# Tạo class cho các cổng chờ
class QueueGraphics:
    text_height = 50
    icon_top_margin = 5
    
    def __init__(self, vehicle_file, vehicle_icon_width, gate_image_file, gate_line_list, canvas, x_top, y_top, emp_icon_file,num_emps):
        # File chứa icon xe 
        self.vehicle_file = vehicle_file
        # Chiều rộng icon xe
        self.vehicle_icon_width = vehicle_icon_width
        # File chứa icon cổng 
        self.gate_image_file = gate_image_file
        # List các cổng của từng loại
        self.gate_line_list = gate_line_list
        # Những graphics này sẽ được hiển thị trên canvas cho việc trình bày giả lập các luồng xe
        self.canvas = canvas
        # Tọa độ x trên cùng
        self.x_top = x_top
        # Tọa độ y trên cùng
        self.y_top = y_top
        # File chứa icon nhân viên
        self.emp_icon_file = emp_icon_file
        # Số nhân viên
        self.num_emps = num_emps
        self.rectangles = []

        # Tải ảnh lên TK và lưu vào các biến tương ứng
        self.gate_image = tk.PhotoImage(file=self.gate_image_file)
        self.vehicle_image = tk.PhotoImage(file=self.vehicle_file)
        self.emp_icon_image = tk.PhotoImage(file=self.emp_icon_file)
        # Tạo dictionary lưu các icon phương tiện của từng cổng
        self.vehicle_icons = defaultdict(lambda: [])
        
        # Tạo các hình ảnh cổng
        for i in range(gate_line_list):
            canvas.create_image(x_top, y_top + (i * 1.25 * self.text_height), anchor=tk.NW, image=self.gate_image)
        self.canvas.update()
        
        # Tạo các hình ảnh nhân viên
        for i in range(num_emps):
            canvas.create_image(x_top - 35, y_top + (i * 1.25 * self.text_height), anchor=tk.NW, image=self.emp_icon_image)
        self.canvas.update()

    # Thêm icon xe máy vào hàng dựa trên chiều dài của hàng hiện tại
    def add_to_line(self, gate_line):
        count = len(self.vehicle_icons[gate_line])
        x = self.x_top + 100 + (count * self.vehicle_icon_width)
        y = self.y_top + ((gate_line - 1) * 1.25 * self.text_height) + self.icon_top_margin
        self.vehicle_icons[gate_line].append(
                self.canvas.create_image(x, y, anchor=tk.NW, image=self.vehicle_image)
        )
        self.canvas.update()
    
    # Bỏ icon xe máy khỏi hàng
    def remove_from_line(self, gate_line):
        if len(self.vehicle_icons[gate_line]) == 0: return
        to_del = self.vehicle_icons[gate_line].pop()
        self.canvas.delete(to_del)
        self.canvas.update()

# Tạo các cổng và số nhân viên tương ứng
def graphic_gates(canvas, x_top, y_top):
    rfid_emp_num = int(round(RFID_GATE_LINES * RFID_EMPS_PER_LINE, 0))
    paper_emp_num =  int(round(PAPER_GATE_LINES * PAPER_EMPS_PER_LINE, 0))
    
    # Position the RFID gate above the PAPER gate
    graphic_rfid_gates = QueueGraphics(r"images\xe xanh.png", 70, r'images\cong xanh.png', RFID_GATE_LINES, canvas, x_top, y_top, r'images\nguoi xanh.png',rfid_emp_num)
    graphic_paper_gates = QueueGraphics(r"images\xe cam.png", 70, r'images\cong cam.png', PAPER_GATE_LINES, canvas, x_top, y_top * RFID_GATE_LINES * 7.5, r'images\nguoi cam.png',paper_emp_num)

    return graphic_rfid_gates, graphic_paper_gates


# Tạo class cho bảng thông tin chứa đồng hồ đếm giờ và thống kê thời gian chờ trung bình cho đến hiện tại trên UI
class ClockAndData:
    def __init__(self, canvas, x1, y1, x2, y2, time):
        # Khai báo tọa độ của bảng
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        
        # Display thời gian hiện tại lên bảng này
        self.time = canvas.create_text(self.x1 + 10, self.y1, text = "Current Time = "+ get_current_time(time), anchor = tk.NW, font=('Helvetica', 15), fill='#000080')
        
        # Display thời gian chờ trung bình
        wait_time_string = avg_wait(total_waits)
        self.overall_avg_wait = canvas.create_text(self.x1 + 10, self.y1 + 35, text = "Avg. Wait Time = "+ wait_time_string, anchor = tk.NW, font=('Helvetica', 12), fill='#000080')
        
        # Display tình trạng giao thông và đổi màu tương sng
        traffic_status = check_traffic_status(get_current_time(time))
        traffic_status_color = {'high': 'red', 'avg': 'orange', 'low': 'blue'}
        self.traffic = canvas.create_text(self.x1 + 10, self.y1 + 68, text = "Traffic Status = "+ traffic_status, anchor = tk.NW, fill=traffic_status_color[traffic_status])
        self.traffic_rectangle = canvas.create_rectangle(self.x1 + 10, self.y1 + 100, self.x2-70, self.y2+15, fill=traffic_status_color[traffic_status], outline='')
        
        # CẬP NHẬT lại canvas
        self.canvas.update()

    # Qua mỗi nhịp thời gian sẽ xóa thông tin cũ và hiển thị thông tin mới
    def tick(self, time):
        # Xóa thông tin cũ
        self.canvas.delete(self.time)
        self.canvas.delete(self.overall_avg_wait)
        self.canvas.delete(self.traffic)

        # Display thời gian hiện tại lên bảng này
        self.time = canvas.create_text(self.x1 + 10, self.y1, text = "Current Time = "+ get_current_time(time), anchor = tk.NW, font=('Helvetica', 15), fill='#000080')
        
        # Display thời gian chờ trung bình
        wait_time_string = avg_wait(total_waits)
        self.overall_avg_wait = canvas.create_text(self.x1 + 10, self.y1 + 35, text = "Avg. Wait Time = "+ wait_time_string, anchor = tk.NW, font=('Helvetica', 12), fill='#000080')
        
        # Display tình trạng giao thông và đổi màu tương sng
        traffic_status = check_traffic_status(get_current_time(time))
        traffic_status_color = {'high': 'red', 'avg': 'orange', 'low': 'blue'}
        self.traffic = canvas.create_text(self.x1 + 10, self.y1 + 68, text = "Traffic Status = "+ traffic_status, anchor = tk.NW, fill=traffic_status_color[traffic_status])
        self.traffic_rectangle = canvas.create_rectangle(self.x1 + 10, self.y1 + 100, self.x2-70, self.y2+15, fill=traffic_status_color[traffic_status], outline='')
        
        # HIỂN THỊ CÁC PLOT
        a1.cla()
        a1.set_title(f"RFID | Avg. wait time: {avg_wait(rfid_total_waits)}")
        a1.set_ylabel("Wait Time (seconds)")
        # Step plot, cập nhật theo thời gian
        a1.step([ t for (t, waits) in rfid_total_waits.items() ], [ np.mean(waits) for (t, waits) in rfid_total_waits.items() ], color='#4A9658')
        
        
        a2.cla()
        a2.set_title(f"PAPER | Avg. wait time: {avg_wait(paper_total_waits)}")
        a2.set_xlabel("Time")
        # Step plot, cập nhật theo thời gian
        a2.step([ t for (t, waits) in paper_total_waits.items() ], [ np.mean(waits) for (t, waits) in paper_total_waits.items() ], color='#FF914C')
        
        # Tính trung bình trượt ở thời điểm hiện tại
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
        
        # Define số step cho trung bình trượt
        step = 10
        # Tính trung bình trượt cho từng loại cổng
        moving_average_rfid = moving_average(rfid_total_waits, step=step)
        moving_average_paper = moving_average(paper_total_waits, step=step)


        a3.cla()
        # Chuyển thành dạng list để dễ dàng truyền vào title của biểu đồ a3 và cập nhật qua thời gian
        current_moving_avg_rfid = list(moving_average_rfid.values())[-1] if moving_average_rfid else 0
        current_moving_avg_paper = list(moving_average_paper.values())[-1] if moving_average_paper else 0

        a3.set_title(f"Moving Average, step={step}\nRFID: {seconds_to_minutes_string(current_moving_avg_rfid)}\nPAPER: {seconds_to_minutes_string(current_moving_avg_paper)}")
        a3.set_xlabel("Time")
        a3.set_ylabel("Avg. Wait Time (seconds)")

        # Step plot, cập nhật theo thời gian
        # Plot chứa 2 đường, mỗi đường đại diện cho một cổng
        a3.step([ t for (t, moving_avg) in moving_average_rfid.items() ], [ moving_avg for (t, moving_avg) in moving_average_rfid.items() ], label='RFID', color='#4A9658')
        a3.step([ t for (t, moving_avg) in moving_average_paper.items() ], [ moving_avg for (t, moving_avg) in moving_average_paper.items() ], label='PAPER', color='#FF914C')

        a3.legend()


        data_plot.draw()
        self.canvas.update()
        
# Nhận các thông tin về hình ảnh cổng từ và cổng giấy trên giao diện
graphic_rfid_gates, graphic_paper_gates = graphic_gates(canvas, 340, 10)

# Tính toán tọa độ để hiển thị bảng đồng hồ và data
clock_and_data_width = 190
clock_and_data_height = 70
x1 = 10
# Bảng ở giữa theo chiều dọc canvas
y1 = (canvas_height / 2) - (clock_and_data_height / 2)
# Tọa độ trục hoành - phía bên phải của bảng
x2=x1+clock_and_data_width
# Tọa độ trục tung - phía dưới của bảng
y2=y1+clock_and_data_height
clock = ClockAndData(canvas, 
                     x1=x1, 
                     y1=y1, 
                     x2=x2, 
                     y2=y2, 
                     time=0)

def create_clock(env):
    """
        Tạo đồng hồ để điều chỉnh tốc độ của giả lập khi hiển thị lên UI.
        Bằng cách điều chỉnh kích thước mỗi nhịp thời gian trong giả lập.
        Với mỗi giây trôi qua ngoài đời thật, thì bao nhiêu giây trôi qua trong giả lập.
        secs_passed_in_sim_per_real_sec sẽ là một nhịp giả lập.
    """
    secs_passed_in_sim_per_real_sec = int(input('Seconds passed in simulation per real second: '))
    
    while True:
        yield env.timeout(secs_passed_in_sim_per_real_sec)
        clock.tick(env.now)
        

        
# TẠO MÔI TRƯỜNG
env = simpy.Environment()

# Tạo các tài nguyên của môi trường
# Mỗi tài nguyên chỉ được một xe sử dụng (scan, error) cùng lúc, còn những xe khác phải chờ
rfid_gate_lines = [simpy.Resource(env, capacity=1) for _ in range(RFID_GATE_LINES)]
paper_gate_lines = [simpy.Resource(env, capacity=1) for _ in range(PAPER_GATE_LINES)]
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
with open(r'output\GUI_events.json', 'w') as outfile:
    json.dump({"RFID GATES": RFID_GATE_LINES, 
               "PAPER GATES": PAPER_GATE_LINES,
               "RFID EMPLOYEES": int(RFID_GATE_LINES * RFID_EMPS_PER_LINE),
               "PAPER EMPLOYEES": PAPER_GATE_LINES * PAPER_EMPS_PER_LINE,
               "events": event_log}, outfile, indent=4)