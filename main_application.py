print("Please wait ...")
import tkinter as tk
import customtkinter as ck
import mediapipe as mp
import cv2
from PIL import Image, ImageTk 
import numpy as np
import pandas as pd
from tkinter import DISABLED, HORIZONTAL, ttk
from tkinter.filedialog import askopenfilename
import datetime
import traceback
from RangeSlider.RangeSlider import RangeSliderH
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter.messagebox import askokcancel
import os
import webbrowser
import time
import matplotlib.transforms as mtransforms
from math import pi
import random
from numpy.linalg import norm 
import math
from scipy.signal import find_peaks
import threading

print("Welcome!")

ck.set_appearance_mode('light')
ck.set_default_color_theme('blue')

root = ck.CTk()
root.overrideredirect(False)
root.overrideredirect(False)
root.attributes('-fullscreen',True)
root.title("User dashboard")

#####################################################################################################################################################################
#### global function ####################################################################################################################################################
#####################################################################################################################################################################

def image_load(im_name, resize_size):
    path = "./assets/" + im_name
    im = ImageTk.PhotoImage(image = Image.open(path).resize(resize_size, Image.ANTIALIAS))
    return im
def bar_figure_gen_func(fig_size, facecolor, x, y, color = 'b', width = 0.25, ylabel = "intensity"):
    figure = plt.Figure(figsize=fig_size, dpi=100)
    figure.patch.set_facecolor(facecolor)
    figure.tight_layout()
    ax = figure.add_subplot(111)
    #ax.tick_params(left = True, right = False , labelleft = True ,
    #                labelbottom = False, bottom = False)
    ax.bar(x=x,height=y, color = color, width = width)
    ax.set_ylabel(ylabel)
    return figure
def line_bar_figure_gen_func(fig_size, facecolor, x, y, color = 'b', ylabel = "intensity"):
    figure = plt.Figure(figsize=fig_size, dpi=100)
    figure.patch.set_facecolor(facecolor)
    figure.tight_layout()
    ax = figure.add_subplot(111)
    #ax.tick_params(left = True, right = False , labelleft = True ,
    #                labelbottom = False, bottom = False)
    ax.bar(x=x,height=y, color = color, width = 0.5)
    ax.plot(x, y*1.5)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(linestyle='--', linewidth=0.5)
    return figure
def radar_chart_func(fig_size, facecolor):
        df = pd.DataFrame({
        'group': ['A','B','C','D'],
        'var1': [38, 1.5, 30, 4],
        'var2': [29, 10, 9, 34],
        'var3': [8, 39, 23, 24],
        'var4': [7, 31, 33, 14],
        'var5': [28, 15, 32, 14]
        })
        
        # number of variable
        categories=list(df)[1:]
        N = len(categories)
        
        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values=df.loc[0].drop('group').values.flatten().tolist()
        values += values[:1]
        values
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the spider plot
        #figure = plt.Figure()
        #ax = figure.add_subplot(111,polar=True)
        plt.switch_backend('agg')
        ax = plt.subplot(111, polar=True)
        # Draw one axe per variable + add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([10,20,30])
        ax.set_yticklabels(["10","20","30"])
        ax.set_ylim(0,40)
        
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
        ax.figure.set_size_inches(fig_size[0], fig_size[1])
        figure = ax.get_figure()

        return figure
def circle_progress_func(complete_perc):
    fig, ax = plt.subplots(figsize=(0.5, 0.5))
    wedgeprops = {'width':0.3, 'edgecolor':'black', 'linewidth':1}
    complete_perc = complete_perc
    left_perc = 100 - complete_perc
    ax.pie([complete_perc,left_perc], wedgeprops=wedgeprops, startangle=90, colors=['#5DADE2', '#515A5A'])
    ax.text(0, 0, "{}%".format(complete_perc), ha='center', va='center', fontsize=7)
    fig.tight_layout(pad=0)

    return fig
############################# OLST assessment ###################################################################
######################################################################################################
def lifting_count_func(right_tose, left_tose):
    lifting_foot_count = 0
    time = len(right_tose)
    right_tose_df = pd.DataFrame(right_tose, columns = ["x", "y", "z"])
    left_tose_df = pd.DataFrame(left_tose, columns = ["x", "y", "z"])

    right_tose_height = right_tose_df["y"]
    left_tose_height = left_tose_df["y"]


    if abs(right_tose_height.mean()) > abs(left_tose_height.mean()):
        landing_foot_height = left_tose_height
        threshold = ((landing_foot_height.mean())+ 3.5*(landing_foot_height.std()))
        lifting_foot_count = len(find_peaks(landing_foot_height, threshold = threshold)[0])
        
    if abs(right_tose_height.mean()) < abs(left_tose_height.mean()):
        landing_foot_height = right_tose_height
        threshold = ((landing_foot_height.mean())+ 3.5*(landing_foot_height.std()))
        lifting_foot_count = len(find_peaks(landing_foot_height, threshold = threshold)[0])

    #if lifting_foot_count > 12:
    #    lifting_foot_count = 12
    #if lifting_foot_count > 12:
    #    lifting_foot_count = 12

    return [lifting_foot_count, landing_foot_height, threshold]
def hand_off_count_func(right_hand, left_hand):
    hand_iliac_count = 0
    time = len(right_hand)
    right_hand_df = pd.DataFrame(right_hand, columns = ["x", "y", "z"])
    left_hand_df = pd.DataFrame(left_hand, columns = ["x", "y", "z"])

    R_L_distance = right_hand_df["x"] - left_hand_df["x"]
    hand_threshold = (R_L_distance.mean())+(R_L_distance.std())*3.5
    for i in range(time):
        try:
            if abs(R_L_distance[i]) < (hand_threshold) and  abs(R_L_distance[i+1]) > (hand_threshold): 
                hand_iliac_count = hand_iliac_count + 1
        except:
            continue
            
    #if hand_iliac_count > 12:
    #    hand_iliac_count = 12
        
    return [hand_iliac_count, R_L_distance, hand_threshold]
def leg_angle_count_func(right_pelvic, left_pelvic, right_knee, left_knee, strn):
    R_angle_all = []
    L_angle_all = []
    time = len(right_pelvic)
    COM = (np.array(right_pelvic) + np.array(left_pelvic))/2
    V_STRN_COM = COM[:,:-1] - np.array(strn)[:,:-1]
    V_RASI_knee = np.array(right_knee)[:,:-1] - np.array(right_pelvic)[:,:-1]
    V_LASI_knee = np.array(left_knee)[:,:-1] - np.array(left_pelvic)[:,:-1]

    #cos(angle) = dot(A,B) / (norm(A).*norm(B))

    for vector_num in range(V_STRN_COM.shape[0]):
        R_cos_value = np.dot(V_STRN_COM[vector_num], V_RASI_knee[vector_num]) / (norm(V_STRN_COM[vector_num], 2)*norm(V_RASI_knee[vector_num], 2))
        R_angle_all.append(math.acos(R_cos_value)* 180 / math.pi)

        L_cos_value = np.dot(V_STRN_COM[vector_num], V_LASI_knee[vector_num]) / (norm(V_STRN_COM[vector_num], 2)*norm(V_LASI_knee[vector_num], 2))
        L_angle_all.append(math.acos(L_cos_value)* 180 / math.pi)

    R_angle_all = np.array(R_angle_all)
    L_angle_all = np.array(L_angle_all)

    angle_count = 0
    if abs(np.mean(R_angle_all)) > abs(np.mean(L_angle_all)): ### right leg main moving leg
        main_moving_leg = R_angle_all
        for i in range(time-1):
            if abs(R_angle_all[i]- abs(np.mean(R_angle_all[:150]))) < 12 and  abs(R_angle_all[i+1] - abs(np.mean(R_angle_all[:150]))) > 12:
                angle_count = angle_count + 1

    elif abs(np.mean(R_angle_all)) < abs(np.mean(L_angle_all)): ### left leg main moving leg
        main_moving_leg = L_angle_all
        for i in range(time-1):
            if abs(L_angle_all[i]- abs(np.mean(L_angle_all[:150]))) < 12 and  abs(L_angle_all[i+1]- abs(np.mean(L_angle_all[:150]))) > 12:
                angle_count = angle_count + 1
    
    #if angle_count > 12:
    #    angle_count = 12
    
    return [angle_count, main_moving_leg]
def falling_count_func(right_tose, left_tose):
    falling_count = 0
    time = len(right_tose)
    right_tose_df = pd.DataFrame(right_tose, columns = ["x", "y", "z"])
    left_tose_df = pd.DataFrame(left_tose, columns = ["x", "y", "z"])

    R_L_distance = right_tose_df["y"] - left_tose_df["y"]
    for i in range(time):
        try:
            if abs(R_L_distance[i]) > 0.05 and abs(R_L_distance[i+1]) < 0.05 : 
                falling_count = falling_count + 1
        except:
            continue
    
    #if falling_count > 12:
    #    falling_count = 12
    
    return [falling_count, R_L_distance]
def remaining_out_count_func(video_array, landing_foot_height, threshold, main_moving_leg, R_L_distance, hand_threshold, R_L_distance_foot):
    remain_time = round(len(video_array)/12)
    time = len(R_L_distance_foot)
    out_position_count = 0
    t = np.array(range(time))
    period = t[::remain_time]
    for i in period:
        try:
            if (sum((landing_foot_height[i:i+remain_time]) > threshold)>=remain_time) or (sum((main_moving_leg[i:i+remain_time] - main_moving_leg[:500].mean()) > 12)>= remain_time) or (sum((R_L_distance[i:i+remain_time]) > hand_threshold) >= remain_time) or (sum(R_L_distance_foot[1:1+remain_time]< 0.05) >= remain_time):   
                out_position_count = out_position_count + 1
        except:
            continue
    #if out_position_count > 12:
    #    out_position_count = 12
    return out_position_count
def OLST_movement_assessment_func(result):
    ############################### OLST assessment ####################################################
    ######################################################################################################
    right_tose = []
    left_tose = []
    right_knee = []
    left_knee = []
    right_pelvic = []
    left_pelvic = []
    right_hand = []
    left_hand = []
    strn = []
    for time_frame in range(len(result)):
        try:
            results = result[time_frame]
            right_tose_coor = [results.pose_landmarks.landmark[32].x, results.pose_landmarks.landmark[32].y, results.pose_landmarks.landmark[32].z]
            left_tose_coor = [results.pose_landmarks.landmark[31].x, results.pose_landmarks.landmark[31].y, results.pose_landmarks.landmark[31].z]
            right_tose.append(right_tose_coor)
            left_tose.append(left_tose_coor)

            right_knee_coor = [results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[26].y, results.pose_landmarks.landmark[26].z]
            left_knee_coor = [results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[25].z]
            right_knee.append(right_knee_coor)
            left_knee.append(left_knee_coor)


            right_pelvic_coor = [results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[24].y, results.pose_landmarks.landmark[24].z]
            left_pelvic_coor = [results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[23].y, results.pose_landmarks.landmark[23].z]
            right_pelvic.append(right_pelvic_coor)
            left_pelvic.append(left_pelvic_coor)


            right_hand_coor = [results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y, results.pose_landmarks.landmark[20].z]
            left_hand_coor = [results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y, results.pose_landmarks.landmark[19].z]
            right_hand.append(right_hand_coor)
            left_hand.append(left_hand_coor)
            
            strn_coor = [(results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x)/2, (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y)/2, (results.pose_landmarks.landmark[11].z + results.pose_landmarks.landmark[12].z)/2]
            strn.append(strn_coor)
        except:
            continue
    lifting_foot_count, landing_foot_height, threshold = lifting_count_func(right_tose, left_tose)
    hand_iliac_count, R_L_distance, hand_threshold = hand_off_count_func(right_hand, left_hand)
    angle_count, main_moving_leg = leg_angle_count_func(right_pelvic, left_pelvic, right_knee, left_knee, strn)
    falling_count, R_L_distance_foot = falling_count_func(right_tose, left_tose)
    out_position_count = remaining_out_count_func(draw_video_array, landing_foot_height, threshold, main_moving_leg, R_L_distance, hand_threshold, R_L_distance_foot)
    all_count = lifting_foot_count + hand_iliac_count + angle_count + falling_count + out_position_count

    return lifting_foot_count, hand_iliac_count, angle_count, falling_count, out_position_count, all_count
#####################################################################################################################################################################
#### video record ####################################################################################################################################################
#####################################################################################################################################################################

def analysis():
    try:
        ########################################################   initialize analyzed information  ########################
        global analyzed_video
        global draw_video_array
        global result
        global current_analyzed_duration
        global analyzed_duration
        global loaded_file
        global upload_win 

        analyzed_duration = current_analyzed_duration

        draw_video_array = []
        result = []
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        analysis_progress_win = tk.Toplevel()
        analysis_progress_win.geometry("250x50")
        x = upload_win.winfo_x()
        y = upload_win.winfo_y()
        analysis_progress_win.geometry("+%d+%d" %(x+500,y+300))
        analysis_progress_win.configure(bg='#E5E8E8')
        analysis_progress_win.title("Analyzing...")
        analysis_progress = ttk.Progressbar(analysis_progress_win, orient=HORIZONTAL, length=100, mode = "determinate")
        analysis_progress.place(x=50, y=20)
        analysis_progress_Label = ck.CTkLabel(analysis_progress_win, height=40, width=50, bg_color = "white",text_font=("Arial", 15), text_color="black", padx=10)
        analysis_progress_Label.place(x=150, y=10)
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            print("mp apply")
            for frame in analyzed_video:
                
                if frame is None:
                    break
                image = frame
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                #cv2.imshow("test",cv2.flip(image, 0))
                cv2.waitKey(0)

                try:
                    draw_video_array.append(image)
                    result.append(results)
                except:
                    continue
                
                analysis_progress_Label.configure(text='{}%'.format(0)) 
                analysis_progress['value'] = analysis_progress['value'] + 0.6/len(analyzed_video)*180
                if analysis_progress['value'] >= 99:
                    analysis_progress['value'] = 99
                analysis_progress_Label.configure(text='{}%'.format(round(analysis_progress['value']))) 
                analysis_progress_win.update_idletasks()
        analysis_progress_Label.configure(text='{}%'.format(100))
        analysis_progress_win.destroy()
        lifting_foot_count, hand_iliac_count, angle_count, falling_count, out_position_count, all_count = OLST_movement_assessment_func(result)
        t = datetime.datetime.now()
        date = t.strftime("%Y-%m-%d")
        text_infor = ("Analyzed video info \n \n Date: {} \n Name: {} \n Resolution: {} \n Length: {}s \n lifting_foot_count: {} \n hand_iliac_count: {} \n angle_count: {} \n falling_count: {} \n out_position_count: {} \n all_count: {}")
        text_infor = text_infor.format(date, loaded_file[-1], "348 x 591", round(analyzed_duration, 1), lifting_foot_count, hand_iliac_count, angle_count, falling_count, out_position_count, all_count)
        analyzed_information = tk.Label(upload_win, text = text_infor)
        analyzed_information.configure(justify="left", borderwidth=2, relief="sunken", width= 25, padx = 5, anchor='w')
        analyzed_information.place(x=850, y=175)

        
    except Exception:
        traceback.print_exc()
        pass 

def video_cut(var, index, mode):
    try:
        global video_array
        global analyzed_video
        global duration
        global current_analyzed_duration
        global uploaded_lmain
        global start_label
        global end_label
        global right_value
        global left_value
        global upload_win

        S = left_value.get()
        if S == 0:
            S = 1
        E = right_value.get()
        
        int_frame_num = S/101
        int_frame = round(int_frame_num * len(video_array))
        int_time = round(int_frame_num * duration, 1)
        int_time = round(int_time, 1)

        end_frame_num = E/101
        end_frame = round(end_frame_num * len(video_array))
        end_time = round(end_frame_num * duration, 1)
        end_time = round(end_time, 1)

        current_analyzed_duration = end_time-int_time 

        analyzed_video = video_array[int_frame:end_frame]
        im = cv2.rotate(video_array[int_frame], cv2.ROTATE_180)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(image = Image.fromarray(im))
        uploaded_lmain.configure(image = image)
        uploaded_lmain.image = image

        start_label = tk.Label(upload_win, text=str(int_time)+"s", height = 1, width= 4, borderwidth=2, relief="sunken")
        start_label.place(x=50, y=725)
        end_label = tk.Label(upload_win,text=str(end_time)+"s", height = 1, width= 4, borderwidth=2, relief="sunken")
        end_label.place(x=805, y=725)
    except:
        pass
def recording_push_func():
    global recording_cap
    global recorded_video
    global video_array
    global analyzed_video
    global current_analyzed_duration
    global duration
    global loaded_file
    global length
    global recording_win
    global left_value
    global right_value
    global upload_win
    global uploaded_frame
    global uploaded_lmain

    if recorded_video is not None:
        left_value = tk.DoubleVar()
        right_value = tk.DoubleVar()

        upload_win = tk.Toplevel()
        upload_win.geometry("1100x800")
        upload_win.configure(bg='#E5E8E8')
        upload_win.title("Edit uploaded video")

        analysis_button = ck.CTkButton(upload_win, text='Analysis', command=analysis, height=40, width=150, text_font=("Arial", 15), text_color="black", fg_color="#ABB2B9", hover_color = "#566573", border_color = "black", border_width = 2)
        analysis_button.place(x=50, y=30)

        uploaded_frame = tk.Frame(upload_win, height=600, width=770, highlightbackground="black", highlightthickness=1)
        uploaded_frame.place(x=50, y=110)
        uploaded_frame.configure(bg = "white")

        uploaded_lmain = tk.Label(uploaded_frame) 
        uploaded_lmain.place(x=0, y=0) 

        slider = RangeSliderH(
                        upload_win,
                        [left_value, right_value],
                        show_value = False,
                        bgColor = "#E5E8E8",
                        min_val=0,
                        max_val=100,
                        Width=720, 
                        Height=65,
                        line_s_color = "black",
                        line_width = 3,
                        bar_color_inner = "white",
                        bar_color_outer = "black",
                        bar_radius = 5,
                        auto=True
                    )
        slider.place(x=85, y=700)
        left_value.trace_add('write', video_cut)
        right_value.trace_add('write', video_cut)

        slider_notification = tk.Label(upload_win, text="Edit the desire frame with this slider", bg='#E5E8E8', fg = "black")
        slider_notification.place(x = 840, y = 625)


        video_array = []
        for i in recorded_video:
            video_array.append(cv2.flip(i, 0))
        im = cv2.flip(recorded_video[0], 1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(image = Image.fromarray(im))
        uploaded_lmain.configure(image = image)
        uploaded_lmain.image = image
        

        analyzed_video = video_array
        fps = 40      
        frame_count = int(len(video_array))
        duration = duration
        current_analyzed_duration = duration
        length = int(len(video_array))
        loaded_file = [1,"webcam"]    
        t = datetime.datetime.now()
        date = t.strftime("%Y-%m-%d")
        or_width = video_array[0].shape[0]
        or_height = video_array[0].shape[1]
        wid_hei = str(round(or_width)) + " x " + str(round(or_height))
        text_infor = "Uploaded video info"+ "\n" + "\n" + "Date: {}".format(date)+ "\n" + "Name: {}".format(loaded_file[-1])+ "\n" + "Resolution: {}".format(wid_hei)+ "\n" + "Length: {}s".format(round(duration, 1))
        upload_information = tk.Label(upload_win, text = text_infor)
        upload_information.configure(justify="left", borderwidth=2, relief="sunken", width= 25, padx = 5, anchor='w')
        upload_information.place(x=850, y=30)


        start_label = tk.Label(upload_win, text=str(1)+"s", height = 1, width= 4, borderwidth=2, relief="sunken")
        start_label.place(x=50, y=725)
        end_label = tk.Label(upload_win, text=str(round(duration))+"s", height = 1, width= 4, borderwidth=2, relief="sunken")
        end_label.place(x=805, y=725)

        start_label.tkraise(aboveThis=slider)
        end_label.tkraise(aboveThis=slider)
        uploaded_frame.tkraise(aboveThis=slider)

        recording_upload_success_label = tk.Label(recording_win, text="Recorded video uploaded!", font=("Arial", 15))
        recording_upload_success_label.place(x=500, y=600)
        recording_win.destroy()

def recording_stop_func():
    global recording_cap
    global recorded_video
    global recording_button_upload
    global start_time
    global end_time
    global duration

    recording_cap.release()
    end_time = time.time()
    duration  = round(end_time- start_time,1)
    recording_button_upload = ck.CTkButton(recording_win, text='Upload', command = recording_push_func, height=40, width=150, text_font=("Arial", 15), text_color="black", fg_color="#ABB2B9", hover_color = "#566573", border_color = "black", border_width = 2)
    recording_button_upload.place(x=275, y=525)
    recording_button_start = ck.CTkButton(recording_win, text='Record', command = recording_start_func, height=40, width=150, text_font=("Arial", 15), text_color="black", fg_color="green", hover_color = "#566573", border_color = "black", border_width = 2)
    recording_button_start.place(x=100, y=525)

def recording_display_store_func():
    global recording_cap
    global recorded_video_lmain
    global recorded_video_display_frame
    global recording_win
    global recorded_video

    ret, recording_frame = recording_cap.read()
    img = recording_frame
    recorded_video.append(cv2.flip(recording_frame, 1))
    img = cv2.flip(img, 1)
    #img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr) 
    recorded_video_lmain.imgtk = imgtk 
    recorded_video_lmain.configure(image=imgtk)
    recorded_video_lmain.after(25, recording_display_store_func)

def recording_start_func():
    global recording_cap
    global recorded_video_lmain
    global recorded_video_display_frame
    global recording_win
    global recorded_video
    global recording_button_stop
    global start_time

    recorded_video = []
    recording_cap = cv2.VideoCapture(0)
    recording_cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
    recording_cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1500)
    recording_button_stop = ck.CTkButton(recording_win, text='Stop', command = recording_stop_func, height=40, width=150, text_font=("Arial", 15), text_color="black", fg_color="red", hover_color = "#566573", border_color = "black", border_width = 2)
    recording_button_stop.place(x=100, y=525)
    start_time = time.time()
    recording_display_store_func()
    
def confirm():
    global recording_cap
    global recording_win
    try:
        if recording_cap is not None:
            ans = askokcancel(title = "Webcam off", message="Your webcam is off")
            try:
                if ans:
                    recording_cap.release()
                    recording_win.destroy()
                else:
                    recording_win.destroy()
            except:
                pass
        else:
            recording_win.destroy()
            pass
    except:
        recording_win.destroy()
        pass

def recording_window():
    global recorded_video_display_frame
    global recorded_video_lmain
    global recording_cap
    global recording_win
    global recorded_video
    global recording_button_start

    recorded_video = []

    recording_win = tk.Toplevel(bg="#E5E8E8")
    recording_win.geometry("1000x725")
    recording_win.wm_title("Instant video recording")

    recorded_video_display_frame = tk.Frame(recording_win)
    recorded_video_display_frame.configure(bg="white", height=600, width=700)
    recorded_video_display_frame.place(x=100, y=50)
    recorded_video_lmain = tk.Label(recorded_video_display_frame) 
    recorded_video_lmain.place(x=0, y=0)
    
    recording_button_start = ck.CTkButton(recording_win, text='Record', command = recording_start_func, height=40, width=150, text_font=("Arial", 15), text_color="black", fg_color="green", hover_color = "#566573", border_color = "black", border_width = 2)
    recording_button_start.place(x=100, y=525)
    
    recording_win.protocol("WM_DELETE_WINDOW", confirm)
#####################################################################################################################################################################
#### app recall ####################################################################################################################################################
#####################################################################################################################################################################
def root_exit():
    root.quit()

#### dashboard ####################################################################################################################################################
def dashboard():
    window = ck.CTkToplevel()
    #window.geometry('1000x700')
    window.overrideredirect(False)
    window.overrideredirect(False)
    window.attributes('-fullscreen',True)
    window.title("User dashboard")
    
    
    def overall_dash_func():
        global upper_body_performance_im
        global core_body_performance_im
        global lower_body_performance_im
        global upper_body_performance_status_figure, lower_body_performance_status_figure, time_series_plot_im, fre_time_series_plot_im, vol_time_series_plot_im, goal_plot_im

        overall_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        upper_body_dashboard.configure(list_of_content_frame, text="Upper body", command=upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        core_body_dashboard.configure(list_of_content_frame, text="Core body", command=core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        lower_body_dashboard.configure(list_of_content_frame, text="Lower body", command=lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        #####################################################################################################################################################################
        #### body performance ###############################################################################################################################################
        #####################################################################################################################################################################
        overall_frame = tk.Frame(window, width=1600, height=1200, bg="#ebebec" )
        overall_frame.place(x= 300, y = 0)
        upper_performance_score = 80
        upper_body_performance_frame = tk.Frame(overall_frame, width=300, height=125)
        upper_body_performance_frame.place(x = 125, y = 25)
        upper_body_performance_im = image_load("upper_performance_frame_bg.png", resize_size = (300,165))
        upper_body_performance_label = tk.Label(upper_body_performance_frame, image=upper_body_performance_im, bg="#ebebec" )
        upper_body_performance_label.pack()
        upper_body_performance_score_label = tk.Label(overall_frame, text = "{}/100".format(upper_performance_score), font=("Arial", 15, 'bold'), bg="#8faadc")
        upper_body_performance_score_label.place(x = 250, y = 90)
        upper_body_performance_status_figure = image_load( im_name = "improve_im.jpg", resize_size = (42,90))
        upper_body_performance_status_label = tk.Label(overall_frame, image = upper_body_performance_status_figure, bg="#8faadc")
        upper_body_performance_status_label.place(x = 335, y = 80)

        core_performance_score = 70
        core_body_performance_frame = tk.Frame(overall_frame, width=300, height=125, bg = "#33B3EF")
        core_body_performance_frame.place(x = 450, y = 25)
        core_body_performance_im = image_load("core_performance_frame_bg.png", resize_size = (300,165))
        core_body_performance_label = tk.Label(core_body_performance_frame, image=core_body_performance_im, bg="#ebebec" )
        core_body_performance_label.pack()
        core_body_performance_score_label = tk.Label(overall_frame, text = "{}/100".format(core_performance_score), font=("Arial", 15, 'bold'), bg="#a9d18e")
        core_body_performance_score_label.place(x = 575, y = 90)

        lower_performance_score = 60
        lower_body_performance_frame = tk.Frame(overall_frame, width=300, height=125, bg = "#33B3EF")
        lower_body_performance_frame.place(x = 775, y = 25)
        lower_body_performance_im = image_load("lower_performance_frame_bg.png", resize_size = (300,165))
        lower_body_performance_label = tk.Label(lower_body_performance_frame, image=lower_body_performance_im, bg="#ebebec" )
        lower_body_performance_label.pack()
        lower_body_performance_score_label = tk.Label(overall_frame, text = "{}/100".format(lower_performance_score), font=("Arial", 15, 'bold'), bg="#ffd966")
        lower_body_performance_score_label.place(x = 900, y = 90)
        lower_body_performance_status_figure = image_load( im_name = "decrease_im.jpg", resize_size = (42,90))
        lower_body_performance_status_label = tk.Label(overall_frame, image = lower_body_performance_status_figure, bg="#ffd966")
        lower_body_performance_status_label.place(x = 985, y = 80)

        #####################################################################################################################################################################
        #### time series plot ###############################################################################################################################################
        #####################################################################################################################################################################
        def bar_figure_plot_func():
            try:
                global targeted_plot_period
                fig_size = (4.5,2.5)
                facecolor="white"
                color = '#8faadc'
                width = 0.5
                ylabel = "intensity"
                if targeted_plot_period == "w":
                    x = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    fre_y = []
                    vol_y = []
                    for i in range(0,7):
                        fre_y.append(random.randint(0,10))
                        vol_y.append(random.randint(0,10))
                    fre_y = np.array(fre_y)
                    vol_y = np.array(vol_y)
                elif targeted_plot_period == "m":
                    x = ["W1", "W2", "W3", "W4"]
                    fre_y = []
                    vol_y = []
                    for i in range(0,4):
                        fre_y.append(random.randint(0,10))
                        vol_y.append(random.randint(0,10))
                    fre_y = np.array(fre_y)
                    vol_y = np.array(vol_y)
                elif targeted_plot_period == "y":
                    x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Juy", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    fre_y = []
                    vol_y = []
                    for i in range(0,12):
                        fre_y.append(random.randint(0,10))
                        vol_y.append(random.randint(0,10))
                    fre_y = np.array(fre_y)
                    vol_y = np.array(vol_y)
                else:
                    x = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    fre_y = []
                    vol_y = []
                    for i in range(0,7):
                        fre_y.append(random.randint(0,10))
                        vol_y.append(random.randint(0,10))
                    fre_y = np.array(fre_y)
                    vol_y = np.array(vol_y)
                fre_figure_frame = tk.Label(overall_frame, bg="white")
                fre_figure_frame.place(x = 140, y = 775)
                fre_figure = bar_figure_gen_func(fig_size = (4.5,2.5), facecolor="white", x=x, y=fre_y, color = '#8faadc', width = 0.5, ylabel = "intensity")
                fre_plot = FigureCanvasTkAgg(fre_figure, master = fre_figure_frame)
                fre_plot.get_tk_widget().pack()
                vol_figure_frame = tk.Label(overall_frame, bg="white")
                vol_figure_frame.place(x = 640, y = 775)
                vol_figure = bar_figure_gen_func(fig_size = (4.5,2.5), facecolor="white", x=x, y=vol_y, color = '#8faadc', width = 0.5, ylabel = "intensity")
                vol_plot = FigureCanvasTkAgg(vol_figure, master = vol_figure_frame)
                vol_plot.get_tk_widget().pack()
            except:
                print("failed")
                pass
        def line_bar_figure_plot_func():
            try:
                global targeted_plot_period
                fig_size = (8.5,4.5)
                facecolor="white"
                color = '#8faadc'
                ylabel = "intensity"
                if targeted_plot_period == "w":
                    x = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    y = []
                    for i in range(0,7):
                        y.append(random.randint(0,10))
                    y = np.array(y)
                elif targeted_plot_period == "m":
                    x = ["W1", "W2", "W3", "W4"]
                    y = []
                    for i in range(0,4):
                        y.append(random.randint(0,10))
                    y = np.array(y)
                elif targeted_plot_period == "y":
                    x = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Juy", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    y = []
                    for i in range(0,12):
                        y.append(random.randint(0,10))
                    y = np.array(y)
                else:
                    x = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    y = []
                    for i in range(0,7):
                        y.append(random.randint(0,10))
                    y = np.array(y)
                figure = plt.Figure(figsize=fig_size, dpi=100)
                figure.patch.set_facecolor(facecolor)
                figure.tight_layout()
                ax = figure.add_subplot(111)
                #ax.tick_params(left = True, right = False , labelleft = True ,
                #                labelbottom = False, bottom = False)
                ax.bar(x=x,height=y, color = color, width = 0.5)
                ax.plot(x, y*1.5)
                ax.set_ylabel(ylabel, fontsize=18)
                ax.grid(linestyle='--', linewidth=0.5)
                time_series_figure_frame = tk.Label(overall_frame, bg="white")
                time_series_figure_frame.place(x = 175, y = 250)
                time_series_figure = figure
                time_series_plot = FigureCanvasTkAgg(time_series_figure, master = time_series_figure_frame)
                time_series_plot.get_tk_widget().pack()
            except:
                print("failed")
                pass

        def week_button_disable_func():
            global targeted_plot_period
            targeted_plot_period = "w"
            line_bar_figure_plot_func()
            bar_figure_plot_func()
            week_button.configure(fg_color = "#8faadc")
            month_button.configure(state="normal", bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc")
            year_button.configure(state="normal", bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc",)
            week_button.configure(state = tk.DISABLED)  
        def month_button_disable_func():
            global targeted_plot_period
            targeted_plot_period = "m"
            line_bar_figure_plot_func()
            bar_figure_plot_func()
            month_button.configure(fg_color = "#8faadc")
            week_button.configure(state="normal", bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc")
            year_button.configure(state="normal", bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc")
            month_button.configure(state = tk.DISABLED)
        def year_button_disable_func():
            global targeted_plot_period
            targeted_plot_period = "y"
            line_bar_figure_plot_func()
            bar_figure_plot_func()
            year_button.configure(fg_color = "#8faadc")
            week_button.configure(state="normal", bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc")
            month_button.configure(state="normal", bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc")
            year_button.configure(state = tk.DISABLED)

        fre_time_series_plot_frame = tk.Frame(overall_frame, width=475, height=300, bg="#33B3EF")
        fre_time_series_plot_frame.place(x = 125, y = 750)
        fre_time_series_plot_im = image_load("time_series_plot_bg.png", resize_size = (475,300))
        fre_time_series_plot_label = tk.Label(fre_time_series_plot_frame, image=fre_time_series_plot_im, bg="#ebebec")
        fre_time_series_plot_label.pack()
        fre_figure_frame = tk.Label(overall_frame, bg="white")
        fre_figure_frame.place(x = 140, y = 775)
        fre_figure = bar_figure_gen_func(fig_size = (4.5,2.5), facecolor="white", x=[1,2,3], y=[1,2,3], color = '#8faadc', width = 0.5, ylabel = "intensity")
        fre_plot = FigureCanvasTkAgg(fre_figure, master = fre_figure_frame)
        fre_plot.get_tk_widget().pack()


        vol_time_series_plot_frame = tk.Frame(overall_frame, width=475, height=300, bg="#33B3EF")
        vol_time_series_plot_frame.place(x = 625, y = 750)
        vol_time_series_plot_im = image_load("time_series_plot_bg.png", resize_size = (475,300))
        vol_time_series_plot_label = tk.Label(vol_time_series_plot_frame, text = "volume plot", fg = "red",image=vol_time_series_plot_im, bg="#ebebec", compound='center')
        vol_time_series_plot_label.pack()
        vol_figure_frame = tk.Label(overall_frame, bg="white")
        vol_figure_frame.place(x = 640, y = 775)
        vol_figure = bar_figure_gen_func(fig_size = (4.5,2.5), facecolor="white", x=[1,2,3], y=[1,2,3], color = '#8faadc', width = 0.5, ylabel = "intensity")
        vol_plot = FigureCanvasTkAgg(vol_figure, master = vol_figure_frame)
        vol_plot.get_tk_widget().pack()

        time_series_plot_frame = tk.Frame(overall_frame,bg="#33B3EF")
        time_series_plot_frame.place(x = 125, y = 225)
        time_series_plot_im = image_load("time_series_plot_bg.png", resize_size = (950,500))
        time_series_plot_label = tk.Label(time_series_plot_frame, image=time_series_plot_im, bg="#ebebec")
        time_series_plot_label.pack()
        time_series_figure_frame = tk.Label(overall_frame, bg="white")
        time_series_figure_frame.place(x = 175, y = 250)
        time_series_figure = line_bar_figure_gen_func(fig_size = (8.5,4.5), facecolor="white", x=["Mon", "Tue", "Wed", "Thu", "Fri"], y=np.array([1,5,26,18,5]), color = '#8faadc', ylabel = "intensity")
        time_series_plot = FigureCanvasTkAgg(time_series_figure, master = time_series_figure_frame)
        time_series_plot.get_tk_widget().pack()

        week_button = ck.CTkButton(overall_frame, text="week", command = lambda:[week_button_disable_func()], width=100, height = 25, bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc", border_color = "white", border_width = 2)
        week_button.place(x = 331, y = 161)
        month_button = ck.CTkButton(overall_frame, text="month", command = lambda:[month_button_disable_func()], width=100, height = 25, bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc", border_color = "white", border_width = 2)
        month_button.place(x = 430, y = 161)
        year_button = ck.CTkButton(overall_frame, text="year", command = lambda:[year_button_disable_func()], width=100, height = 25, bg_color = "#ebebec", fg_color = "white", hover_color = "#8faadc", border_color = "white", border_width = 2)
        year_button.place(x = 529, y = 161)

        #####################################################################################################################################################################
        #### time series plot ###############################################################################################################################################
        #####################################################################################################################################################################

        goal_plot_frame = tk.Frame(overall_frame, width=400, height=1400, bg="#33B3EF")
        goal_plot_frame.place(x = 1150, y = 25)
        goal_plot_im = image_load("vertical_white_board.png", resize_size = (400,1025))
        goal_plot_label = tk.Label(goal_plot_frame, image=goal_plot_im, bg="#ebebec")
        goal_plot_label.pack()
        goal_plot_figure_frame = tk.Label(overall_frame, width=450, height=325, bg="white")
        goal_plot_figure_frame.place(x = 1200, y = 50)
        goal_plot_figure = radar_chart_func(fig_size = (3,3), facecolor="white")
        goal_plot_plot = FigureCanvasTkAgg(goal_plot_figure, master = goal_plot_figure_frame)
        goal_plot_plot.get_tk_widget().pack()

        #####################################################################################################################################################################
        #### current health status ##########################################################################################################################################
        #####################################################################################################################################################################
        Gender = "male"
        Wei = 80
        BMI = 25
        Fat = 15
        Ben_pres = 100
        Squ = 100
        Pla = 120
        info = [Gender, Wei, BMI, Fat, Ben_pres, Squ, Pla]
        cur_health_status_label = tk.Label(overall_frame
                                        , text="Gender: {}\n \nWeight(kg): {}\n \nBMI (kg/m^2): {}\n \nFat %: {}\n \nBenchpress (kg): {}\n \nSquat (kg): {}\n \nPlank (s): {}".format(*info)
                                        , font=("Arial", 14)
                                        , anchor="e"
                                        , justify="left"
                                        , bg="white")
        cur_health_status_label.place(x = 1200, y = 400)

        weight_progress_label = tk.Label(overall_frame, bg="white")
        weight_progress_label.place(x = 1450, y = 440)
        weight_progress_figure = circle_progress_func(complete_perc = 100)
        weight_progress_plot = FigureCanvasTkAgg(weight_progress_figure, master = weight_progress_label)
        weight_progress_plot.get_tk_widget().pack()

        BMI_progress_label = tk.Label(overall_frame, bg="white")
        BMI_progress_label.place(x = 1450, y = 495)
        BMI_progress_figure = circle_progress_func(complete_perc = 80)
        BMI_progress_plot = FigureCanvasTkAgg(BMI_progress_figure, master = BMI_progress_label)
        BMI_progress_plot.get_tk_widget().pack()

        Fat_progress_label = tk.Label(overall_frame, bg="white")
        Fat_progress_label.place(x = 1450, y = 545)
        Fat_progress_figure = circle_progress_func(complete_perc = 60)
        Fat_progress_plot = FigureCanvasTkAgg(Fat_progress_figure, master = Fat_progress_label)
        Fat_progress_plot.get_tk_widget().pack()

        Ben_pres_progress_label = tk.Label(overall_frame, bg="white")
        Ben_pres_progress_label.place(x = 1450, y = 600)
        Ben_pres_progress_figure = circle_progress_func(complete_perc = 70)
        Ben_pres_progress_plot = FigureCanvasTkAgg(Ben_pres_progress_figure, master = Ben_pres_progress_label)
        Ben_pres_progress_plot.get_tk_widget().pack()

        Squ_progress_label = tk.Label(overall_frame, bg="white")
        Squ_progress_label.place(x = 1450, y = 650)
        Squ_progress_figure = circle_progress_func(complete_perc = 50)
        Squ_progress_plot = FigureCanvasTkAgg(Squ_progress_figure, master = Squ_progress_label)
        Squ_progress_plot.get_tk_widget().pack()

        Pla_progress_label = tk.Label(overall_frame, bg="white")
        Pla_progress_label.place(x = 1450, y = 705)
        Pla_progress_figure = circle_progress_func(complete_perc = 10)
        Pla_progress_plot = FigureCanvasTkAgg(Pla_progress_figure, master = Pla_progress_label)
        Pla_progress_plot.get_tk_widget().pack()


        average_exercise_session_bg_label= tk.Label(overall_frame
                                                    , text = "Average \n\n\n\n\n\n Sessions/week"
                                                    , font=("Arial", 14)
                                                    ,bg="white")
        average_exercise_session_bg_label.place(x = 1275, y = 800)

        aver_session_val = 2
        average_exercise_session_label= tk.Label(overall_frame
                                                    ,text = aver_session_val
                                                    , font=("Arial", 40, "bold")
                                                    ,bg="white")
        average_exercise_session_label.place(x = 1330, y = 850)

        
    def upper_dash_func():
        upper_dash_frame = tk.Frame(window,width=1600, height=1200, bg="blue" )
        upper_dash_frame.place(x= 300, y = 0)
        upper_body_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        overall_dashboard.configure(list_of_content_frame, text="Overall status", command=lambda:[threading.Thread(target=overall_dash_func).start()], width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        core_body_dashboard.configure(list_of_content_frame, text="Core body", command=core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        lower_body_dashboard.configure(list_of_content_frame, text="Lower body", command=lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")

    def core_dash_func():
        core_dash_frame = tk.Frame(window,width=1600, height=1200, bg="red" )
        core_dash_frame.place(x= 300, y = 0)
        core_body_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        overall_dashboard.configure(list_of_content_frame, text="Overall status", command=lambda:[threading.Thread(target=overall_dash_func).start()], width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        upper_body_dashboard.configure(list_of_content_frame, text="Upper body", command=upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        lower_body_dashboard.configure(list_of_content_frame, text="Lower body", command=lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")

    def lower_dash_func():
        lower_dash_frame = tk.Frame(window,width=1600, height=1200, bg="green" )
        lower_dash_frame.place(x= 300, y = 0)
        lower_body_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        overall_dashboard.configure(list_of_content_frame, text="Overall status", command=lambda:[threading.Thread(target=overall_dash_func).start()], width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        upper_body_dashboard.configure(list_of_content_frame, text="Upper body", command=upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        core_body_dashboard.configure(list_of_content_frame, text="Core body", command=core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
    def exit():
        window.withdraw()

    #####################################################################################################################################################################
    #### List of content frame ####################################################################################################################################################
    #####################################################################################################################################################################

    list_of_content_frame = tk.Frame(window, width=300, height=1200, bg = "#3498DB")
    list_of_content_frame.place(x=0, y=0)

    overall_dashboard = ck.CTkButton(list_of_content_frame, text="Overall status", command=lambda:[threading.Thread(target=overall_dash_func).start()], width=220, height = 45, fg_color="#ebebec", text_color="black", text_font=("Arial", 12), state="DISABLED")
    overall_dashboard.place(x=10, y=175)

    upper_body_dashboard = ck.CTkButton(list_of_content_frame, text="Upper body", command=upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12))
    upper_body_dashboard.place(x=10, y=225)

    core_body_dashboard = ck.CTkButton(list_of_content_frame, text="Core body", command=core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12))
    core_body_dashboard.place(x=10, y=275)

    lower_body_dashboard = ck.CTkButton(list_of_content_frame, text="Lower body", command=lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12))
    lower_body_dashboard.place(x=10, y=325)

    exit_button = ck.CTkButton(list_of_content_frame, text="Exit", command=exit, width=150, height = 50)
    exit_button.place(x=50, y=700)

    overall_dash_func()
    




#### movement assessment ##########################################################################################################################################
def movement_assessment_func():
    movement_window = ck.CTkToplevel()
    #window.geometry('1000x700')
    movement_window.overrideredirect(False)
    movement_window.overrideredirect(False)
    movement_window.attributes('-fullscreen',True)
    movement_window.title("User dashboard")
    def movement_overall_dash_func():
        global movement_upper_body_performance_im
        def OLST_movement_assessment():
            recording_window()

        movement_overall_dash_frame = tk.Frame(movement_window,width=1600, height=1200 )
        movement_overall_dash_frame.place(x= 300, y = 0)
        
        movement_overall_frame = tk.Frame(movement_window, width=1600, height=1200, bg="#ebebec" )
        movement_overall_frame.place(x= 300, y = 0)
        movement_upper_body_performance_frame = tk.Frame(movement_overall_frame, width=300, height=252)
        movement_upper_body_performance_frame.place(x = 125, y = 25)
        movement_upper_body_performance_im = image_load("movement_assessment_blue.png", resize_size = (300,252))
        movement_upper_body_performance_label = tk.Label(movement_upper_body_performance_frame, image=movement_upper_body_performance_im, bg="#ebebec" )
        movement_upper_body_performance_label.pack()
        movement_upper_body_performance_button = ck.CTkButton(movement_overall_frame, text="Start", command = OLST_movement_assessment, bg_color="#8faadc")
        movement_upper_body_performance_button.place(x = 155, y = 175)

        movement_overall_dashboard.configure(movement_list_of_content_frame, fg_color = "#ebebec", state = tk.DISABLED)
        movement_upper_body_dashboard.configure(movement_list_of_content_frame, text="Upper body", command=movement_upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_core_body_dashboard.configure(movement_list_of_content_frame, text="Core body", command=movement_core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_lower_body_dashboard.configure(movement_list_of_content_frame, text="Lower body", command=movement_lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")

    def movement_upper_dash_func():
        movement_upper_dash_frame = tk.Frame(movement_window,width=1600, height=1200, bg="blue" )
        movement_upper_dash_frame.place(x= 300, y = 0)
        movement_upper_body_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        movement_overall_dashboard.configure(movement_list_of_content_frame, text="All assessments", command=movement_overall_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_core_body_dashboard.configure(movement_list_of_content_frame, text="Core body", command=movement_core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_lower_body_dashboard.configure(movement_list_of_content_frame, text="Lower body", command=movement_lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")

    def movement_core_dash_func():
        movement_core_dash_frame = tk.Frame(movement_window,width=1600, height=1200, bg="red" )
        movement_core_dash_frame.place(x= 300, y = 0)
        movement_core_body_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        movement_overall_dashboard.configure(movement_list_of_content_frame, text="All assessments", command=movement_overall_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_upper_body_dashboard.configure(movement_list_of_content_frame, text="Upper body", command=movement_upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_lower_body_dashboard.configure(movement_list_of_content_frame, text="Lower body", command=movement_lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")

    def movement_lower_dash_func():
        movement_lower_dash_frame = tk.Frame(movement_window,width=1600, height=1200, bg="green" )
        movement_lower_dash_frame.place(x= 300, y = 0)
        movement_lower_body_dashboard.configure(fg_color = "#ebebec", state = tk.DISABLED)
        movement_overall_dashboard.configure(movement_list_of_content_frame, text="All assessments", command=movement_overall_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_upper_body_dashboard.configure(movement_list_of_content_frame, text="Upper body", command=movement_upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")
        movement_core_body_dashboard.configure(movement_list_of_content_frame, text="Core body", command=movement_core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12), state="normal")

    def movement_exit():
        movement_window.withdraw()
    #####################################################################################################################################################################
    #### List of content frame ####################################################################################################################################################
    #####################################################################################################################################################################

    movement_list_of_content_frame = tk.Frame(movement_window, width=300, height=1200, bg = "#3498DB")
    movement_list_of_content_frame.place(x=0, y=0)

    movement_overall_dashboard = ck.CTkButton(movement_list_of_content_frame, text="All assessments", command=movement_overall_dash_func, width=220, height = 45, fg_color="#ebebec", text_color="black", text_font=("Arial", 12), state="DISABLED")
    movement_overall_dashboard.place(x=10, y=175)

    movement_upper_body_dashboard = ck.CTkButton(movement_list_of_content_frame, text="Upper body", command=movement_upper_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12))
    movement_upper_body_dashboard.place(x=10, y=225)

    movement_core_body_dashboard = ck.CTkButton(movement_list_of_content_frame, text="Core body", command=movement_core_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12))
    movement_core_body_dashboard.place(x=10, y=275)

    movement_lower_body_dashboard = ck.CTkButton(movement_list_of_content_frame, text="Lower body", command=movement_lower_dash_func, width=220, height = 45, fg_color="#3498DB", text_color="white", text_font=("Arial", 12))
    movement_lower_body_dashboard.place(x=10, y=325)

    movement_exit_button = ck.CTkButton(movement_list_of_content_frame, text="Exit", command=movement_exit, width=150, height = 50)
    movement_exit_button.place(x=50, y=700)

    movement_overall_dash_func()

#####################################################################################################################################################################
#### main window button func ########################################################################################################################################
#####################################################################################################################################################################

button_size = [200, 150]
dashboard_button = ck.CTkButton(root, text ="Dashboard", command = lambda:[threading.Thread(target=dashboard).start()], width=button_size[0], height=button_size[1], text_font=("Arial", 20))
dashboard_button.place(x = 200, y = 300)

movement_evaluation_button = ck.CTkButton(root, text ="Movement Evaluation", command = movement_assessment_func, width=button_size[0], height=button_size[1], text_font=("Arial", 20))
movement_evaluation_button.place(x = 600, y = 300)

Exit_button = ck.CTkButton(root, text ="Exit", command = root_exit, width=button_size[0], height=button_size[1], text_font=("Arial", 20))
Exit_button.place(x = 1000, y = 300)


root.mainloop()