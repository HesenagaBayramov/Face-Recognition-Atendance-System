# Import the tkinter library and alias it as tk for easier reference

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import customtkinter as ctk   # for UI improvment
import csv
import os  
import cv2
import numpy as np
import datetime


# ============================ Main Window Setup ============================

ctk.set_appearance_mode("dark")          # "light", "dark", "system"
ctk.set_default_color_theme("blue")      # "blue", "green", "dark-blue"

root = ctk.CTk()  
root.title("Smart Attendance System")
root.geometry("600x500")


# ============================ Constants and Directories ============================

STUDENTS_FILE = "students.csv"
DATASET_DIR = "dataset"
ATTENDANCE_DIR = "attendance"
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINER_FILE = "trainer.yml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ============================ ROLES ============================

CURRENT_ROLE = None     #This will store the role you are currently logged in with.
CURRENT_USER_NAME = None  #admin/teacher username
CURRENT_STUDENT_ID = None  #student ID if logged in as student

ROLES = ["admin", "teacher", "student"]  # List of possible roles

# Admins
ADMINS = {
    "admin": "admin123",   # username: password
}

# Teachers
TEACHERS = {
    "teacher1": "teach123",
    "teacher2": "teach234",
}

# Groups
GROUPS = [
    "ZU-054",
    "ZU-044",
    "ZU-034", 
    "ZU-024",
    "ZU-014",
    "BS-08.24",
]



# ============================ Functions ============================

# Configure buttons based on role
def configre_buttons_by_role():
    """
    Enables/disables the primary buttons based on the value of CURRENT_ROLE.
    This function will be called after a successful login.
    """
    if CURRENT_ROLE == "admin":
        btn_register.configure(state="normal")
        btn_train.configure(state="normal")
        btn_attendance.configure(state="normal")
        btn_view.configure(state="normal")
        btn_view_by_date.configure(state="normal")
        btn_stats_range.configure(state="normal")

    elif CURRENT_ROLE == "teacher":
        btn_register.configure(state="disabled")
        btn_train.configure(state="disabled")
        btn_attendance.configure(state="normal")
        btn_view.configure(state="normal")
        btn_view_by_date.configure(state="normal")
        btn_stats_range.configure(state="normal")

    elif CURRENT_ROLE == "student":
        btn_register.configure(state="disabled")
        btn_train.configure(state="disabled")
        btn_attendance.configure(state="disabled")
        btn_view.configure(state="normal")
        btn_view_by_date.configure(state="normal")
        btn_stats_range.configure(state="normal")

    else:
        btn_register.configure(state="disabled")
        btn_train.configure(state="disabled")
        btn_attendance.configure(state="disabled")
        btn_view.configure(state="disabled")
        btn_view_by_date.configure(state="disabled")
        btn_stats_range.configure(state="disabled")

# Open login window
def open_login_window():
    """
    When the program is opened, it opens a separate window for Admin / Teacher / Student selection.
    It calls the login function corresponding to each role.
    """
    login_win = ctk.CTkToplevel(root)
    login_win.title("Login - Role Selection")
    login_win.geometry("300x250")
    login_win.grab_set()  # Make this window modal
    # This makes the login window always on top of the main window
    login_win.transient(root)    # Stays on top of root
    login_win.lift()             # Go on top of the root window
    login_win.focus_force()      # Focus claviature

    # Help window to make position middle
    login_win.resizable(False, False)

    title = ctk.CTkLabel(
        login_win,
        text="Select Role to Login",
        font=("Helvetica", 18, "bold")
    )
    title.pack(pady=(20, 10))

    # Düymələr üçün frame (daha səliqəli düzülüş üçün)
    btn_frame = ctk.CTkFrame(login_win)
    btn_frame.pack(pady=10, padx=20, fill="x")

    btn_admin = ctk.CTkButton(
        btn_frame,
        text="Admin",
        width=200,
        height=36,
        corner_radius=18,
        command=lambda: admin_login(login_win)
    )
    btn_admin.pack(pady=5)

    btn_teacher = ctk.CTkButton(
        btn_frame,
        text="Teacher",
        width=200,
        height=36,
        corner_radius=18,
        command=lambda: teacher_login(login_win)
    )
    btn_teacher.pack(pady=5)

    btn_student = ctk.CTkButton(
        btn_frame,
        text="Student",
        width=200,
        height=36,
        corner_radius=18,
        command=lambda: student_login(login_win)
    )
    btn_student.pack(pady=5)

# Admin login function
def admin_login(win):
    """Handles admin login."""
    global CURRENT_ROLE, CURRENT_USER_NAME

    username = simpledialog.askstring("Admin Login", "Enter Admin Username:", parent=win)
    if username is None:
        return  # User cancelled

    password = simpledialog.askstring("Admin Login", "Enter Admin Password:", show='*', parent=win)
    if password is None:
        return  # User cancelled
    
    # Simple check

    if ADMINS.get(username) == password:
        CURRENT_ROLE = "admin"
        CURRENT_USER_NAME = username
        messagebox.showinfo("Login Successful", f"Welcome, Admin {username}!", parent=win)
        configre_buttons_by_role()
        win.destroy() # Close login window
    else:
        messagebox.showerror("Login Failed", "Invalid admin credentials.", parent=win)

# Teacher login function
def teacher_login(win):
    """Handles teacher login."""
    global CURRENT_ROLE, CURRENT_USER_NAME

    username = simpledialog.askstring("Teacher Login", "Enter Teacher Username:", parent=win)
    if username is None:
        return  # User cancelled

    password = simpledialog.askstring("Teacher Login", "Enter Teacher Password:", show='*', parent=win)
    if password is None:
        return  # User cancelled
    
    # Simple check
    if TEACHERS.get(username) == password:
        CURRENT_ROLE = "teacher"
        CURRENT_USER_NAME = username
        messagebox.showinfo("Login Successful", f"Welcome, Teacher {username}!", parent=win)
        configre_buttons_by_role()
        win.destroy() # Close login window
    else:
        messagebox.showerror("Login Failed", "Invalid teacher credentials.", parent=win)

# Student login function
def student_login(win):
    """Handles student login."""
    global CURRENT_ROLE, CURRENT_STUDENT_ID

    student_id = simpledialog.askstring("Student Login", "Enter Student ID:", parent=win)
    if student_id is None:
        return  # User cancelled

    # Check if student ID is numeric
    if not student_id.isdigit():
        messagebox.showerror("Error", "Student ID must be an numric.", parent=win)
        return
    
    sid = int(student_id)
    students = load_students() # Load students from file

    if sid in students:
        info = students[sid]
        name = info.get("name", "")
        group = info.get("group", "")

        CURRENT_ROLE = "student"
        CURRENT_STUDENT_ID = sid

        messagebox.showinfo("Login Successful", 
                            f"Welcome, {name}!\nID: {sid}\nGroup: {group}",
                            parent=win
        )

        configre_buttons_by_role()
        win.destroy() # Close login window

    else:
        messagebox.showerror("Error", "No student found this ID", parent=win)

# Ensure students file exists
def ensure_students_file():
    """If students.csv does not exist, it creates it with a title."""
    if not os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["student_id", "name", "group"])


# Add student to file
def add_student_to_file(student_id: str, name: str, group: str) -> bool:
    """Adds a new student to the file. Returns False if the ID already exists."""
    ensure_students_file()

    # THIS LINE MUST BE PRESENT
    existing_ids = set()

    # Let's read the existing IDs
    with open(STUDENTS_FILE, mode='r', newline='', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            existing_ids.add(row["student_id"])
    if student_id in existing_ids:
        return False  # ID already exists
    
    # Append the new student
    with open(STUDENTS_FILE, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([student_id, name, group])

    return True

# Register student and face capture
def capture_face_for_student(student_id: str):
    """ 
    Captures facial images from the camera for the given student ID
    and writes them to the dataset folder in the format User.<id>.<say>.jpg.
    """
    
    # Does the Haarcascade file really exist?
    if not os.path.exists(HAAR_CASCADE_FILE):
        messagebox.showerror(
            "Error",
            f"Haarcascade file '{HAAR_CASCADE_FILE}' not found."
        )
        return
    
    # Load Haarcascade for face detection
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)

    # Open the camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return
    
    messagebox.showinfo(
        "Information",
        "Camera is on.\n"
        "Face the camera and move slightly left and right.\n"
        "You can press 'q' to exit.\n"
        "About 50 pictures will be taken."
    )

    count = 0
    target_count = 50   # Number of images to capture

    try:
        # While loop to capture images
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # We make the color grayscale (gray)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2, 
                minNeighbors=5
            )

            for (x, y, w, h) in faces:
                count += 1
                
                # Cut Face part
                face_img = gray[y:y + h, x:x + w]

                # File name format: User.<id>.<num>.jpg
                file_name = os.path.join(DATASET_DIR, f"User.{student_id}.{count}.jpg")
                cv2.imwrite(file_name, face_img)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Face Capture - q = exit', frame)

            # 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # If we have enough samples, stop
            if count >= target_count:
                break
    # We use finally to ensure resources are released
    finally:
        cam.release()
        cv2.destroyAllWindows()

    messagebox.showinfo(
        "Success",
        f"{count} face images captured for Student ID {student_id}."
    )

# Load students from file
def load_students():
    """
    Returns {id: {"name": name, "group": group}} dict from students.csv.
    """
    ensure_students_file()
    students = {}

    with open(STUDENTS_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("student_id", "").strip()
            if sid.isdigit():
                students[int(sid)] = {
                    "name": row.get("name", "") or "",
                    "group": row.get("group", "") or "",
                }

    return students



# Ask group function
def ask_group():
    """Opens a small window with a Combobox to select a group. 
    Returns selected group string or None if canceled.
    """
    win = ctk.CTkToplevel(root)
    win.title("Select Group")
    win.geometry("300x200")
    win.transient(root)
    win.grab_set()
    win.lift()
    win.focus_force()

    title = ctk.CTkLabel(
        win,
        text="Select student's group:",
        font=("Helvetica", 13)
    )
    title.pack(pady=(15, 10))

    group_var = tk.StringVar()

    # Burda Groups sənin mövcud siyahındır (ZU-054, ZU-044 və s.)
    combo = ctk.CTkComboBox(
        win,
        variable=group_var,
        values=GROUPS,        
        state="readonly",
        width=200,
        height=32,
        corner_radius=12
    )

    if GROUPS:
        combo.set(GROUPS[0])  # default choosen
    combo.pack(pady=5)

    result = {"value": None}

    def on_ok():
        val = group_var.get()
        if not val:
            messagebox.showwarning("Warning", "Please select a group.", parent=win)
            return
        result["value"] = val
        win.destroy()

    def on_cancel():
        result["value"] = None
        win.destroy()

    btn_frame = ctk.CTkFrame(win)
    btn_frame.pack(pady=15)
    
    btn_ok = ctk.CTkButton(
        btn_frame,
        text="OK",
        width=80,
        height=32,
        corner_radius=16,
        command=on_ok
    )
    btn_ok.pack(side="left", padx=5)

    btn_cancel = ctk.CTkButton(
        btn_frame,
        text="Cancel",
        width=80,
        height=32,
        corner_radius=16,
        fg_color="#555555",
        hover_color="#444444",
        command=on_cancel
    )
    btn_cancel.pack(side="left", padx=5)

    win.wait_window()  # Wait until this window is closed
    return result["value"]


def ask_group_filter():
    """Opens a small window with a Combobox to select a group filter. 
    Returns selected group string or None if canceled.
    """
    win = tk.Toplevel(root)
    win.title("Select Group")
    win.geometry("300x200")
    win.grab_set()

    label = tk.Label(win, text="Select group to filter:", font=("Helvetica", 11))
    label.pack(pady=10)

    selected_group = tk.StringVar()

    combo = ttk.Combobox(win, textvariable=selected_group, values=["All"] + GROUPS, state="readonly")
    combo.current(0)
    combo.pack(pady=5)

    result = {"group": None}

    def apply_filter():
        group = selected_group.get()
        result["group"] = None if group == "All" else group
        win.destroy()
    
    btn = tk.Button(win, text="Apply Filter", command=apply_filter)
    btn.pack(pady=10)

    win.wait_window()  # Wait until this window is closed
    return result["group"]

# Register student function
def register_student():
    # Get ID
    student_id = simpledialog.askstring("Student ID", "Enter Student ID:", parent=root)
    if not student_id:
        return  # User cancelled
    
    # Let's check that the ID is an integer (professional approach)

    if not student_id.isdigit():
        messagebox.showerror("Invalid ID", "Student ID must be an integer.", parent=root)
        return
    
    # Get Name
    name = simpledialog.askstring("Student Name", "Enter Student Name:", parent=root)
    if not name:
        return  # User cancelled
    
    # Select Group(Combobox)
    group = ask_group()
    if group is None:
        # user cancelled the group selection
        messagebox.showwarning("Cancelled", "Group selection cancelled(no group selected).", parent=root)
        return

    # Add to file
    ok = add_student_to_file(student_id, name, group)

    if not ok:
        messagebox.showerror("Error! Duplicated ID", f"Student ID {student_id} already exists.", parent=root)
        return
    else:
        messagebox.showinfo("Success",
                            f"Student {name} (ID: {student_id}, Group: {group}) registered successfully.",
                            parent=root
        )

    messagebox.showinfo(
        "Success",
        f"Student added:\nID: {student_id}\nName: {name}\nGroup: {group}\n\n"
        "Face images will now be taken.",
        parent=root
    )

    # Turn on the camera and capture face images
    capture_face_for_student(student_id)

# Train model function
def train_model():
    """
    Trains an LBPH model from the face images in the dataset/ directory
    and writes them to the trainer.yml file.
    """

    # 1. Find dataset images
    image_paths = [
        os.path.join(DATASET_DIR, f) 
        for f in os.listdir(DATASET_DIR) 
        if f.lower().endswith('.jpg')
    ]

    if not image_paths:
        messagebox.showwarning(
            "Attention",
            "Dataset folder is empty.\nYou must first collect face images with Register Student."
        )
        return

    face_samples = []
    ids = []

    # 2. Read each image and extract the ID from the filename
    for image_path in image_paths:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue    # just in case 

        # Filename format: User.<id>.<num>.jpg
        filename = os.path.basename(image_path)
        parts = filename.split('.')
        # parts: ['User', '<id>', '<num>', 'jpg']
        if len(parts) >= 3 and parts[1].isdigit():
            student_id = int(parts[1])
            ids.append(student_id)
            face_samples.append(img)
    if not face_samples:
        messagebox.showerror(
            "Xəta",
            "Dataset-dən tələbə ID-lərini oxumaqda problem yarandı."
        )
        return
    
    # 3. We create and train the LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_FILE)

    messagebox.showinfo(
        "Success",
        f"The model was successfully trained and written to the file '{TRAINER_FILE}'."
    )

# Take attendance function
def take_attendance():
    """
    It recognizes the face using the trained model (trainer.yml),
    writes student_id, name, time to the attendance_<date>.csv file.
    """

    # 1. Is the trainer file available?
    if not os.path.isfile(TRAINER_FILE):
        messagebox.showerror("Error", "You must 'Train Model' first. trainer.yml not found.")
        return

    # 2. Is the Haarcascade file available?
    if not os.path.isfile(HAAR_CASCADE_FILE):
        messagebox.showerror("Error", f"{HAAR_CASCADE_FILE} file not found.")
        return

    # 3. Load the model and cascade
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)

    # 4. Load student list (id -> name)
    students = load_students()
    if not students:
        messagebox.showwarning("Attention", "Student list is empty. Add a student first.")
        return

    # 5. Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera could not be opened.")
        return

    # 6. Today's attendance file
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today_str}.csv")

    file_exists = os.path.isfile(attendance_file)
    attended_ids = set()

    # Read previous attendances (to avoid registering the same person twice)
    if file_exists:
        with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("student_id", "").isdigit():
                    attended_ids.add(int(row["student_id"]))

    # Open file in append mode
    f_att = open(attendance_file, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(f_att)
    if not file_exists:
        writer.writerow(["student_id", "name", "group", "time"])

    messagebox.showinfo(
        "Information",
        "Camera turned on.\n"
        "Students take turns looking at the camera.\n"
        "Press 'q' to exit."
    )

    # 7. Camera loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]

                # Take prediction from the model
                student_id_pred, confidence = recognizer.predict(face_img)

            # The smaller the confidence, the better the fit
                if confidence < 70:
                    info = students.get(student_id_pred)

                    if info is not None:
                        name = info.get("name", "Unknown")
                        group = info.get("group", "")
                    else:
                        name = "Unknown"
                        group = ""

                    cv2.putText(
                        frame,
                        f"{name} ({int(confidence)})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # If this student has already been enrolled today, do not enroll again
                    if student_id_pred not in attended_ids and name != "Unknown":
                        attended_ids.add(student_id_pred)
                        now_time = datetime.datetime.now().strftime("%H:%M:%S")
                        writer.writerow([student_id_pred, name, group, now_time])
                        print(f"Attendance: {student_id_pred} - {name} ({group})- {now_time}")
                else:
                    # Unknown face
                    cv2.putText(
                        frame,
                        "Unknown",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow("Attendance - q = exit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        f_att.close()
        cv2.destroyAllWindows()

    messagebox.showinfo("Success", "Attendance session ended.")

# Ask date or filter dunction
def ask_date_range():
    """
    Asks user for a start and end date (YYYY-MM-DD).
    Returns (start_date, end_date) as datetime.date objects,
    or (None, None) if cancelled or invalid.
    """
    start_str = simpledialog.askstring(
        "Date Range",
        "Enter START date (YYYY-MM-DD):",
        parent=root
    )

    if not start_str:
        return None, None

    end_str = simpledialog.askstring(
        "Date Range",
        "Enter END date (YYYY-MM-DD):",
        parent=root
    )

    if not end_str:
        return None, None

    try:
        start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD.", parent=root)
        return None, None

    if end_date < start_date:
        messagebox.showerror("Error", "End date cannot be before start date.", parent=root)
        return None, None

    return start_date, end_date

def view_attendance_stats_range():
    """
    Shows how many times each student attended between a date range.
    If logged in as a student, shows only that student's count.
    """
    start_date, end_date = ask_date_range()
    if start_date is None or end_date is None:
        return

    # stats: {student_id: {"name": ..., "group": ..., "count": N}}
    stats = {}

    # All files in ATTENDANCE_DIR
    for filename in os.listdir(ATTENDANCE_DIR):
        if not filename.startswith("attendance_") or not filename.endswith(".csv"):
            continue

        # Extract date part: attendance_YYYY-MM-DD.csv
        date_part = filename[len("attendance_"):-4]  # we cutted "attendance_" and ".csv"

        try:
            file_date = datetime.date.fromisoformat(date_part)
        except ValueError:
            continue  # if file name and format not bwe correct

        if not (start_date <= file_date <= end_date):
            continue  # dont in between

        attendance_file = os.path.join(ATTENDANCE_DIR, filename)

        with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "").strip()
                name = row.get("name", "")
                group = row.get("group", "")

                if not sid:
                    continue

                # IF student is login → count only student ID
                if CURRENT_ROLE == "student" and CURRENT_STUDENT_ID is not None:
                    if sid != str(CURRENT_STUDENT_ID):
                        continue

                if sid not in stats:
                    stats[sid] = {
                        "name": name,
                        "group": group,
                        "count": 0
                    }
                stats[sid]["count"] += 1

    if not stats:
        messagebox.showinfo(
            "No Data",
            f"No attendance records found between {start_date} and {end_date}.",
            parent=root
        )
        return

    # Show results in a window
    win = tk.Toplevel(root)
    win.title(f"Attendance Stats: {start_date} → {end_date}")
    win.geometry("900x300")

    tree = ttk.Treeview(win, columns=("id", "name", "group", "count"), show="headings")
    tree.heading("id", text="Student ID")
    tree.heading("name", text="Name")
    tree.heading("group", text="Group")
    tree.heading("count", text="Total Attendance")
    tree.pack(fill=tk.BOTH, expand=True)

    # Sort stats by count descending 
    for sid, info in sorted(stats.items(), key=lambda item: item[1]["count"], reverse=True):
        tree.insert(
            "",
            tk.END,
            values=(sid, info["name"], info["group"], info["count"])
        )


def view_attendance_by_date():
    """
    Asks user to enter a date (YYYY-MM-DD),
    then reads the attendance file for that date
    and displays it in a table.
    If logged in as student, shows only that student's records.
    """
    # Ask for date
    date_str = simpledialog.askstring(
        "Select Date",
        "Enter date (YYYY-MM-DD):",
        parent=root
    )

    if not date_str:
        return  # User cancelled
    
    # Validate date format
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.", parent=root)
        return
    
    # File path for the selected date
    attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
    
    if not os.path.isfile(attendance_file):
        messagebox.showwarning("Not Found", f"No attendance found for {date_str}.", parent=root)
        return
    group_filter = ask_group_filter()

    # Open new window
    win = tk.Toplevel(root)
    win.title(f"Attendance - {date_str}")
    win.geometry("800x300")

    tree = ttk.Treeview(win, columns=("id", "name", "group", "time"), show="headings")
    tree.heading("id", text="Student ID")
    tree.heading("name", text="Name")
    tree.heading("group", text="Group")
    tree.heading("time", text="Time")
    tree.pack(fill=tk.BOTH, expand=True)

    # Read from file
    with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:

        
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("student_id", "")
            name = row.get("name", "")
            group = row.get("group", "")
            time_ = row.get("time", "")

            # If
            if CURRENT_ROLE == "student" and CURRENT_STUDENT_ID is not None:
                if sid != str(CURRENT_STUDENT_ID):
                    continue

            
            if group_filter is not None and group != group_filter:
                continue

            tree.insert(
                "",
                tk.END,
                values=(sid, name, group, time_)
            )


def view_attendance():
    """
    Reads today's attendance file and displays it in a table.
    """
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    attendance_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today_str}.csv")

    if not os.path.isfile(attendance_file):
        messagebox.showinfo("Information", f"There is no attendance file for today.\n({attendance_file})", parent=root)
        return
    
    
    group_filter = ask_group_filter()

    # New window(Toplevel)
    win = tk.Toplevel(root)
    win.title(f"Attendance - {today_str}")
    win.geometry("800x300")

    tree = ttk.Treeview(win, columns=("id", "name", "group", "time"), show="headings")
    tree.heading("id", text="Student ID")
    tree.heading("name", text="Name")
    tree.heading("group", text="Group")
    tree.heading("time", text="Time")
    tree.pack(fill=tk.BOTH, expand=True)

    # Just to see what's happening (debug)
    print("Reading attendance from:", attendance_file)

    with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0

        

        for row in reader:
            row_count += 1
            sid = row.get("student_id", "")
            name = row.get("name", "")
            group = row.get("group", "")
            time_ = row.get("time", "")

            # Student filter
            # If logged in as student, show only their records
            if CURRENT_ROLE == "student" and CURRENT_STUDENT_ID is not None:
                if sid != str(CURRENT_STUDENT_ID):
                    continue

            # Group filter
            if group_filter is not None and group != group_filter:
                continue

            tree.insert(
                "",
                tk.END,
                values=(sid, name, group, time_)
            )

    print("Total rows read:", row_count)


# Exit application function
def exit_app():
    root.destroy()



# ============================ UI Components ============================

title_label = ctk.CTkLabel(
    root,
    text="Smart Attendance System",
    font=("Helvetica", 26, "bold")
)
title_label.pack(pady=(40, 20))

btn_register = ctk.CTkButton(
    root,
    text="Register Student",
    font=("Helvetica", 14),
    width=220,
    height=40,
    corner_radius=20,
    state="disabled",
    command=register_student
)
btn_register.pack(pady=5)

btn_train = ctk.CTkButton(
    root,
    text="Train Model",
    font=("Helvetica", 14),
    width=220,
    height=40,
    corner_radius=20,
    state="disabled",
    command=train_model
)
btn_train.pack(pady=5)

btn_attendance = ctk.CTkButton(
    root,
    text="Take Attendance",
    font=("Helvetica", 14),
    width=220,
    height=40,
    corner_radius=20,
    state="disabled",
    command=take_attendance
)
btn_attendance.pack(pady=5)

btn_view = ctk.CTkButton(
    root,
    text="View Attendance",
    font=("Helvetica", 14),
    width=220,
    height=40,
    state="disabled",
    corner_radius=20,
    command=view_attendance
)
btn_view.pack(pady=5)

btn_view_by_date = ctk.CTkButton(
    root,
    text="View Attendance by Date",
    font=("Helvetica", 14),
    width=220,
    height=40,
    state="disabled",
    corner_radius=20,
    command=view_attendance_by_date
)
btn_view_by_date.pack(pady=5)

btn_stats_range = ctk.CTkButton(
    root,
    text="Attendance Stats (Date Range)",
    font=("Helvetica", 14),
    width=220,
    height=40,
    state="disabled",
    corner_radius=20,
    command=view_attendance_stats_range
)
btn_stats_range.pack(pady=5)

btn_exit = ctk.CTkButton(
    root,
    text="Exit",
    font=("Helvetica", 14),
    width=220,
    height=40,
    fg_color="#FF5555",
    hover_color="#CC4444",
    corner_radius=20,
    command=exit_app
)
btn_exit.pack(pady=(20, 10))


open_login_window()  # Open the login window on start

root.mainloop()  # Start the Tkinter event loop