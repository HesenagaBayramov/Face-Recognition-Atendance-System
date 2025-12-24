import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import customtkinter as ctk
import csv
import os
import cv2
import numpy as np
import datetime

class SmartAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("600x600")
        self.root.grid_columnconfigure(0, weight=1)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Constants and Directories
        self.STUDENTS_FILE = "students.csv"
        self.DATASET_DIR = "dataset"
        self.ATTENDANCE_DIR = "attendance"
        self.HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
        self.TRAINER_FILE = "trainer.yml"

        os.makedirs(self.DATASET_DIR, exist_ok=True)
        os.makedirs(self.ATTENDANCE_DIR, exist_ok=True)

        # Roles and Users
        self.CURRENT_ROLE = None    #This will store the role you are currently logged in with.
        self.CURRENT_USER_NAME = None   #admin/teacher username
        self.CURRENT_STUDENT_ID = None  #student ID if logged in as student


        self.ROLES = ["admin", "teacher", "student"]
        self.ADMINS = {"admin": "adminadmin"}
        self.TEACHERS = {"teacher1": "teach123", "teacher2": "teach234"}
        self.GROUPS = [
            "ZU-054", "ZU-044", "ZU-034", "ZU-024", "ZU-014", "BS-08.24"
        ]

        self._create_widgets()

    #-------------------- UI Components --------------------
    def _create_widgets(self):
        # Title
        self.title_label = ctk.CTkLabel(
            self.root,
            text="Smart Attendance System",
            font=("Helvetica", 26, "bold")
        )
        self.title_label.grid(row=0, column=0, pady=(40, 20), sticky="n")

        # Buttons
        self.btn_register = ctk.CTkButton(
            self.root,
            text="Register Student",
            font=("Helvetica", 14),
            width=220,
            height=40,
            corner_radius=20,
            state="disabled",
            command=self.register_student
        )
        self.btn_register.grid(row=1, column=0, pady=5)

        self.btn_train = ctk.CTkButton(
            self.root,
            text="Train Model",
            font=("Helvetica", 14),
            width=220,
            height=40,
            corner_radius=20,
            state="disabled",
            command=self.train_model
        )
        self.btn_train.grid(row=2, column=0, pady=5)

        self.btn_attendance = ctk.CTkButton(
            self.root,
            text="Take Attendance",
            font=("Helvetica", 14),
            width=220,
            height=40,
            corner_radius=20,
            state="disabled",
            command=self.take_attendance,
            fg_color="green"
        )
        self.btn_attendance.grid(row=3, column=0, pady=5)

        self.btn_view = ctk.CTkButton(
            self.root,
            text="View Attendance",
            font=("Helvetica", 14),
            width=220,
            height=40,
            state="disabled",
            corner_radius=20,
            command=self.view_attendance
        )
        self.btn_view.grid(row=4, column=0, pady=5)

        self.btn_view_by_date = ctk.CTkButton(
            self.root,
            text="View Attendance by Date",
            font=("Helvetica", 14),
            width=220,
            height=40,
            state="disabled",
            corner_radius=20,
            command=self.view_attendance_by_date
        )
        self.btn_view_by_date.grid(row=5, column=0, pady=5)

        self.btn_stats_range = ctk.CTkButton(
            self.root,
            text="Attendance Stats (Date Range)",
            font=("Helvetica", 14),
            width=220,
            height=40,
            state="disabled",
            corner_radius=20,
            command=self.view_attendance_stats_range
        )
        self.btn_stats_range.grid(row=6, column=0, pady=5)

        self.btn_exit = ctk.CTkButton(
            self.root,
            text="Exit",
            font=("Helvetica", 14),
            width=220,
            height=40,
            fg_color="#FF5555",
            hover_color="#CC4444",
            corner_radius=20,
            command=self.exit_app
        )
        self.btn_exit.grid(row=7, column=0, pady=(20, 10))
        
        # Author
        self.author_label = ctk.CTkLabel(
            self.root,
            text="Bayramof",
            font=("Helvetica", 12, "bold")
        )
        self.author_label.grid(row=8, column=0, pady=(10, 10), sticky="e", padx=20)


    def configre_buttons_by_role(self):
        """
        Enables/disables the primary buttons based on the value of self.CURRENT_ROLE.
        This function will be called after a successful login.
        """
        if self.CURRENT_ROLE == "admin":
            self.btn_register.configure(state="normal")
            self.btn_train.configure(state="normal")
            self.btn_attendance.configure(state="normal")
            self.btn_view.configure(state="normal")
            self.btn_view_by_date.configure(state="normal")
            self.btn_stats_range.configure(state="normal")

        elif self.CURRENT_ROLE == "teacher":
            self.btn_register.configure(state="disabled")
            self.btn_train.configure(state="disabled")
            self.btn_attendance.configure(state="normal")
            self.btn_view.configure(state="normal")
            self.btn_view_by_date.configure(state="normal")
            self.btn_stats_range.configure(state="normal")

        elif self.CURRENT_ROLE == "student":
            self.btn_register.configure(state="disabled")
            self.btn_train.configure(state="disabled")
            self.btn_attendance.configure(state="disabled")
            self.btn_view.configure(state="normal")
            self.btn_view_by_date.configure(state="normal")
            self.btn_stats_range.configure(state="normal")

        else:
            self.btn_register.configure(state="disabled")
            self.btn_train.configure(state="disabled")
            self.btn_attendance.configure(state="disabled")
            self.btn_view.configure(state="disabled")
            self.btn_view_by_date.configure(state="disabled")
            self.btn_stats_range.configure(state="disabled")

    def open_login_window(self):
        """
        When the program is opened, it opens a separate window for Admin / Teacher / Student selection.
        It calls the login function corresponding to each role.
        """
        login_win = ctk.CTkToplevel(self.root)
        login_win.title("Login - Role Selection")
        login_win.geometry("300x250")
        login_win.grab_set()  # Make this window modal

        # This makes the login window always on top of the main window
        login_win.transient(self.root)    # Stays on top of root
        login_win.lift()             # Go on top of the root window
        login_win.focus_force()      # Focus claviature

        # User can't change window size
        login_win.resizable(False, False)

        title = ctk.CTkLabel(
            login_win,
            text="Select Role to Login",
            font=("Helvetica", 18, "bold")
        )
        title.pack(pady=(20, 10))

        # Frame for buttons
        btn_frame = ctk.CTkFrame(login_win)
        btn_frame.pack(pady=10, padx=20, fill="x")

        btn_admin = ctk.CTkButton(
            btn_frame,
            text="Admin",
            width=200,
            height=36,
            corner_radius=18,
            command=lambda: self.admin_login(login_win)
        )
        btn_admin.pack(pady=5)

        btn_teacher = ctk.CTkButton(
            btn_frame,
            text="Teacher",
            width=200,
            height=36,
            corner_radius=18,
            command=lambda: self.teacher_login(login_win)
        )
        btn_teacher.pack(pady=5)

        btn_student = ctk.CTkButton(
            btn_frame,
            text="Student",
            width=200,
            height=36,
            corner_radius=18,
            command=lambda: self.student_login(login_win)
        )
        btn_student.pack(pady=5)

    def admin_login(self, win):
        """Handles admin login."""
        username = simpledialog.askstring("Admin Login", "Enter Admin Username:", parent=win)
        if username is None:
            return  # User cancelled

        password = simpledialog.askstring("Admin Login", "Enter Admin Password:", show='*', parent=win)
        if password is None:
            return  # User cancelled

        # Simple check
        if self.ADMINS.get(username) == password:
            self.CURRENT_ROLE = "admin"
            self.CURRENT_USER_NAME = username
            messagebox.showinfo("Login Successful", f"Welcome, Admin {username}!", parent=win)
            self.configre_buttons_by_role()
            win.destroy() # Close login window
        else:
            messagebox.showerror("Login Failed", "Invalid admin credentials.", parent=win)

    def teacher_login(self, win):
        """Handles teacher login."""
        username = simpledialog.askstring("Teacher Login", "Enter Teacher Username:", parent=win)
        if username is None:
            return  # User cancelled

        password = simpledialog.askstring("Teacher Login", "Enter Teacher Password:", show='*', parent=win)
        if password is None:
            return  # User cancelled

        # Simple check
        if self.TEACHERS.get(username) == password:
            self.CURRENT_ROLE = "teacher"
            self.CURRENT_USER_NAME = username
            messagebox.showinfo("Login Successful", f"Welcome, Teacher {username}!", parent=win)
            self.configre_buttons_by_role()
            win.destroy() # Close login window
        else:
            messagebox.showerror("Login Failed", "Invalid teacher credentials.", parent=win)

    def student_login(self, win):
        """Handles student login."""
        student_id = simpledialog.askstring("Student Login", "Enter Student ID:", parent=win)
        if student_id is None:
            return  # User cancelled

        # Check if student ID is numeric
        if not student_id.isdigit():
            messagebox.showerror("Error", "Student ID must be an numric.", parent=win)
            return

        sid = int(student_id)
        students = self.load_students() # Load students from file

        if sid in students:
            info = students[sid]
            name = info.get("name", "")
            group = info.get("group", "")

            self.CURRENT_ROLE = "student"
            self.CURRENT_STUDENT_ID = sid

            messagebox.showinfo("Login Successful",
                                f"Welcome, {name}!\nID: {sid}\nGroup: {group}",
                                parent=win
            )

            self.configre_buttons_by_role()
            win.destroy() # Close login window

        else:
            messagebox.showerror("Error", "No student found this ID", parent=win)

    def ensure_students_file(self):
        """If students.csv does not exist, it creates it with a title."""
        if not os.path.exists(self.STUDENTS_FILE):
            with open(self.STUDENTS_FILE, mode='w', newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["student_id", "name", "group"])

    def add_student_to_file(self, student_id: str, name: str, group: str) -> bool:
        """Adds a new student to the file. Returns False if the ID already exists."""
        self.ensure_students_file()

        existing_ids = set()
        try:
            with open(self.STUDENTS_FILE, mode='r', newline='', encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_ids.add(row["student_id"])
        except FileNotFoundError:
            pass  # File will be created by ensure_students_file

        if student_id in existing_ids:
            return False  # ID already exists

        with open(self.STUDENTS_FILE, mode='a', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([student_id, name, group])

        return True

    def load_students(self):
        """
        Returns {id: {"name": name, "group": group}} dict from students.csv.
        """
        self.ensure_students_file()
        students = {}

        with open(self.STUDENTS_FILE, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "").strip()
                if sid.isdigit():
                    students[int(sid)] = {
                        "name": row.get("name", "") or "",
                        "group": row.get("group", "") or "",
                    }
        return students

    def capture_face_for_student(self, student_id: str):
        """
        Captures facial images from the camera for the given student ID
        and writes them to the dataset folder in the format User.<id>.<say>.jpg.
        """
        if not os.path.exists(self.HAAR_CASCADE_FILE):
            messagebox.showerror(
                "Error",
                f"Haarcascade file '{self.HAAR_CASCADE_FILE}' not found."
            )
            return

        face_cascade = cv2.CascadeClassifier(self.HAAR_CASCADE_FILE)
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        messagebox.showinfo(
            "Information",
            "Camera is on.\n"
            "Face the camera and move slightly left and right.\n"
            "You can press 'q' to exit.\n"
            "About 100 pictures will be taken."
        )

        count = 0
        target_count = 100

        try:
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                for (x, y, w, h) in faces:
                    count += 1
                    face_img = gray[y:y + h, x:x + w]
                    file_name = os.path.join(self.DATASET_DIR, f"User.{student_id}.{count}.jpg")
                    cv2.imwrite(file_name, face_img)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.imshow('Face Capture - q = exit', frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or count >= target_count:
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()

        messagebox.showinfo(
            "Success",
            f"{count} face images captured for Student ID {student_id}."
        )

    def ask_group(self):
        """Opens a small window with a Combobox to select a group.
        Returns selected group string or None if canceled.
        """
        win = ctk.CTkToplevel(self.root)
        win.title("Select Group")
        win.geometry("300x200")
        win.transient(self.root)
        win.grab_set()
        win.lift()
        win.focus_force()

        title = ctk.CTkLabel(win, text="Select student's group:", font=("Helvetica", 13))
        title.pack(pady=(15, 10))

        group_var = tk.StringVar()
        # Here Group current list (ZU-054, ZU-044 etc.)
        combo = ctk.CTkComboBox(win, variable=group_var, values=self.GROUPS, state="readonly", width=200, height=32, corner_radius=12)
        if self.GROUPS:
            combo.set(self.GROUPS[0])
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
        btn_ok = ctk.CTkButton(btn_frame, text="OK", width=80, height=32, corner_radius=16, command=on_ok)
        btn_ok.pack(side="left", padx=5)
        btn_cancel = ctk.CTkButton(btn_frame, text="Cancel", width=80, height=32, corner_radius=16, fg_color="#555555", hover_color="#444444", command=on_cancel)
        btn_cancel.pack(side="left", padx=5)

        win.wait_window()
        return result["value"]

    def register_student(self):
        student_id = simpledialog.askstring("Student ID", "Enter Student ID:", parent=self.root)
        if not student_id:
            return
        if not student_id.isdigit():
            messagebox.showerror("Invalid ID", "Student ID must be an integer.", parent=self.root)
            return

        name = simpledialog.askstring("Student Name", "Enter Student Name:", parent=self.root)
        if not name:
            return

        group = self.ask_group()
        if group is None:
            messagebox.showwarning("Cancelled", "Group selection cancelled.", parent=self.root)
            return

        if not self.add_student_to_file(student_id, name, group):
            messagebox.showerror("Error! Duplicated ID", f"Student ID {student_id} already exists.", parent=self.root)
            return
        else:
            messagebox.showinfo("Success", f"Student {name} (ID: {student_id}, Group: {group}) registered successfully.", parent=self.root)

        messagebox.showinfo(
            "Success",
            f"Student added:\nID: {student_id}\nName: {name}\nGroup: {group}\n\n"
            "Face images will now be taken.",
            parent=self.root
        )
        self.capture_face_for_student(student_id)

    def train_model(self):
        """
        Trains an LBPH model from the face images in the dataset/ directory
        and writes them to the trainer.yml file.
        """
        image_paths = [os.path.join(self.DATASET_DIR, f) for f in os.listdir(self.DATASET_DIR) if f.lower().endswith('.jpg')]
        if not image_paths:
            messagebox.showwarning(
                "Attention",
                "Dataset folder is empty.\nYou must first collect face images with Register Student."
            )
            return

        face_samples = []
        ids = []
        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            filename = os.path.basename(image_path)
            parts = filename.split('.')
            if len(parts) >= 3 and parts[1].isdigit():
                student_id = int(parts[1])
                ids.append(student_id)
                face_samples.append(img)
        if not face_samples:
            messagebox.showerror(
                "Error",
                "Could not read student IDs from the dataset."
            )
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(self.TRAINER_FILE)

        messagebox.showinfo(
            "Success",
            f"The model was successfully trained and written to the file '{self.TRAINER_FILE}'."
        )

    def take_attendance(self):
        """
        It recognizes the face using the trained model (trainer.yml),
        writes student_id, name, time to the attendance_<date>.csv file.
        """
        if not os.path.isfile(self.TRAINER_FILE):
            messagebox.showerror("Error", "You must 'Train Model' first. trainer.yml not found.")
            return
        if not os.path.isfile(self.HAAR_CASCADE_FILE):
            messagebox.showerror("Error", f"{self.HAAR_CASCADE_FILE} file not found.")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.TRAINER_FILE)
        face_cascade = cv2.CascadeClassifier(self.HAAR_CASCADE_FILE)
        students = self.load_students()
        if not students:
            messagebox.showwarning("Attention", "Student list is empty. Add a student first.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera could not be opened.")
            return

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        attendance_file = os.path.join(self.ATTENDANCE_DIR, f"attendance_{today_str}.csv")
        file_exists = os.path.isfile(attendance_file)
        attended_ids = set()

        if file_exists:
            with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("student_id", "").isdigit():
                        attended_ids.add(int(row["student_id"]))

        with open(attendance_file, mode="a", newline="", encoding="utf-8") as f_att:
            writer = csv.writer(f_att)
            if not file_exists:
                writer.writerow(["student_id", "name", "group", "time"])

            messagebox.showinfo(
                "Information",
                "Camera turned on.\n"
                "Students take turns looking at the camera.\n"
                "Press 'q' to exit."
            )

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

                    for (x, y, w, h) in faces:
                        face_img = gray[y:y + h, x:x + w]
                        student_id_pred, confidence = recognizer.predict(face_img)

                        if confidence < 70:
                            info = students.get(student_id_pred)
                            name = info.get("name", "Unknown") if info else "Unknown"
                            group = info.get("group", "") if info else ""
                            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            if student_id_pred not in attended_ids and name != "Unknown":
                                attended_ids.add(student_id_pred)
                                now_time = datetime.datetime.now().strftime("%H:%M:%S")
                                writer.writerow([student_id_pred, name, group, now_time])
                                print(f"Attendance: {student_id_pred} - {name} ({group})- {now_time}")
                        else:
                            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    cv2.imshow("Attendance - q = exit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()

        messagebox.showinfo("Success", "Attendance session ended.")

    def ask_date_range(self):
        """
        Asks user for a start and end date (YYYY-MM-DD).
        Returns (start_date, end_date) as datetime.date objects,
        or (None, None) if cancelled or invalid.
        """
        start_str = simpledialog.askstring("Date Range", "Enter START date (YYYY-MM-DD):", parent=self.root)
        if not start_str:
            return None, None
        end_str = simpledialog.askstring("Date Range", "Enter END date (YYYY-MM-DD):", parent=self.root)
        if not end_str:
            return None, None
        try:
            start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD.", parent=self.root)
            return None, None
        if end_date < start_date:
            messagebox.showerror("Error", "End date cannot be before start date.", parent=self.root)
            return None, None
        return start_date, end_date

    def view_attendance_stats_range(self):
        """
        Shows how many times each student attended between a date range.
        If logged in as a student, shows only that student's count.
        """
        start_date, end_date = self.ask_date_range()
        if start_date is None or end_date is None:
            return

        stats = {}
        for filename in os.listdir(self.ATTENDANCE_DIR):
            if not filename.startswith("attendance_") or not filename.endswith(".csv"):
                continue
            date_part = filename[len("attendance_"):-4]
            try:
                file_date = datetime.date.fromisoformat(date_part)
            except ValueError:
                continue
            if not (start_date <= file_date <= end_date):
                continue

            attendance_file = os.path.join(self.ATTENDANCE_DIR, filename)
            with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("student_id", "").strip()
                    if not sid:
                        continue
                    if self.CURRENT_ROLE == "student" and self.CURRENT_STUDENT_ID is not None and sid != str(self.CURRENT_STUDENT_ID):
                        continue
                    if sid not in stats:
                        stats[sid] = {"name": row.get("name", ""), "group": row.get("group", ""), "count": 0}
                    stats[sid]["count"] += 1
        if not stats:
            messagebox.showinfo("No Data", f"No attendance records found between {start_date} and {end_date}.", parent=self.root)
            return

        win = tk.Toplevel(self.root)
        win.title(f"Attendance Stats: {start_date} → {end_date}")
        win.geometry("900x300")
        tree = ttk.Treeview(win, columns=("id", "name", "group", "count"), show="headings")
        tree.heading("id", text="Student ID")
        tree.heading("name", text="Name")
        tree.heading("group", text="Group")
        tree.heading("count", text="Total Attendance")
        tree.pack(fill=tk.BOTH, expand=True)
        for sid, info in sorted(stats.items(), key=lambda item: item[1]["count"], reverse=True):
            tree.insert("", tk.END, values=(sid, info["name"], info["group"], info["count"]))

    def ask_group_filter(self):
        """Opens a small window with a Combobox to select a group filter.
        Returns selected group string or None if canceled.
        """
        win = tk.Toplevel(self.root)
        win.title("Select Group")
        win.geometry("300x200")
        win.grab_set()
        label = tk.Label(win, text="Select group to filter:", font=("Helvetica", 11))
        label.pack(pady=10)
        selected_group = tk.StringVar()
        combo = ttk.Combobox(win, textvariable=selected_group, values=["All"] + self.GROUPS, state="readonly")
        combo.current(0)
        combo.pack(pady=5)
        result = {"group": None}
        def apply_filter():
            group = selected_group.get()
            result["group"] = None if group == "All" else group
            win.destroy()
        btn = tk.Button(win, text="Apply Filter", command=apply_filter)
        btn.pack(pady=10)
        win.wait_window()
        return result["group"]

    def view_attendance_by_date(self):
        """
        Asks user to enter a date (YYYY-MM-DD),
        then reads the attendance file for that date
        and displays it in a table.
        If logged in as student, shows only that student's records.
        """
        date_str = simpledialog.askstring("Select Date", "Enter date (YYYY-MM-DD):", parent=self.root)
        if not date_str:
            return
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.", parent=self.root)
            return

        attendance_file = os.path.join(self.ATTENDANCE_DIR, f"attendance_{date_str}.csv")
        if not os.path.isfile(attendance_file):
            messagebox.showwarning("Not Found", f"No attendance found for {date_str}.", parent=self.root)
            return
        group_filter = self.ask_group_filter()
        win = tk.Toplevel(self.root)
        win.title(f"Attendance - {date_str}")
        win.geometry("800x300")
        tree = ttk.Treeview(win, columns=("id", "name", "group", "time"), show="headings")
        tree.heading("id", text="Student ID")
        tree.heading("name", text="Name")
        tree.heading("group", text="Group")
        tree.heading("time", text="Time")
        tree.pack(fill=tk.BOTH, expand=True)

        with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "")
                if self.CURRENT_ROLE == "student" and self.CURRENT_STUDENT_ID is not None and sid != str(self.CURRENT_STUDENT_ID):
                    continue
                if group_filter is not None and row.get("group", "") != group_filter:
                    continue
                tree.insert("", tk.END, values=(sid, row.get("name", ""), row.get("group", ""), row.get("time", "")))

    def view_attendance(self):
        """
        Reads today's attendance file and displays it in a table.
        """
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        attendance_file = os.path.join(self.ATTENDANCE_DIR, f"attendance_{today_str}.csv")
        if not os.path.isfile(attendance_file):
            messagebox.showinfo("Information", f"There is no attendance file for today.\n({attendance_file})", parent=self.root)
            return
        group_filter = self.ask_group_filter()
        win = tk.Toplevel(self.root)
        win.title(f"Attendance - {today_str}")
        win.geometry("800x300")
        tree = ttk.Treeview(win, columns=("id", "name", "group", "time"), show="headings")
        tree.heading("id", text="Student ID")
        tree.heading("name", text="Name")
        tree.heading("group", text="Group")
        tree.heading("time", text="Time")
        tree.pack(fill=tk.BOTH, expand=True)

        with open(attendance_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("student_id", "")
                if self.CURRENT_ROLE == "student" and self.CURRENT_STUDENT_ID is not None and sid != str(self.CURRENT_STUDENT_ID):
                    continue
                if group_filter is not None and row.get("group", "") != group_filter:
                    continue
                tree.insert("", tk.END, values=(sid, row.get("name", ""), row.get("group", ""), row.get("time", "")))
    
    def exit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = SmartAttendanceSystem(root)
    app.open_login_window()
    root.mainloop()
