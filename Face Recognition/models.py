# models.py

class Admin:
    def __init__(self, username, password):
        self.username = username
        self.password = password

class Teacher:
    def __init__(self, username, password):
        self.username = username
        self.password = password

class Student:
    def __init__(self, student_id, name, group):
        self.student_id = student_id
        self.name = name
        self.group = group
