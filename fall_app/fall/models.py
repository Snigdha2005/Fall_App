from django.db import models
import json

def email_default():
    return json.dumps({"email": "to1@example.com"})

class Caretaker(models.Model):
    name = models.CharField(max_length=200)
    id_no = models.IntegerField(unique=True)
    email_id = models.TextField("Email ID", default=email_default)
    caretaker_phone_no = models.CharField(max_length=15, unique=True)

    def __str__(self):
        return self.name, self.id_no, self.caretaker_phone_no

class User(models.Model):
    CARETAKER = "CT"
    PATIENT = "PA"
    USER_CHOICES = [
        (CARETAKER, "Caretaker"),
        (PATIENT, "Patient"),
    ]
    user_type = models.CharField(max_length=2, choices=USER_CHOICES, default=PATIENT)

    def __str__(self):
        return self.get_user_type_display()

class Patient(models.Model):
    patient_name = models.CharField(max_length=200)
    patient_id = models.IntegerField(unique=True)
    patient_phone_no = models.CharField(max_length=15, unique=True)
    patient_email_id = models.TextField("Email ID", default=email_default)
    caretaker = models.ForeignKey(Caretaker, on_delete=models.CASCADE, related_name='patients')

    def __str__(self):
        return self.patient_name, self.patient_phone_no, (self.caretaker).caretaker_phone_no

class FallHistory(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='fall_histories')
    fall_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.patient.patient_name} fell at {self.fall_time}"

