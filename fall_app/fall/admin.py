from django.contrib import admin
from .models import Caretaker, User, Patient, FallHistory

# Register your models here.
admin.site.register(Caretaker)
admin.site.register(User)
admin.site.register(Patient)
admin.site.register(FallHistory)
