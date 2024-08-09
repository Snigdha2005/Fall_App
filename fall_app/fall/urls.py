from django.urls import path
from . import views

urlpatterns = [path("evaluate_fall/", views.evaluate_fall, name="evaluate_fall"), ]
