from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('recognize_face/', views.recognize_face, name='recognize_face'),
]