
from django.urls import path, include
from django.conf.urls import url
from StaffTurnoverApp import views

urlpatterns = [
    path('', views.index, name='index'),
]
