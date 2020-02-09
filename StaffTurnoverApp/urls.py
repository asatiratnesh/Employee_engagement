
from django.urls import path, include
from django.conf.urls import url
from StaffTurnoverApp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('^index/$', views.index, name='index'),
    path('saveUserInfo/', views.saveUserInfo, name='saveUserInfo'),
    path('employee/', views.employee, name='employee'),
    path('saveEmpInfo/', views.saveEmpInfo, name='saveEmpInfo'),
    path('saveUserInfo/emplLimeGraph/', views.emplLimeGraph, name='emplLimeGraph'),

]
