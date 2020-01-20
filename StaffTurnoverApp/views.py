from django.shortcuts import render, redirect
from .models import UserInfo
import csv


def index(request):
    return render(request, 'index.html')

def saveUserInfo(request):
    if request.method == 'POST':
        user_info = UserInfo()
        user_info.email = request.POST['email']
        user_info.companyName = request.POST['organization']
        user_info.save()

    data = csv.reader(request.FILES['employee_data'])
    print('csv obj------------------------', data)
    return redirect('index')
