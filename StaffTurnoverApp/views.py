from django.shortcuts import render, redirect
from .models import UserInfo,EmplData
import csv
from StaffTurnoverApp.functions import *

def index(request):
    return render(request, 'index.html')

def saveUserInfo(request):
    if request.method == 'POST':
        #user_info = UserInfo()
        # user_info.email = request.POST['email']
        # user_info.companyName = request.POST['organization']
        # user_info.save()
        # file_reader = TextIOWrapper(request.FILES['employee_data'].file, encoding=request.encoding)
        # reader = csv.reader(file_reader)
        # next(reader, None)
        # graphic = staffTurnoverResult(reader)
        columns_mapping_dict = {}
        columns_mapping_dict["age"] = request.POST['age']
        columns_mapping_dict["dailyRate"] = request.POST['dailyRate']
        columns_mapping_dict["distanceFromHome"] = request.POST['distanceFromHome']
        columns_mapping_dict["environmentSatisfaction"] = request.POST['environmentSatisfaction']
        columns_mapping_dict["jobLevel"] = request.POST['jobLevel']
        columns_mapping_dict["jobRole"] = request.POST['jobRole']
        columns_mapping_dict["maritalStatus"] = request.POST['maritalStatus']
        columns_mapping_dict["monthlyIncome"] = request.POST['monthlyIncome']
        columns_mapping_dict["overTime"] = request.POST['overTime']
        columns_mapping_dict["percentSalaryHike"] = request.POST['percentSalaryHike']
        columns_mapping_dict["relationshipSatisfaction"] = request.POST['relationshipSatisfaction']
        columns_mapping_dict["totalWorkingYears"] = request.POST['totalWorkingYears']

        emp_data_csv = request.FILES['employee_data']
        emp_data_dataframe = pd.read_csv(emp_data_csv)
        graphic = staffTurnoverResult(columns_mapping_dict, emp_data_dataframe)
        # emp_data= EmplData()
        # for row in reader:
        #         empl_data = EmplData.objects.create(
        #             Age=row[0],
        #             Attrition=row[1],
        #             BusinessTravel=row[2],
        #             DailyRate=row[3],
        #             Department=row[4],
        #             DistanceFromHome=row[5],
        #             Education=row[6],
        #             EducationField=row[7],
        #             EmployeeCount=row[8],
        #             EmployeeNumber=row[9],
        #             EnvironmentSatisfaction=row[10],
        #             Gender=row[11],
        #             HourlyRate=row[12],
        #             JobInvolvement=row[13],
        #             JobLevel=row[14],
        #             JobRole=row[15],
        #             JobSatisfaction=row[16],
        #             MaritalStatus=row[17],
        #             MonthlyIncome=row[18],
        #             MonthlyRate=row[19],
        #             NumCompaniesWorked=row[20],
        #             Over18=row[21],
        #             OverTime=row[22],
        #             PercentSalaryHike=row[23],
        #             PerformanceRating=row[24],
        #             RelationshipSatisfaction=row[25],
        #             StandardHours=row[26],
        #             StockOptionLevel=row[27],
        #             TotalWorkingYears=row[28],
        #             TrainingTimesLastYear=row[29],
        #             WorkLifeBalance=row[30],
        #             YearsAtCompany=row[31],
        #             YearsInCurrentRole=row[32],
        #             YearsSinceLastPromotion=row[33],
        #             YearsWithCurrManager=row[34],
        #             userId=UserInfo.objects.latest('id')
        #         )
                # print(row[0],row[1],row[2])

    return render(request, 'prediction_dashboard.html', {'graphic':graphic})

def employee(request):
    return render(request, 'employee.html')

def saveEmpInfo(request):
    if request.method == 'POST':
        columns_mapping_dict = {}
        columns_mapping_dict["age"] = request.POST['age']
        columns_mapping_dict["dailyRate"] = request.POST['dailyRate']
        columns_mapping_dict["distanceFromHome"] = request.POST['distanceFromHome']
        columns_mapping_dict["environmentSatisfaction"] = request.POST['environmentSatisfaction']
        columns_mapping_dict["jobLevel"] = request.POST['jobLevel']
        columns_mapping_dict["jobRole"] = request.POST['jobRole']
        columns_mapping_dict["maritalStatus"] = request.POST['maritalStatus']
        columns_mapping_dict["monthlyIncome"] = request.POST['monthlyIncome']
        columns_mapping_dict["overTime"] = request.POST['overTime']
        columns_mapping_dict["percentSalaryHike"] = request.POST['percentSalaryHike']
        columns_mapping_dict["relationshipSatisfaction"] = request.POST['relationshipSatisfaction']
        columns_mapping_dict["totalWorkingYears"] = request.POST['totalWorkingYears']
        emp_data_csv = request.FILES['employee_data']
        emp_data_dataframe = pd.read_csv(emp_data_csv)
        graphic = staffTurnoverResult(columns_mapping_dict, emp_data_dataframe)
    return render(request, 'prediction_dashboard.html', {'graphic': graphic})