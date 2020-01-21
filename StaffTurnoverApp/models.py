from django.db import models

# Create your models here.

# User models
class UserInfo(models.Model):
    companyName = models.CharField(max_length=50)
    email = models.EmailField(max_length=254,unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.companyName

# User models


class EmplData(models.Model):
    userId = models.ForeignKey(UserInfo, on_delete=models.CASCADE)
    Age= models.CharField(max_length=3)
    Attrition= models.CharField(max_length=6)
    BusinessTravel= models.CharField(max_length=40)
    DailyRate= models.CharField(max_length=6)
    Department= models.CharField(max_length=40)
    DistanceFromHome= models.CharField(max_length=6)
    Education= models.CharField(max_length=6)
    EducationField= models.CharField(max_length=40)
    EmployeeCount= models.CharField(max_length=10)
    EmployeeNumber= models.CharField(max_length=10)
    EnvironmentSatisfaction= models.CharField(max_length=10)
    Gender= models.CharField(max_length=10)
    HourlyRate= models.CharField(max_length=6)
    JobInvolvement= models.CharField(max_length=6)
    JobLevel= models.CharField(max_length=6)
    JobRole= models.CharField(max_length=40)
    JobSatisfaction= models.CharField(max_length=6)
    MaritalStatus= models.CharField(max_length=20)
    MonthlyIncome= models.CharField(max_length=20)
    MonthlyRate= models.CharField(max_length=20)
    NumCompaniesWorked= models.CharField(max_length=10)
    Over18= models.CharField(max_length=6)
    OverTime= models.CharField(max_length=6)
    PercentSalaryHike= models.CharField(max_length=10)
    PerformanceRating= models.CharField(max_length=10)
    RelationshipSatisfaction= models.CharField(max_length=10)
    StandardHours= models.CharField(max_length=10)
    StockOptionLevel= models.CharField(max_length=10)
    TotalWorkingYears= models.CharField(max_length=10)
    TrainingTimesLastYear= models.CharField(max_length=10)
    WorkLifeBalance= models.CharField(max_length=10)
    YearsAtCompany= models.CharField(max_length=10)
    YearsInCurrentRole= models.CharField(max_length=10)
    YearsSinceLastPromotion= models.CharField(max_length=10)
    YearsWithCurrManager= models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.EmployeeNumber
