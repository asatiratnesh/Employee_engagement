from io import TextIOWrapper
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import squarify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split


#import squarify


def staffTurnoverResult(columns_mapping_dict, emp_data):
    # ml code goed here
    # df = emp_data
    # df['Gender'].replace({'Male': 0, 'Female': 1},inplace = True)
    # df['OverTime'].replace({'No': 0, 'Yes': 1},inplace = True)
    #
    # categorical_columns = ['BusinessTravel', 'Department', 'EducationField', "JobRole", "MaritalStatus"]
    #
    # df = pd.get_dummies(df, columns=categorical_columns)
    #
    # df.drop(["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours","Attrition"], axis = 1, inplace = True)
    #
    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    #
    # with open(settings.MEDIA_ROOT+"picklemodelRF.pkl", 'rb') as pickle_file:
    #     model = pickle.load(pickle_file)
    # predictionList = model.predict_proba(df)[:,1]
    #
    # result_df = emp_data.iloc[emp_data.index, :]
    #
    # result_df["predicted_values"] = predictionList
    # result_df['predicted_values'] = result_df['predicted_values'].multiply(100)
    #
    # cost_percentage = 0.214
    # cost_reduction = result_df.apply(lambda row: (row.MonthlyIncome * 14) * cost_percentage, axis = 1)
    # result_df["cost_reduction"] = cost_reduction
    #
    # df_top_five = result_df.head(20)
    #
    # empl_count_per_job = {}
    # job_roles = []
    # for i in emp_data[columns_mapping_dict['jobRole']]:
    #     if i in job_roles:
    #         empl_count_per_job[i] = empl_count_per_job[i]+1
    #     else:
    #         job_roles.append(i)
    #         empl_count_per_job[i] = 1
    # treemap_empl_job = plotEmplVsJob(empl_count_per_job);

    return limeGraph(emp_data)


def plotEmplVsJob(empl_count_per_job):

    df = pd.DataFrame(list(empl_count_per_job.items()), columns=['JobRole', 'EmplCount'])
    squarify.plot(sizes=df['EmplCount'], label=df['JobRole'], alpha=.4 )
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    treemap_empl_job = base64.b64encode(image_png)
    treemap_empl_job = treemap_empl_job.decode('utf-8')
    return treemap_empl_job



def limeGraph(empl_data):
    df = empl_data
    df['Attrition'].replace({'No': 0, 'Yes': 1},inplace = True)
    df['Gender'].replace({'Male': 0, 'Female': 1},inplace = True)
    df['OverTime'].replace({'No': 0, 'Yes': 1},inplace = True)
    lb_make = LabelEncoder()
    df['BusinessTravel']=lb_make.fit_transform(df['BusinessTravel'].astype(str))
    df['Department']=lb_make.fit_transform(df['Department'].astype(str))

    df['EducationField']=lb_make.fit_transform(df['EducationField'].astype(str))

    df['JobRole']=lb_make.fit_transform(df['JobRole'].astype(str))

    df['MaritalStatus']=lb_make.fit_transform(df['MaritalStatus'].astype(str))
    # Removing unecessary features

    df.drop(["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], axis = 1, inplace = True)
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.head()
    x, y = df.drop(["Attrition"], axis = 1), df.Attrition
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 888)
    # Random Forest

    model1 = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2,
                               oob_score=False, random_state=10)
    RF = model1.fit(x_train, y_train)
    score = model1.score(x_test, y_test)
    print('Random Forest model score is %0.4f' %score)
    y_predicted = RF.predict(x_test)

    sortignDF = x_test
    sortignDF["y_predicted"] = y_predicted

    #dd = sortignDF.loc[sortignDF['y_predicted'] == 1]
    # for i in range(19):
    #     predict_fn_rf = lambda x: RF.predict_proba(x).astype(float)
    #     X = x_train.values
    #     explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = x_train.columns,class_names=['No','Yes'],kernel_width=5)
    #     choosen_instance = x.loc[[i]].values[0]
    #     exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
    #     exp.show_in_notebook(show_all=False)
    #     print("dsfadsfsdfdfsdf", i)
    #     path = settings.MEDIA_ROOT+"'"
    #     print(path+i+".html")
    #     exp.save_to_file(path+i+".html")

    dfCost = empl_data.tail(294)
    print("This is DF cost-------",dfCost)
    dfCost["predicted_values"] = y_predicted
    # result_df = dfCost.iloc[dfCost.index, :]

    # result_df["predicted_values"] = y_predicted
    # result_df['predicted_values'] = result_df['predicted_values'].multiply(100)
    #
    #
    cost_percentage = 0.214
    cost_reduction = dfCost.apply(lambda row: (row.MonthlyIncome * 14) * cost_percentage, axis = 1)
    dfCost["cost_reduction"] = cost_reduction

    return dfCost, dfCost.head(20)
