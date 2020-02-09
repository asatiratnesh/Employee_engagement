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
def staffTurnoverResult(columns_mapping_dict):
    return limeGraph()


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



def limeGraph(empl_id=None, data=None):
    if data is not None:
        csv_empl = data
    else:
        csv_empl = pd.read_csv(settings.MEDIA_ROOT+"empl_data.csv")

    if 'AgeOfEMP' in csv_empl:
        csv_empl.rename(columns = {'AgeOfEMP':'Age'}, inplace = True)
        df = csv_empl[["Age", "DailyRate", "DistanceFromHome", "EnvironmentSatisfaction", "JobLevel",
          "JobRole", "MaritalStatus", "MonthlyIncome", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction",
          "TotalWorkingYears",'Attrition']]
    else:
        df = csv_empl[["Age", "DailyRate", "DistanceFromHome", "EnvironmentSatisfaction", "JobLevel",
          "JobRole", "MaritalStatus", "MonthlyIncome", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction",
          "TotalWorkingYears",'Attrition']]

    df['Attrition'].replace({'No': 0, 'Yes': 1},inplace = True)
    # df['Gender'].replace({'Male': 0, 'Female': 1},inplace = True)
    df['OverTime'].replace({'No': 0, 'Yes': 1},inplace = True)

    lb_make = LabelEncoder()
    df['JobRole']=lb_make.fit_transform(df['JobRole'].astype(str))
    df['MaritalStatus']=lb_make.fit_transform(df['MaritalStatus'].astype(str))
    # Removing unecessary features
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.head()
    y1 = df.drop(["Attrition"], axis = 1)


    with open(settings.MEDIA_ROOT+"train_12model.pkl", 'rb') as pickle_file:
         model = pickle.load(pickle_file)
    y_predicted = model.predict_proba(y1)[:,1]

    if empl_id is not None:
        data_set = pd.read_csv(settings.MEDIA_ROOT+"WA_empl.csv")
        if 'AgeOfEMP' in data_set:
            data_set.rename(columns = {'AgeOfEMP':'Age'}, inplace = True)
            data_set_col = data_set[["Age", "DailyRate", "DistanceFromHome", "EnvironmentSatisfaction", "JobLevel",
              "JobRole", "MaritalStatus", "MonthlyIncome", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction",
              "TotalWorkingYears",'Attrition']]
        else:
            data_set_col = data_set[["Age", "DailyRate", "DistanceFromHome", "EnvironmentSatisfaction", "JobLevel",
              "JobRole", "MaritalStatus", "MonthlyIncome", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction",
              "TotalWorkingYears",'Attrition']]

        data_set_col['Attrition'].replace({'No': 0, 'Yes': 1},inplace = True)
        data_set_col['OverTime'].replace({'No': 0, 'Yes': 1},inplace = True)
        lb_make = LabelEncoder()
        data_set_col['JobRole']=lb_make.fit_transform(data_set_col['JobRole'].astype(str))
        data_set_col['MaritalStatus']=lb_make.fit_transform(data_set_col['MaritalStatus'].astype(str))

        scaler = MinMaxScaler()
        data_set_col = pd.DataFrame(scaler.fit_transform(data_set_col), columns=data_set_col.columns)
        data_set_col.head()
        x, y = data_set_col.drop(["Attrition"], axis = 1), data_set_col.Attrition
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 888)
        model1 = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2,
                                   oob_score=False, random_state=10)
        RF = model1.fit(x_train, y_train)
        x = x.append(y1)
        x = x.iloc[len(y1.index):]

        predict_fn_rf = lambda x: RF.predict_proba(x).astype(float)
        X = x.values
        explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = x.columns,class_names=['No','Yes'],kernel_width=5)
        choosen_instance = x.loc[[int(empl_id)]].values[0]
        exp = explainer.explain_instance(choosen_instance, predict_fn_rf,num_features=10)
        exp.show_in_notebook(show_all=False)
        exp.save_to_file(settings.BASE_DIR+"/StaffTurnoverApp/static/"+str(empl_id)+".html")
        return empl_id

    csv_empl["predicted_values"] = y_predicted
    csv_empl['predicted_values'] = csv_empl['predicted_values'].multiply(100)

    cost_percentage = 0.214
    cost_reduction = csv_empl.apply(lambda row: (row.MonthlyIncome * 14) * cost_percentage, axis = 1)
    csv_empl["cost_reduction"] = cost_reduction
    csv_empl["cost_reduction"] = csv_empl["cost_reduction"].round(decimals=2)
    # csv_empl.sort_values("cost_reduction", ascending=True)
    df_complete = csv_empl.sort_values(by='cost_reduction', ascending=False)
    return df_complete
