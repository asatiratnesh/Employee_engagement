from io import TextIOWrapper
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import squarify
import pandas as pd

#import squarify


def staffTurnoverResult(columns_mapping_dict, emp_data):
    # ml code goed here
    empl_count_per_job = {}
    job_roles = []
    for i in emp_data[columns_mapping_dict['jobRole']]:
        if i in job_roles:
            empl_count_per_job[i] = empl_count_per_job[i]+1
        else:
            job_roles.append(i)
            empl_count_per_job[i] = 1
    treemap_empl_job = plotEmplVsJob(empl_count_per_job);
    return treemap_empl_job


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
