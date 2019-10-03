# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  
data=pd.read_csv(path)
data_sample=data.sample(n=sample_size,random_state=0)
sample_mean=data_sample.installment.mean()
sample_std=data_sample.installment.std()
margin_of_error=z_critical*sample_std/np.sqrt(sample_size)
confidence_interval=[]
confidence_interval.append(sample_mean-margin_of_error)
confidence_interval.append(sample_mean+margin_of_error)
true_mean=data.installment.mean()
if true_mean>confidence_interval[0] and true_mean<confidence_interval[1]:
    print('true',true_mean)
else:
    print('false')



# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])
fig, axes = plt.subplots(3,1, figsize=(20,10))
for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        m.append(data['installment'].sample(n=sample_size[i]).mean())
    mean_series=pd.Series(m)    
print(mean_series)


# --------------
from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate']=data['int.rate'].str.replace('%','').astype(float)/100
print(data['int.rate'].head())
z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
print(z_statistic, p_value)
if p_value<0.05:
    print('reject')
else:
    print('accept')


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic, p_value=ztest(data[data['paid.back.loan']=='No']['installment'],data[data['paid.back.loan']=='Yes']['installment'])
print(z_statistic,p_value)
if p_value<0.05:
    print('reject')
else:
    print('accept')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
#print(yes)
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed=pd.concat([yes.transpose(),no.transpose()],keys=['Yes','No'],axis=1)
print(observed)
chi2, p, dof, ex = stats.chi2_contingency(observed)
if chi2>critical_value:
    print('reject')
else:
    print('accept')



