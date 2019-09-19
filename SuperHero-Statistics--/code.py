# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path) 
data['Gender'].replace('-','Agender',inplace=True)
gender_count=data['Gender'].value_counts()
gender_count.plot(kind='bar')
#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
print(alignment)
plt.pie(alignment,autopct='%.1f%%',shadow=True)



# --------------
#Code starts here
sc_df=data[['Strength','Combat']].copy()
sc_covariance = data.Strength.cov(data.Combat) #calclate covariance between two columns
sc_strength=data['Strength'].std() #calculating standard deviationb
sc_combat=data['Combat'].std() #calculating standard deviation
mul=sc_strength*sc_combat # product of calculated above standard deviation
sc_pearson= sc_covariance/mul #calculating pearson coefficient
print('Pearsons Correlation Coefficient:',sc_pearson)
print('============================================================================')
ic_df=data[['Intelligence','Combat']].copy()
ic_covariance=data.Intelligence.cov(data.Combat)   #calclate covariance 
ic_intelligence=data.Intelligence.std() #calculating standard deviation
ic_combat=data['Combat'].std() #calculating standard deviation
ic_mul=ic_intelligence*ic_combat # product of calculated above standard deviation
ic_pearson=ic_covariance/ic_mul #calculating pearson coefficient
print('Pearsons Correlation Coefficient:',ic_pearson)


# --------------
#Code starts here
total_high=data['Total'].quantile(0.99)
super_best=data[data['Total']>total_high]
super_best_names=super_best['Name'].tolist()
print('Best Superhero in Universe:',super_best_names)

fig,(ax_1,ax_2,ax_3)=plt.subplots(1,3,figsize=[40,30])
ax_1.boxplot(super_best['Intelligence'])
ax_1.set(title='Intelligence')
ax_2.boxplot(super_best['Speed'])
ax_2.set(title='Speed')
ax_3.boxplot(super_best['Power'])
ax_3.set(title='Power')
plt.show()



# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3)=plt.subplots(1,3,figsize=[40,30])
ax_1.boxplot(super_best['Intelligence'])
ax_1.set(title='Intelligence')
ax_2.boxplot(super_best['Speed'])
ax_2.set(title='Speed')
ax_3.boxplot(super_best['Power'])
ax_3.set(title='Power')
plt.show()


