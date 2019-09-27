# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(path)
p_a = len(df[df['fico']>700])/len(df)

p_b = len(df[df['purpose']=='debt_consolidation'])/len(df)

df1 = df[df['purpose']=='debt_consolidation']

p_a_b = (len(df[(df['fico']==700) & (df['purpose']=='debt_consolidation')])/len(df))/p_a

p_b_a = (len(df[(df['fico']==700) & (df['purpose']=='debt_consolidation')])/len(df))/p_b


result = p_b_a==p_b
print(result)



# --------------
# code starts here
prob_lp = df[df['paid.back.loan']=='Yes'].shape[0]/df.shape[0]

prob_cs = df[df['credit.policy']=='Yes'].shape[0]/df.shape[0]

new_df = df[df['paid.back.loan']=='Yes']

prob_pd_cs_df = df[(df['paid.back.loan'] == 'Yes') & (df['credit.policy'] == 'Yes')]

p_num = prob_pd_cs_df.shape[0]/df.shape[0]

prob_pd_cs = p_num/prob_lp

bayes = prob_pd_cs*prob_lp/prob_cs

print(bayes)





# code ends here


# --------------
# code starts here
vc = df['purpose'].value_counts()
plt.bar(x=vc.index,height=vc.values,align='center')
plt.show()

df1 = df[df['paid.back.loan']=='No']
vc_df1 = df1['purpose'].value_counts()
plt.bar(x=vc_df1.index,height=vc_df1.values,align='center')
plt.show()


# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
plt.hist(x=df['installment'])
plt.show()
plt.hist(x=df['log.annual.inc'])
plt.show()




# code ends here


