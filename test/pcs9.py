from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)
pcs=PCA()
pca_result=pcs.fit_transform(scaled_data)

#

explained_var=pcs.explained_variance_ratio_
print(explained_var)
cumultitve_varinnce=np.cumsum(explained_var)
print(cumultitve_varinnce)

plt.figure(figsize=(8,6))
plt.plot(range(1,len(explained_var)+1),cumultitve_varinnce,marker='o')
plt.title('cumu expained var')
plt.xlabel('princ compo')
plt.ylabel("cumu expained var")
plt.show()

n_components=np.argmax(cumultitve_varinnce>=0.90)+1
print(f"{n_components}")