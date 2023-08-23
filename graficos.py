import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Calcula a matriz de correlação
print(df.corr())
# Cria um heatmap com todas as features 
sns.heatmap(df.corr())
plt.title("Heatmap de todas as Features")
plt.show()
