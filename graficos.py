import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
x = data.data
# Calcula a matriz de correlação
print(df.corr())
# Cria um heatmap com todas as features 
sns.heatmap(df.corr())
plt.title("Heatmap de todas as Features")
plt.show()
# Obtém a lista de nomes das features
feature_names = data.feature_names
num_features = x.shape[1]

#Histograma de cada feature individualmente
plt.figure(figsize=(30, 10))
for i in range(num_features):
    plt.subplot(5, 6, i + 1)
    plt.hist(x[:, i], bins=50, color='blue', alpha=0.7)
    plt.title(f'Histograma de {feature_names[i]}')

plt.tight_layout()
plt.show()




