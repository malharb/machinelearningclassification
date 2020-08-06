import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()

df = sns.load_dataset('iris')

#use machine learning for classification. Via logistic regression and KNN
#1) logistic regression

from sklearn.model_selection import train_test_split
x = df.drop('species',axis=1)
y = df['species']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

logistic_model.fit(x_train,y_train)
predictions_lr = logistic_model.predict(x_test)

#2) KNN

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('species',axis=1))
scaled_df = scaler.transform(df.drop('species',axis=1))

from sklearn.neighbors import KNeighborsClassifier
#finding optimum K
error_values = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    
    error_values.append(np.mean(pred_i != y_test))

optimum = pd.DataFrame(error_values,index=range(1,51),columns=['Error']) #=47

# k = 47 is optimum with a mean error of 0!
knn = KNeighborsClassifier(n_neighbors = 47)
knn.fit(x_train,y_train)
predictions_knn = knn.predict(x_test)

from sklearn.metrics import accuracy_score
logistic_accuracy = accuracy_score(y_test,predictions_lr)
knn_accuracy = accuracy_score(y_test,predictions_knn)

print(f"Accuracy using a Logistic Regression algorithm: {logistic_accuracy}")
print(f"Accuracy using a KNN algorithm: {knn_accuracy}")

input('Press ENTER to exit')


