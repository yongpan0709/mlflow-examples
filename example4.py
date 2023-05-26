import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("runs:/e084bd9b5e584fc289bf42bd095652ba")
predictions = model.predict(X_test)
print(predictions)