from typing import Union

from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pyod.models.knn import KNN
from sklearn.model_selection import train_test_split

app = FastAPI()

neigh = None
clf =None
#d = None

@app.on_event("startup")
def load_train_model():
    df = pd.read_csv("iris_ok.csv")
    global neigh
    neigh = KNeighborsClassifier(n_neighbors= len(np.unique(df['y'])))
    neigh.fit(df[df.columns[:4]].values.tolist(), df['y'])
    global clf
    clf = KNN()
    clf.fit(df[df.columns[:4]].values.tolist(), df['y'])
    print('Training finished')

@app.get("/predict")
def predict(p1: float, p2: float, p3: float, p4: float):
    result = neigh.predict([[p1, p2, p3, p4]])[0]
    return {'result': int(result)}

@app.get("/anomaly")
def anomaly(p1: float, p2: float, p3: float, p4: float):
    result = clf.predict([[p1, p2, p3, p4]])[0]
    return {'result': int(result)}
    

@app.get("/")
def read_root():
    return {"Hello": "World"}

'''
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
    '''

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host=os.environ['HOST'], port=os.environ['PORT'])