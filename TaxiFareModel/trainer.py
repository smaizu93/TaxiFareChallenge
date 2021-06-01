import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data


class Trainer:
    def __init__(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    def set_pipeline(self):
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        #create time pipeline
        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        #create preproc pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")


        #final pipe
        self.pipe = Pipeline([
                        ('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())
                        ])


    def run(self):
        self.set_pipeline()
        self.pipe.fit(self.X_train,self.y_train)

    def evaluate(self,X_test,y_test):
        yhat=self.pipe.predict(X_test)
        return compute_rmse(yhat,y_test)

if __name__=="__main__":
    N = 10_000
    df = get_data(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    print(trainer.evaluate(X_test, y_test))
