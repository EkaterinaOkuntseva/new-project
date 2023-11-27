import logging
import os
from datetime import datetime

import dill
import glob
import json
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

path = ('C:/Users/Ekat/PycharmProjects/airflow_hw')

def predict() -> None:
    # Определяем модель
    model_filename = f'{path}/data/models/cars_pipe_202311200927.pkl'       ####(f'{path}/data/models')[0]

    # Загружаем обученную модель
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    # Создаем датафрейм с колонками car_id и результатом предсказания
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])

    # Перебор объектов json в папке data/test
    for file_name in glob.glob(f'{path}/data/test/*.json'):
        with open(file_name) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)       # делаем предсказание
            X = {'car_id': df.id, 'pred': y}    #определяем нужные колонки
            df1 = pd.DataFrame(X)  #определяем датафрейм из выбраных колонок
            df_pred = pd.concat([df_pred, df1], axis=0)  #объединить предсказания в один датафрейм
        print(df_pred)

        # сохранить в формате csv в папку predictions
        df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
