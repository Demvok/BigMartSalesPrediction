import os
import pandas as pd
import pyspark as sp
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressionModel

spark = SparkSession.builder.appName('BigMartSales').getOrCreate()
model = DecisionTreeRegressionModel.load('model')
encoder = pd.read_csv('constants/encoder.csv', index_col=0)

def getInterval(pred): 
    df_stats = pd.read_csv('constants/stats.csv', index_col='stat')
    Z_SCORE = df_stats.loc['z_score'].values[0]
    STD = df_stats.loc['std'].values[0]
    return (pred - Z_SCORE*STD, pred + Z_SCORE*STD)

MAE = 765.4424351438166
def getErrorInterval(prediction):
    return (round(prediction - MAE, 3), round(prediction + MAE, 3))

def predict_item(params):
   pd.DataFrame(params).T.to_csv('data/tmp.csv', index=False)
   df_to_predict = spark.read.csv('data/tmp.csv', header=True, inferSchema=True)
   df_to_predict = VectorAssembler(inputCols=[str(i) for i in range(9)], outputCol='featuresVector').transform(df_to_predict).select('featuresVector')
   return model.transform(df_to_predict).collect()[0][1]

def predict_file(path):
   df_to_predict = spark.read.csv(path, header=True, inferSchema=True).drop('Item_Identifier').drop('Outlet_Identifier')
   res = spark.read.csv(path, header=True, inferSchema=True).drop('Item_Identifier').drop('Outlet_Identifier')
   res = res.withColumn('id', sp.sql.functions.monotonically_increasing_id())

   encoder = pd.read_csv('constants/encoder.csv', index_col=0)
   for encod_col in encoder.T.columns:
      variables = encoder.loc[encod_col].dropna().tolist()
      for i in range(len(variables)):
         df_to_predict = df_to_predict.replace(variables[i], str(i))
      df_to_predict = df_to_predict.withColumn(encod_col, df_to_predict[encod_col].cast('integer'))

   df_to_predict = VectorAssembler(inputCols=["Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_Type", "Item_MRP", "outletAge", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"], outputCol='featuresVector').transform(df_to_predict).select('featuresVector')
   prediction = model.transform(df_to_predict).drop('featuresVector').withColumnRenamed('prediction', 'Predicted_Sales')
   prediction = prediction.withColumn('id', sp.sql.functions.monotonically_increasing_id())
   res.join(prediction, 'id').drop('id').toPandas().to_csv('prediction.csv')

print('*-*-*')
input()
import time
time.sleep(5)
print('\n'*10)

while True:
    print('Welcome to item sales prediction, please specify work mode:')
    print('1. From file (specific template)')
    print('2. Single item')
    response = input('> ')
    if (response == '1'):
        print('Your file must be in correct format (.csv): ["Item weight", "Is item Low Fat", "Item visibility", "Item type", "Item MRP", "Outlet age", "Outlet size", "Outlet location type", "Outlet type"]')
        availible_files = os.listdir('data')
        print(*[f"{i+1}. {availible_files[i]}\n" for i in range(len(availible_files))], end='')
        response = int(input('>')) - 1
        predict_file("data/" + availible_files[response])
    else:
        params = [-1 for _ in range(9)]
        print('Please, fill following fields')
        print('Item weight ~(4-21): ', end='')
        params[0] = float(input())
        print('Is item Low Fat (0/1): ', end='')
        params[1] = int(input())        
        print('Item visibility ~(0-0.5): ', end='')
        params[2] = float(input())        
        print('Item type: ')
        print(encoder.loc['Item_Type'].dropna().to_string())
        params[3] = int(input('> '))
        print('Item MRP ~(30-270): ', end='')
        params[4] = float(input())        
        print('Outlet age ~(0-40): ', end='')
        params[5] = int(input())        
        print('Outlet size: ')
        print(encoder.loc['Outlet_Size'].dropna().to_string())
        params[6] = int(input('> '))        
        print('Outlet location type: ')
        print(encoder.loc['Outlet_Location_Type'].dropna().to_string())
        params[7] = int(input('> '))        
        print('Outlet type: ')
        print(encoder.loc['Outlet_Type'].dropna().to_string())
        params[8] = int(input('> '))
        if -1 in params:
            print('Input error! Try again')
        else:
            pred = predict_item(params)
            print(f"Interval for this item is {round(pred,3)} ±{round(MAE, 3)}, or in between {getErrorInterval(pred)}")

    input()