import pandas as pd 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressionModel
spark = SparkSession.builder.appName('BigMartSales').getOrCreate()


model = DecisionTreeRegressionModel.load('model')

def getInterval(pred):
    df_stats = pd.read_csv('constants/stats.csv', index_col='stat')
    Z_SCORE = df_stats.loc['z_score'].values[0]
    STD = df_stats.loc['std'].values[0]
    return (pred - Z_SCORE*STD, pred + Z_SCORE*STD)

def predict_item(params):
   pd.DataFrame(params).T.to_csv('data/tmp.csv', index=False)
   df_to_predict = spark.read.csv('data/tmp.csv', header=True, inferSchema=True)
   df_to_predict = VectorAssembler(inputCols=[str(i) for i in range(9)], outputCol='featuresVector').transform(df_to_predict).select('featuresVector')
   return model.transform(df_to_predict).collect()[0][1]

params = [100, 23, 0, 5, 580, 15, 1, 2, 0]
pred = predict_item(params)
print(f"Interval for this item is {round(pred,3)} ±{round(MAE, 3)}, or in between {getErrorInterval(pred)}")

while True:
    print('Welcome to item sales prediction, please specify work mode:')
    print('1. From file (specific template)')
    print('2. Single item')
    print('>', end='')
    response = input()
    if (response == '1'):
        print('')
    else:
        print()

    print('Restart?')
    input()