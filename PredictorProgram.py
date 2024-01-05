import pandas as pd 
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
spark = SparkSession.builder.appName('BigMartSales').getOrCreate()


df_stats = pd.read_csv('data/stats.csv', index_col='stat')
Z_SCORE = df_stats.loc['z_score'].values[0]
STD = df_stats.loc['std'].values[0]

quick_fit = spark.read.csv('constants/quick_fit.csv', header=True, inferSchema=True).drop('_c0')
toVector = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'outletAge', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']
featuresVector = VectorAssembler(inputCols=toVector,outputCol="featuresVector")
quick_fit = featuresVector.transform(quick_fit)
quick_fit = quick_fit.select('featuresVector', 'Item_Outlet_Sales')
model = DecisionTreeRegressor(featuresCol='featuresVector', labelCol='totalSales', maxBins=2048)
model = model.fit(quick_fit)

def getInterval(pred):
    return (pred - Z_SCORE*STD, pred + Z_SCORE*STD)

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