
# coding: utf-8

# In[1]:


# Must be included at the beginning of each new notebook. Remember to change the app name.
import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Iteration4_docs').getOrCreate()

# If you're getting an error with numpy, please type 'sudo pip install numpy --user' into the EC2 console.
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


# In[2]:


fwData = spark.read.csv("Datasets/FoodWasteDataFAO.csv", inferSchema=True, header=True)


# In[3]:


fwData.printSchema()


# In[4]:


fwData.head()


# In[5]:


for item in fwData.head():
    print(item)


# In[6]:


from pyspark.sql.functions import isnan,isnull, when, count, col
fwData.select([count(when(isnull(c), c)).alias(c) for c in fwData.columns]).show()


# In[7]:


fwData.select([count(when(isnan(c), c)).alias(c) for c in fwData.columns]).show()


# In[8]:


selected_fwData = fwData.select(['geographicaream49','country', 'region', 'crop', 'timepointyears', 
                               'loss_per_clean', 'percentage_loss_of_quantity', 'loss_quantity',
                               'loss_qualitiative', 'loss_monetary', 'activity'])


# In[9]:


selected_fwData.printSchema()


# In[10]:


renamed_fwData = selected_fwData.withColumnRenamed("geographicaream49", "geographicarea")
renamed_fwData.printSchema()


# In[11]:


from pyspark.sql.types import IntegerType
from pyspark.sql.functions import *

renamed_fwData.select("crop").show(2)

allColNames = renamed_fwData.columns
for colName in allColNames:
    renamed_fwData = renamed_fwData.withColumn(colName, regexp_replace(colName, "[,]", ""))
    renamed_fwData = renamed_fwData.withColumn(colName, regexp_replace(colName, "[-]", ""))
    renamed_fwData = renamed_fwData.withColumn(colName, regexp_replace(colName, "[/]", ""))
    renamed_fwData = renamed_fwData.withColumn(colName, regexp_replace(colName, '["]', ""))
    renamed_fwData = renamed_fwData.withColumn(colName, regexp_replace(colName, "[(]", ""))
    renamed_fwData = renamed_fwData.withColumn(colName, regexp_replace(colName, "[)]", ""))

renamed_fwData.select("crop").show(2)


# In[12]:


changedType_fwData = renamed_fwData.withColumn("geographicarea", renamed_fwData["geographicarea"]
                                               .cast(IntegerType())).withColumn("timepointyears", renamed_fwData["timepointyears"]
                                               .cast(IntegerType())).withColumn("loss_per_clean", renamed_fwData["loss_per_clean"]
                                               .cast(IntegerType())).withColumn("percentage_loss_of_quantity", renamed_fwData["percentage_loss_of_quantity"]
                                               .cast(IntegerType())).withColumn("loss_quantity", renamed_fwData["loss_quantity"]
                                               .cast(IntegerType()))

changedType_fwData.printSchema()


# In[13]:


changedType_fwData.select([count(when(isnull(c), c)).alias(c) for c in changedType_fwData.columns]).show()


# In[14]:


droppedNull_fwData = changedType_fwData.dropna(how='any', subset=('geographicarea','country','crop','timepointyears'))
droppedNull_fwData.select([count(when(isnull(c), c)).alias(c) for c in droppedNull_fwData.columns]).show()


# In[15]:


percentage_loss_val = droppedNull_fwData.select(mean(droppedNull_fwData.percentage_loss_of_quantity)).collect()
loss_per_clean_val = droppedNull_fwData.select(mean(droppedNull_fwData.loss_per_clean)).collect()
loss_quantity_val = droppedNull_fwData.select(mean(droppedNull_fwData.loss_quantity)).collect()

percentage_loss_mean = percentage_loss_val[0][0]
loss_per_clean_mean = loss_per_clean_val[0][0]
loss_quantity_mean = loss_quantity_val[0][0]

droppedNull_fwData.select('percentage_loss_of_quantity', 'loss_per_clean', 'loss_quantity').show(2)

droppedNull_fwData = droppedNull_fwData.fillna(percentage_loss_mean, subset=['percentage_loss_of_quantity'])
droppedNull_fwData = droppedNull_fwData.fillna(loss_per_clean_mean, subset=['loss_per_clean'])
droppedNull_fwData = droppedNull_fwData.fillna(loss_quantity_mean, subset=['loss_quantity'])

droppedNull_fwData.select('percentage_loss_of_quantity', 'loss_per_clean', 'loss_quantity').show(2)


# In[16]:


final_fwData = droppedNull_fwData.fillna('missing', subset=['region', 'loss_qualitiative', 'loss_monetary', 'activity'])

final_fwData.select([count(when(isnull(c), c)).alias(c) for c in final_fwData.columns]).show()


# In[17]:


final_fwData = final_fwData.withColumn("id", monotonically_increasing_id())
final_fwData.printSchema()


# In[18]:


final_fwData.select("id").show(10)


# In[19]:


final_fwData = final_fwData.select(["id", "geographicarea", "country", 
                                   "region", "crop", "timepointyears", 
                                   "loss_per_clean", "percentage_loss_of_quantity", "loss_quantity",
                                   "loss_qualitiative", "loss_monetary", "activity"])
final_fwData.printSchema()


# In[20]:


print("No of rows before reduction: " , final_fwData.count())

final_fwData_reduced = final_fwData.filter(final_fwData.id < 15000)

print("No of rows after reduction: " ,final_fwData_reduced.count())


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

geographicareaArray = np.array(final_fwData_reduced.select('geographicarea').collect())
plt.hist(geographicareaArray)
plt.title("Distribution of geographic area")
plt.show()

timepointyearsArray = np.array(final_fwData_reduced.select('timepointyears').collect())
plt.hist(timepointyearsArray)
plt.title("Distribution of timepointyears")
plt.show()

percentageLossArray = np.array(final_fwData_reduced.select('loss_quantity').collect())
plt.hist(percentageLossArray)
plt.title("Distribution of percentage loss of quantity")
plt.show()

lossPerCleanArray = np.array(final_fwData_reduced.select('loss_per_clean').collect())
plt.hist(lossPerCleanArray)
plt.title("Distribution of loss per clean")
plt.show()

CountryArray = np.array(final_fwData_reduced.select('country').collect())
plt.hist(CountryArray)
plt.title("Distribution of country")
plt.show()

CropArray = np.array(final_fwData_reduced.select('crop').collect())
plt.hist(CropArray)
plt.title("Distribution of crop")
plt.show()


# In[47]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["timepointyears", "loss_per_clean"],
    outputCol="features")

transformed = assembler.transform(final_fwData_reduced)
final_Data = transformed.select("features", "geographicarea")

featureIndexer =    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(final_Data)

(trainingFwData, testFwData) = final_Data.randomSplit([0.7, 0.3])
print("size of training data: ", trainingFwData.count())
print("size of test data: ", testFwData.count())


# In[48]:


rf = RandomForestRegressor(featuresCol="features", labelCol="geographicarea")

pipeline = Pipeline(stages=[featureIndexer, rf])

model = pipeline.fit(final_Data)

predictions = model.transform(testFwData)

predictions.select("prediction", "geographicarea", "features").show(5)


# In[49]:


evaluator = RegressionEvaluator(
    labelCol="geographicarea", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

totalResults = predictions.select('geographicarea','prediction')

correctResults = totalResults.filter(totalResults['geographicarea'] == totalResults['prediction'])

countTR = totalResults.count()
print("Correct: " + str(countTR))

countTC = correctResults.count()
print("Total Correct: " + str(countTC)) 


# In[50]:


predictionArray = np.array(predictions.select('prediction').collect())
geographicAreaArray = np.array(predictions.select('geographicarea').collect())
plt.hist(geographicAreaArray, label="actual value")
plt.hist(predictionArray, label="predicted value")
plt.title("Predicted values vs Actual values, without year and loss per clean")
plt.legend(loc='upper right')
plt.show()

