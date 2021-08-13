
import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import when, count, col


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# spark

findspark.init(r"C:\Spark")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext


spark_df = spark.read.csv(r"churn2.csv", header=True, inferSchema=True)
type(spark_df)

spark_df.head()


# Exploratory Data Analysis

# Number of observations and variables
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# Variable types
spark_df.printSchema()
spark_df.dtypes

# [('RowNumber', 'int'),
#  ('CustomerId', 'int'),
#  ('Surname', 'string'),
#  ('CreditScore', 'int'),
#  ('Geography', 'string'),
#  ('Gender', 'string'),
#  ('Age', 'int'),
#  ('Tenure', 'int'),
#  ('Balance', 'double'),
#  ('NumOfProducts', 'int'),
#  ('HasCrCard', 'int'),
#  ('IsActiveMember', 'int'),
#  ('EstimatedSalary', 'double'),
#  ('Exited', 'int')]

spark_df.select(spark_df.Age).show()
spark_df.take(5)
spark_df.head()
spark_df.show()


spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])


# summary statistics
spark_df.describe().show()

spark_df.describe(["CreditScore", "Tenure"]).show()


# Categorical variable class statistics
spark_df.groupby("Exited").count().show()

# Is there an imbalance?
# 1-2037
# 2-7963

spark_df.groupby("Geography").agg({"Exited": "mean"}).show()
# |Geography|        avg(Exited)|
# +---------+-------------------+
# |  Germany|0.32443204463929853|
# |   France|0.16154766653370561|
# |    Spain| 0.1667339523617279|
# +---------+-------------------+


# unique values
spark_df.select("Geography").distinct().show()
# +---------+
# |  Germany|
# |   France|
# |    Spain|
# +---------+

spark_df.select("account_manager").distinct().show()


# groupby
spark_df.groupby("Exited").count().show()
spark_df.groupby("Exited").agg({"Tenure": "mean"}).show()
spark_df.groupby("Exited").agg({"EstimatedSalary": "mean"}).show()
spark_df.groupby("isactivemember").agg({"Exited": "mean"}).show()


# selecting numerical variables
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()

spark_df.select(num_cols).describe().toPandas().transpose()

# selecting categorical variables
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# Summary statistics of numerical variables according to Churn
for col in num_cols:
    spark_df.groupby("Exited").agg({col: "mean"}).show()


############################
# Feature Interaction
############################
spark_df.columns
spark_df.show(10)

spark_df = spark_df.withColumn('hascr_active', spark_df.hascrcard * spark_df.isactivemember)

spark_df = spark_df.withColumn('cr_salary', spark_df.creditscore / spark_df.estimatedsalary)
spark_df = spark_df.withColumn('cr_tenure', spark_df.creditscore * spark_df.tenure)

spark_df = spark_df.withColumn('bal_slry', spark_df.balance / spark_df.estimatedsalary)


# Bucketization / Bining / Num to Cat
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()

for col in num_cols:
    spark_df.select(col).summary("count", "min", "25%", "50%","75%", "max").show()

bucketizer = Bucketizer(splits=[18, 32, 37, 44, 92], inputCol="age", outputCol="AGE_CAT")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)


spark_df = spark_df.withColumn('AGE_CAT', spark_df.AGE_CAT + 1)
spark_df = spark_df.withColumn("AGE_CAT", spark_df["AGE_CAT"].cast("integer"))

bucketizer = Bucketizer(splits=[0, 3, 5, 7, 10], inputCol="tenure", outputCol="TENURE_CAT")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
spark_df = spark_df.withColumn('TENURE_CAT', spark_df.TENURE_CAT + 1)
spark_df = spark_df.withColumn("TENURE_CAT", spark_df["TENURE_CAT"].cast("integer"))

spark_df.show(20)

spark_df.dtypes

############################
# Label Encoding
############################

indexer = StringIndexer(inputCol="gender", outputCol="gender_lab")
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_lab", temp_sdf["gender_lab"].cast("integer"))

spark_df.columns

indexer = StringIndexer(inputCol="geography", outputCol="geography_lab")
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("geography_lab", temp_sdf["geography_lab"].cast("integer"))


############################
# One Hot Encoding
############################

encoder = OneHotEncoder(inputCols=["AGE_CAT", "TENURE_CAT", "geography_lab"],
                        outputCols=["AGE_CAT_OHE", "TENURE_CAT_OHE", "GEOGRAPHY_LABEL_OHE"])

spark_df = encoder.fit(spark_df).transform(spark_df)

spark_df.show(20)


# Defining TARGET

stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))

############################
# Feature'ların Tanımlanması
############################

# sadece modele girecek değişkenler yer almalıdır
spark_df.columns
cols = ['creditscore', 'age', 'tenure', 'balance','numofproducts', 'hascrcard',
        'isactivemember','cr_tenure',"estimatedsalary",
        'bal_slry','gender_lab',"cr_salary", "hascr_active",
         'AGE_CAT_OHE', 'TENURE_CAT_OHE', 'GEOGRAPHY_LABEL_OHE']

va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)


# Final sdf

final_df = va_df.select("features", "label")
final_df.show(5)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

##################################################
# Modeling
##################################################

############################
# Logistic Regression
############################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()


# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()



evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))



############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()



############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()


gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())



cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()

# 0.8636512618813503

