from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from itertools import chain
import datetime
import jdate
import pyspark.sql.functions as f
import numpy
from pyspark.ml.feature import VectorAssembler, StandardScaler, Bucketizer
from pyspark.ml.clustering import KMeans

sc = SparkContext()
sqlc = SQLContext(sc)

data_source_format = 'org.apache.spark.sql.execution.datasources.hbase'

df_demo = sqlc.read\
.options(catalog=catalog_demo)\
.format('org.apache.spark.sql.execution.datasources.hbase')\
.load()

def persian_date(miladi):
    year = int(miladi[:4])
    if miladi[4] == '0':
        month = int(miladi[5])
    else:
        month = int(miladi[4:6])
    if miladi[6] == '0':
        day = int(miladi[7])
    else:
        day = int(miladi[6:])

    jd = jdate.gregorian_to_jd(year,month,day)
    return jdate.jd_to_persian(jd)

def islamic_date(miladi):
    year = int(miladi[:4])
    if miladi[4] == '0':
        month = int(miladi[5])
    else:
        month = int(miladi[4:6])
    if miladi[6] == '0':
        day = int(miladi[7])
    else:
        day = int(miladi[6:])

    jd = jdate.gregorian_to_jd(year,month,day)
    return jdate.jd_to_islamic(jd)


def miladi_date(miladi):
    year = int(miladi[:4])
    if miladi[4] == '0':
        month = int(miladi[5])
    else:
        month = int(miladi[4:6])
    if miladi[6] == '0':
        day = int(miladi[7])
    else:
        day = int(miladi[6:])
    ## shamsi be miladi
    jd = jdate.persian_to_jd(year,month,day)
    return ''.join(str(int(e)) for e in jdate.jd_to_gregorian(jd))

def day_of_week(miladi):
    year = int(miladi[:4])
    if miladi[4] == '0':
        month = int(miladi[5])
    else:
        month = int(miladi[4:6])
    if miladi[6] == '0':
        day = int(miladi[7])
    else:
        day = int(miladi[6:])
    ## shamsi be miladi
    jd = jdate.gregorian_to_jd(year,month,day)
    wday = jdate.jwday(jd)
    return jdate.PERSIAN_WEEKDAYS[int(wday)]

def get_month(converted_date):
    return converted_date[1]

get_month_udf = f.udf(get_month, StringType())

age_udf = f.udf(age, IntegerType())

location_udf = f.udf(location, StringType())

persian_udf = f.udf(persian_date, StringType())

islamic_udf = f.udf(islamic_date, StringType())

miladi_udf = f.udf(miladi_date, StringType())

day_of_week_udf = f.udf(day_of_week, StringType())

df_demo = df_demo.withColumn('islamic', islamic_udf(df_demo.DEMO_CAPTUREDATE))

df_demo = df_demo.withColumn('islamic_month', get_month_udf(df_demo.islamic))

# df_demo = df_demo.withColumn('day_week', day_of_week_udf(df_demo.DEMO_CAPTUREDATE))

df_demo = df_demo.withColumn('age', age_udf(df_demo.birth_Year))

df_demo = df_demo.withColumn('location', location_udf(df_demo.cityTel))

bucketizer = Bucketizer(splits=[0, 10, 20, 30, 40, 60, float('Inf')], inputCol="age", outputCol="age_Category")

df_cat = bucketizer.setHandleInvalid("keep").transform(df_demo)

#df_cat = df_cat.withColumn("age_Category", df_cat["age_Category"].cast("integer")) # converting string type to integer

#df_cat = df_cat.withColumn("age_Category", df_cat["age_Category"].cast("string")) # converting string type to integer

df = df_cat.filter((df_cat.islamic_month == '09'))

dfCount = df.groupBy('ID','DEMO_HOUR').count().select('ID','DEMO_HOUR', f.col('count').alias('count'))

dfMax = dfCount.groupBy('ID').agg(f.last("count").alias("maxCount"),f.last("DEMO_HOUR").alias("interest_HOUR"))

dfMax = dfMax.sort("ID")

dfMax = dfMax.drop('maxCount')

overall_df = df.join(dfMax, ["ID"], how='left')

overall_df.show(100, truncate=False)
# df_cat.groupBy('day_week').count().sort('count').show()
# df_cat.groupBy('age_Category').count().sort('count').show(50)
