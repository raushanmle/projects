file_name = "/home/raushan/projects/popular-datasets/2 Sample - Superstore.xlsx"

from distutils.filelist import findall
import re
from select import select
from pyspark.sql import SparkSession

spark =SparkSession.builder.appName('myapp').getOrCreate()

df = spark.read.option('header', True).csv(path + "SampleSuperstore.csv")

display(df)
display(df.show())

df.display

df.head(10)

df.select('Ship Mode').display()

df.show(n=2)


df.columns

df.printSchema()

display(df)

display(df.limit(5))

df.show()
df.limit(5).show()

df = spark.read.option('header','True').option('delimiter',',').option('inferSchema','True').csv(file_name)

schema = df.schema

df_partial = spark.read.option('header','True').option('delimiter',',').schema(schema).csv(file_name)

from pyspark.sql.types import IntegerType, StringType, StructField,StructType

schema = StructType([
StructField("Year",IntegerType(),True),
StructField("Month",IntegerType(),True),
StructField("DayofMonth",IntegerType(),True),
StructField("IsDepDelayed",StringType(),True)
])

display(df.dtypes)

print(df.schema)

from pyspark.sql.types import DoubleType

display(df['Ship Mode', 'Segment'].show())

display(df[df.columns].show(5))

from IPython.core.display import HTML
display(HTML("<style>pre { white-space: pre !important; }</style>"))

df.limit(6).toPandas()


display(df.groupBy('Segment').avg('Sales').show()

print(df.schema)

from pyspark.sql.types import DoubleType
df1 = df.withColumn("Postal Code", df["Postal Code"].cast(DoubleType()))

df1.dtypes

df.groupBy(['Country','Segment']).agg({'Sales':'sum'}).show()

df.filter(df['Ship Mode'] == 'Standard Class').groupBy('Segment').avg('Sales').show()

display(df
.filter(((df.Origin != 'SAN') & (df.DayOfWeek < 3)) | (df.Origin == 'SFO'))
.groupBy('DayOfWeek')
.avg('arrdelaydouble'))

display(df
.filter(df.Origin != 'SAN')
.filter(df.DayOfWeek < 3)
.groupBy('DayOfWeek')
.avg('arrdelaydouble'))

display(df
.filter(df.Origin == 'SAN')
.groupBy('DayOfWeek')
.avg('arrdelaydouble')
.sort('DayOfWeek'))

display(df
.filter((df.Origin == 'SAN') & (df.Dest == 'SFO'))
.groupBy('DayOfWeek')
.avg('arrdelaydouble'))

display(df
.filter(((df.Origin != 'SAN') & (df.DayOfWeek < 3)) | (df.Origin == 'SFO'))
.groupBy('DayOfWeek')
.avg('arrdelaydouble'))

from pyspark.sql.functions import mean, round
df.filter(df['Ship Mode'] == 'Standard Class').groupBy('Segment').avg('Sales').alias('Avgsales').show()


df.filter(df['Ship Mode'] == 'Standard Class').groupBy('Segment').agg(round(mean('Sales'),2).alias('AvgArrDelay')).show()

display(df
.filter(df.Origin == 'SAN')
.groupBy('DayOfWeek')
.avg('arrdelaydouble')
.sort('DayOfWeek'))

display(df.filter(df.Origin == 'SAN')
.groupBy('DayOfWeek')
.avg('arrdelaydouble')
.orderBy('DayOfWeek')

display(df
.filter(df.Origin == 'SAN')
.groupBy('DayOfWeek')
.agg(round(mean('arrdelaydouble'),2).alias('AvgArrDelay'))
.sort(desc('AvgArrDelay')))

from pyspark.sql.functions import min, max
display(df
.filter(df.Origin == 'SAN')
.groupBy('DayOfWeek')
.agg(min('arrdelaydouble').alias('MinDelay')
, max('arrdelaydouble').alias('MaxDelay')
, (max('arrdelaydouble')-min('arrdelaydouble')).alias('Spread'))
)

display(df
.filter(df.Origin.isin(['SFO','SAN','OAK']))
.filter(df
.DayDate.between(start_date,end_date)
)
.groupBy('Origin','DayDate')
.agg(mean('ArrDelay'))
.orderBy('DayDate'))

airport_list = [row.Origin for row in df.select('Origin').distinct().limit(5).collect()]

from pyspark.sql.functions import when
df = df.withColumn('State',
when(col('Origin') == 'SAN', 'California')
.when(df.Origin == 'LAX', 'California')
.when(df.Origin == 'SAN', 'California')
.when((df.Origin == 'JFK') | (df.Origin == 'LGA') |
(df.Origin == 'BUF'), 'New York')
.otherwise('Other')
)

def bins(flights):
    if flights < 400:
        return 'Small'
    elif flights >= 1000:
        return 'Large'
    else:
        return 'Medium'


from pyspark.sql.functions import count, mean

df_temp = df.groupBy('Origin','Dest').agg(
count('Origin').alias('count'), mean('arrdelaydouble').alias('AvgDelay'))

from pyspark.sql.types import StringType

bins_udf = udf(bins, StringType())
df_temp = df_temp.withColumn("Size", bins_udf("count"))


display(df1.filter(df.Discount.isNull()))

from pyspark.sql.functions import count, when, isnull

display(df1.select([count(when(isnull(c), c)).alias(c) for c in df.columns]))

df1.count(), len(df1.columns)

df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).limit(7).toPandas()

df.select([count(when(isnull('Segment'), 'Segment'))]).limit(7).toPandas()

from pyspark.sql.functions import col
cols = [c for c in df.columns if df.filter(col(c).isNull()).count() > 0 ]


from pyspark.ml.feature import Imputer
df.withColumn('Sales', df.Sales.cast('double'))

def sample (args):
    




import numpy as np
a = np.array([[1,2,3], [4,5,6], [7,8,9]])


a*a
np.mean(a, axis = 0)

x = 'raushan raushan'

import re
re.findall('an', x)