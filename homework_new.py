#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import argparse
from pyspark.sql.functions import count, avg, percentile_approx, concat, split, lit
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql import SparkSession
parser = argparse.ArgumentParser()
parser.add_argument("--ngrams", help="some useful description.")
args, unknown = parser.parse_known_args()

def main(input = unknown[0],output = unknown[1]):

    spark = SparkSession.builder          .appName("testing")          .enableHiveSupport()          .getOrCreate()
    df_new = (spark.read.format("csv").options(header="true")
    .load(input))
    
    df = df_new.dropDuplicates(['INCIDENT_NUMBER'])
    
    df=df.select(concat(df.YEAR,df.MONTH)
              .alias("year_month"),"*")
    
    median_monthly_crimes = df     .groupBy(df.DISTRICT, df.year_month)     .agg(count(df.INCIDENT_NUMBER))     .groupBy('DISTRICT')     .agg(percentile_approx('count(INCIDENT_NUMBER)', 0.5).alias("crimes_monthly"))
    
    df = df.withColumn('first_part_name', split(df.OFFENSE_DESCRIPTION, ' - ').getItem(0))
    temp = df.groupBy('DISTRICT', 'first_part_name').agg(count('INCIDENT_NUMBER').alias('count'))

# define window
    w = Window().partitionBy('DISTRICT').orderBy(F.desc('count'))

# create lookup table
    first_highest = temp       .withColumn('rank', F.dense_rank().over(w))       .filter(F.col('rank') == 1)       .select('DISTRICT',temp.first_part_name.alias('first'))

    second_highest = temp       .withColumn('rank', F.dense_rank().over(w))       .filter(F.col('rank') == 2)       .select('DISTRICT',temp.first_part_name.alias('second'))

    third_highest = temp       .withColumn('rank', F.dense_rank().over(w))       .filter(F.col('rank') == 3)       .select('DISTRICT',temp.first_part_name.alias('third'))

    temp_top_3 = first_highest.join(second_highest, on = 'DISTRICT', how = 'left')
    temp_top_3 = temp_top_3.join(third_highest, on = 'DISTRICT', how = 'left')

    temp_top_3 = temp_top_3.select(temp_top_3.DISTRICT,concat('first',lit(', '),'second',lit(', '),'third').alias('frequent_crime_types'))
    
    final = df     .groupby(df.DISTRICT)     .agg(count(df.INCIDENT_NUMBER).alias('crimes_total'), avg(df.Lat).alias('lat'), avg(df.Long).alias('lng'))
    
    final = final.join(median_monthly_crimes, how = 'left', on = 'DISTRICT')
    
    final = final.join(temp_top_3, how = 'left', on = 'DISTRICT')
    
    final.write.parquet(output) 
    
if __name__ == "__main__":
    main()

