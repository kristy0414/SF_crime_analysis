# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling 

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### In this notebook, you can learn how to use Spark SQL for big data analysis on SF crime data. (https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry). 

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"
# use this file name later

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# from pyspark.sql.functions import to_date, to_timestamp, hour
# df_opt1 = df_opt1.withColumn('Date', to_date(df_opt1.OccurredOn, "MM/dd/yy"))
# df_opt1 = df_opt1.withColumn('Time', to_timestamp(df_opt1.OccurredOn, "MM/dd/yy HH:mm"))
# df_opt1 = df_opt1.withColumn('Hour', hour(df_opt1['Time']))
# df_opt1 = df_opt1.withColumn("DayOfWeek", date_format(df_opt1.Date, "EEEE"))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1 question (OLAP): 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.
# MAGIC 
# MAGIC Below are some example codes to demonstrate the way to use Spark RDD, DF, and SQL to work with big data. You can follow this example to finish other questions. 

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Q1
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
#Spark SQL based
#df_update.createOrReplaceTempView("sf_crime"), this view step is important and need to be done before sql queries
crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(crimeCategory)

# COMMAND ----------

# important hints: 
## first step: spark df or sql to compute the statisitc result 
## second step: export your result to a pandas dataframe. 

spark_df_q1 = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(spark_df_q1)

# crimes_pd_df = crimeCategory.toPandas()

# Spark does not support this function, please refer https://matplotlib.org/ for visuliation. You need to use display to show the figure in the databricks community. 

# display(crimes_pd_df)

# COMMAND ----------

# DBTITLE 1,Visualize your results
import seaborn as sns
fig_dims = (15,6)
fig = plt.subplots(figsize=fig_dims)
spark_df_q1_plot = spark_df_q1.toPandas()
chart=sns.barplot(x = 'category', y = 'count', palette= 'coolwarm',data = spark_df_q1_plot)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')


# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2 question (OLAP)
# MAGIC Counts the number of crimes for different district, and visualize your results

# COMMAND ----------

spark_sql_q2 = spark.sql("SELECT PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY 1 ORDER BY 2 DESC")
display(spark_sql_q2)


# COMMAND ----------

import matplotlib.pyplot as plt
crimes_dis_pd_df = spark_sql_q2.toPandas()
plt.figure()
ax = crimes_dis_pd_df.plot(kind = 'bar',x='PdDistrict',y = 'Count',logy= True,legend = False, align = 'center')
ax.set_ylabel('count',fontsize = 12)
ax.set_xlabel('PdDistrict',fontsize = 12)
plt.xticks(fontsize=8, rotation=30)
plt.title('#2 Number of crimes for different districts')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3 question (OLAP)
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".   
# MAGIC hint 1: SF downtown is defiend  via the range of spatial location. For example, you can use a rectangle to define the SF downtown, or you can define a cicle with center as well. Thus, you need to write your own UDF function to filter data which are located inside certain spatial range. You can follow the example here: https://changhsinlee.com/pyspark-udf/
# MAGIC 
# MAGIC hint 2: SF downtown physical location rectangle a < x < b  and c < y < d. thus, San Francisco Latitude and longitude coordinates are: 37.773972, -122.431297. X and Y represents each. So we assume SF downtown spacial range: X (-122.4213,-122.4313), Y(37.7540,37.7740).
# MAGIC  

# COMMAND ----------

df_opt2 = df_opt1[['IncidntNum', 'Category', 'Descript', 'DayOfWeek', 'Date', 'Time', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y', 'Location']]
display(df_opt2)
df_opt2.createOrReplaceTempView("sf_crime")

# COMMAND ----------

from pyspark.sql.functions import hour, date_format, to_date, month, year
# add new columns to convert Date to date format
df_new = df_opt2.withColumn("IncidentDate",to_date(df_opt2.Date, "MM/dd/yyyy")) 
# extract month and year from incident date
df_new = df_new.withColumn('Month',month(df_new['IncidentDate']))
df_new = df_new.withColumn('Year', year(df_new['IncidentDate']))
display(df_new.take(5))
df_new.createOrReplaceTempView("sf_crime1")

# COMMAND ----------

# sql way
spark_sql_q3 = spark.sql("SELECT IncidentDate, DayOfWeek, COUNT(*) AS Count FROM sf_crime1 WHERE DayOfWeek = 'Sunday' \
                          AND X > -122.4313 AND X < -122.4213 AND Y > 37.7540 AND Y < 37.7740 \
                          GROUP BY IncidentDate, DayOfWeek ORDER BY IncidentDate")

# COMMAND ----------

display(spark_sql_q3)

# COMMAND ----------

# MAGIC %sql select month(IncidentDate), count(*) AS Count from sf_crime1 WHERE DayOfWeek = 'Sunday' 
# MAGIC                           AND X > -122.4313 AND X < -122.4213 AND Y > 37.7540 AND Y < 37.7740 
# MAGIC                           GROUP BY month(IncidentDate) ORDER BY  month(IncidentDate),Count desc

# COMMAND ----------

# MAGIC %md
# MAGIC late June: 54+33+31+28= 146  July 4th is Independence Day, people tend to have a vocation around this holiday and trabel to SF city.
# MAGIC 
# MAGIC December10/11 and new year: 35+28+31= 94 Some companies may approve employees's paid time off before Christmas holiday, and the whole December is holiday season. travel to SF is a choice.
# MAGIC 
# MAGIC the whole July: 29+26+26= 81 July 4th is Independence Day, it is time for travel.
# MAGIC 
# MAGIC the whole September: 29+26+26= 81 the first Monday of September is also another holiday called Labor Day.
# MAGIC 
# MAGIC late October: 33+31= 64 the second Monday of October is Columbus Day that is another holiday.
# MAGIC 
# MAGIC why travel cause crime? especially for June, October and Janaury? Cause when you are not at home, it gives crimer a chance to approach you and especially when you have a tight vocation schedule you will lose your awareness to take care all of the stuff. it also reflects that people like to travel from the whole second half of the year.

# COMMAND ----------

# MAGIC 
# MAGIC %sql select year(IncidentDate), month(IncidentDate), count(*) AS Count from sf_crime1 WHERE DayOfWeek = 'Sunday' 
# MAGIC                           AND X > -122.4313 AND X < -122.4213 AND Y > 37.7540 AND Y < 37.7740 
# MAGIC                           GROUP BY year(IncidentDate), month(IncidentDate) ORDER BY year(IncidentDate), month(IncidentDate),Count desc

# COMMAND ----------

# MAGIC %md 
# MAGIC 2012, 2013, 2016, 2017 is crazy. what's happened? Oh, it is actually the election year. I still remember when I first land to US and I have no idea about America election. But I do remember lots of students sit in the college plaza to watch for the vote activity. And most of them felt sad when it turns out that the result did not meet their expecatation. 
# MAGIC I may think it is reasonable to make a hypothesis that a new presidential election is prone to social unrest. Also, see what happened this year, 2020 is also a election year. Do you feel the whole socitey is safe enough? Not, right.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4 question (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

years = [2015, 2016, 2017, 2018]
df_years = df_new[df_new.Year.isin(years)]
display(df_years.take(10))

# COMMAND ----------

spark_df_q4 = df_years.groupby(['Year', 'Month']).count().orderBy('Year','Month')
display(spark_df_q4)

# COMMAND ----------

df_years.createOrReplaceTempView("sf_crime2")
fig_dims = (20,6)

# COMMAND ----------

# MAGIC %sql select distinct(category) as type, count(*) as Count, year from sf_crime2 where Year in (2015, 2016, 2017, 2018) group by 1,3 order by 2 desc

# COMMAND ----------

# MAGIC %sql select count(*) as Count, year, month from sf_crime2 where Year in (2015, 2016, 2017, 2018) and category='LARCENY/THEFT' group by 2,3 order by 2,3

# COMMAND ----------

# MAGIC %md
# MAGIC the business impact is the theft contributes to the most crime portion. And the 47th Act signed by the governor in the California take on effect at Nov 2014. After 2015 winter, the crime number is boosting under theft category until Jan 2018.The reason for the decline in crime rate since 2018 may be that the San Francisco Police Department has increased uniformed police patrols.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

from pyspark.sql.functions import to_timestamp
# add new columns to convert Time to hour format
df_new1 = df_new.withColumn('IncidentTime', to_timestamp(df_new['Time'],'HH:mm')) 
# extract hour from incident time
df_new2 = df_new1.withColumn('Hour',hour(df_new1['IncidentTime']))
display(df_new2.take(5))

# COMMAND ----------

dates = ['12/15/2015','12/15/2016','12/15/2017']
df_days = df_new2[df_new2.Date.isin(dates)]
spark_df_q5_1 = df_days.groupby('Hour','Date').count().orderBy('Date','Hour')
display(spark_df_q5_1)

# COMMAND ----------

# MAGIC %md
# MAGIC from the plot we can see that:
# MAGIC for 2015, the peak time of crime is 12,14,16,19. it just from noon to the sunset.
# MAGIC for 2016, the peak time of crime is 12,18,19. it concentrate on 18, the dinner time.
# MAGIC for 2017, the peak time of crime is 0, 8,10,15,16,17,18,19,22,23. the data is more even and the interesting trend is that it seems the crime has more records during the midnight, from 22 to 24. I think at that time there is less police power and the crimers are also easy to steal stuff from the dark enviroment.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger district  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

#sql way
spark_sql_q6_s1 = spark.sql( """
                             SELECT PdDistrict, COUNT(*) as Count
                             FROM sf_crime
                             GROUP BY 1
                             ORDER BY 2 DESC
                             LIMIT 3 
                             """ )
display(spark_sql_q6_s1)


# COMMAND ----------

df_new2.createOrReplaceTempView("sf_crime2")
display(df_new2.take(5))

# COMMAND ----------

# MAGIC %sql select category, hour, count(*) from sf_crime2 where PdDistrict in ('SOUTHERN','MISSION','NORTHERN') group by category, hour 
# MAGIC order by category, hour

# COMMAND ----------

# MAGIC %md
# MAGIC the lunch time and dinner time, espcially the dinner time will have more crime cases. And the most of them are theft and assault. Cause the California law won't arrest the crimer who steal something valued under 900 dollars. people will get off from work and go to grocery store, eat dinner outset or hang out with friends. Southern Area probably has less police power and where people live. The route from office to home can describle like from Mission Area to Southern Area. People need to be careful when they go out of the office, take public traffic, and walk to home with awareness. don't look at cellphone, instead should look around the surroundings.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7 question (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

# MAGIC %md
# MAGIC Below is the resolution count for each category.

# COMMAND ----------

# MAGIC %sql select category, count(*) from sf_crime2 
# MAGIC group by category order by count(*) desc limit 10

# COMMAND ----------

# MAGIC %sql select distinct(resolution) as resolve from sf_crime2

# COMMAND ----------

# MAGIC %md
# MAGIC Here, 'None' takes the most portion of the data. I think they are the cases that are not resolved. it is still open in investigation. So we exclude it. usually, to analyze a problem, we need to know 80/20 rule. focus on the 80% of the data and ignore the cases with less data. So I will pick up the top 10 categories and analyze the resolution percentage. The rest of the category will discover in the future if i have more time.

# COMMAND ----------

# MAGIC %sql select distinct Category, Resolution, count(*) over (PARTITION BY Category, Resolution) as sum_div from sf_crime2
# MAGIC where Resolution != 'NONE'
# MAGIC order by Category, sum_div desc

# COMMAND ----------

# MAGIC %sql with cte_1 as 
# MAGIC (select distinct Category, Resolution, count(*) over (PARTITION BY Category, Resolution) as sum_div, 
# MAGIC count(*) over (PARTITION BY Category) as cat_div
# MAGIC from sf_crime2
# MAGIC where Resolution != 'NONE'
# MAGIC order by Category, sum_div desc)
# MAGIC select distinct Category, Resolution, round(sum_div/cat_div,4) from cte_1
# MAGIC where Category in (select category from (select category, count(*) from sf_crime2 where Resolution != 'NONE'
# MAGIC group by category order by count(*) desc limit 10) as a)

# COMMAND ----------

# MAGIC %md
# MAGIC the most cases's resolution is arrest, booked. it means it is easy to make a judgement. but missing person and non-criminal has a variance.
# MAGIC most missing person's result is located. is it means dead? how to prevent it and how to rescue before we found a dead body? 
# MAGIC also, for non-criminal, the most cases are psychopathic cases. Do we need to pay attention to people who has mental health problems? Are they tend to attack people? where do they live and how to make residential areas more safe? 
# MAGIC Besides, we may also think about to adjust the police power to the poor area and protect the people there.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC Use four sentences to summary your work. Like what you have done, how to do it, what the techinical steps, what is your business impact. 
# MAGIC More details are appreciated. You can think about this a report for your manager. Then, you need to use this experience to prove that you have strong background on big  data analysis.  
# MAGIC Point 1:  what is your story ? and why you do this work ?   
# MAGIC Point 2:  how can you do it ?  keywords: Spark, Spark SQL, Dataframe, Data clean, Data visulization, Data size, clustering, OLAP,   
# MAGIC Point 3:  what do you learn from the data ?  keywords: crime, trend, advising, conclusion, runtime 

# COMMAND ----------

# MAGIC %md
# MAGIC I generated reports from different topics and perspectives. such as time, district, year, resolution type, crime type and so on. I cleaned the data, process data to integret the format, and run sql queries to check the data pattern or trends, then visulize the results.
# MAGIC 
# MAGIC I found several business impact that we may need to have notice:
# MAGIC 
# MAGIC 1 election year will need more polce power to the unrest society.
# MAGIC 
# MAGIC 2 lunch time, dinner time is the best time for crimer to take action. need police power patrol around the restaurant plazas.
# MAGIC 
# MAGIC 3 travel time, holiday season is also a peak time for crimes. 
# MAGIC 
# MAGIC the insight is when you want to have a rest and relax, the crime won't let you to take a rest. Just be careful.
