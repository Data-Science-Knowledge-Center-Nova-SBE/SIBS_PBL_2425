# Fraud Business Impact Forecasting: NOVA x SIBS

# Initiating the Spark Session

import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col, avg, when

python_packages_path =  "/home/cdsw/ml_new.tar.gz#environment"
executor_memory="10g"
driver_memory="8g"
executor_cores="3"
maxExecutors=15
memory_overhead= "2g"
n_partitions=200
driver_memory_overhead = 8000
session_name = "trying"

spark = SparkSession\
        .builder\
        .config("spark.yarn.dist.archives", python_packages_path)\
        .config("spark.executor.memory", executor_memory)\
        .config("spark.driver.memory", driver_memory)\
        .config("spark.executor.cores", executor_cores)\
        .config("spark.yarn.executor.memoryOverhead", memory_overhead)\
        .config("spark.dynamicAllocation.enabled", True)\
        .config("spark.dynamicAllocation.maxExecutors", maxExecutors)\
        .config("spark.sql.shuffle.partitions", n_partitions)\
        .config("spark.sql.adaptive.enabled", True)\
        .config("spark.yarn.driver.memoryOverhead",driver_memory_overhead )\
        .config("spark.yarn.queue", "root.nova_sbe")\
        .appName(session_name)\
        .getOrCreate()

# 1. Importing Dataframes

Training dataset and out-of-sample dataset (card status is a complementary table to both datasets).


training = spark.table("nova_sbe.raw_202306_202412")
card_status = spark.table("nova_sbe.card_status_202306_202412")
oos = spark.table("nova_sbe.sample_202412") 
card_status_outsample=spark.table("nova_sbe.card_status_202501_202504")
#Making Sure the Out-Of-Sample has the same columns as the training Dataframe
oos = oos[training.columns]
#Concatinating Card Status Dfs
card_status_full = card_status.union(card_status_outsample)


# 2. Feature Creation
## 2.1 Creating Target Variable

- The Target Variable for this project is `fraud_amount_accpeted` which as the name reveals is the the amount of transactions which were initially  accepted but turned out to be fraudulent 

- To create the Target, we need to define what fraud is, we are doing this with by using the `fraud_sp` and therefore creating the `fraud_label`
- The overall goal with designing the features is to create an exhaustive list, which explains the Target Feature
- The Features are later selected using Mutual Information Analysis Regression

# in the full dataset
df = df.withColumn(
    "fraud_label",
    F.when(
        F.col('fraud_sp') > 0,
        1 # When fraud_sp is positive, transaction is confirmed fraud
    ).when(
        F.col('fraud_sp') <= 0,
        -1 # When fraud_sp is 0 or negative (and not null), transaction is confirmed genuine
    ).otherwise(0) 
)

# in the training dataset to avoid data leakage in the creation of certain feautres 
training = training.withColumn(
    "fraud_label",
    F.when(
        F.col('fraud_sp') > 0,
        1 # When fraud_sp is positive, transaction is confirmed fraud
    ).when(
        F.col('fraud_sp') <= 0,
        -1 # When fraud_sp is 0 or negative (and not null), transaction is confirmed genuine
    ).otherwise(0) 
)
df = df.withColumn(
    "fraud_amount",
    when(col("fraud_label") == 1, col("OPERATION_AMOUNT")).otherwise(0)
)
df=df.withColumn(
    "TRANSACTION_ACCEPTED",
    F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0))
df = df.withColumn(
    "fraud_amount_accepted",
    when(col("TRANSACTION_ACCEPTED") == 1, col("fraud_amount")).otherwise(0)
)

# 2.2 Creating Time Features

- Creating: 
    - `TRANSACTION_MONTH`: Numerical value of the Month of the Transaction (e.g. 11 for November)
    - `HOUR_OF_DAY`: Numerical value for the Hour of the Day (e.g. 14 for 2p.m.)
    - `DAY_OF_WEEK`: Numerical value for the Day of the Week (e.g. 1 for Monday) 
    - `DAY_OF_MONTH`: Numerical value for the Day of the Month (e.g. 31)
    - `TRANSACTION_DATE`: Daily aggregation of the Timestamp
- `TRANSACTION_DATE` will be used for the Daily Aggregation

### 2.2.1 Time Features
df = df.withColumn(
    "TRANSACTION_MONTH",
    F.month(F.col("TRANSACTION_DATETIME"))
).withColumn(
    "HOUR_OF_DAY",
    F.hour(F.col("TRANSACTION_DATETIME"))
).withColumn(
    "DAY_OF_WEEK",
    F.dayofweek(F.col("TRANSACTION_DATETIME"))
).withColumn(
    "DAY_OF_MONTH",
    F.dayofmonth(F.col("TRANSACTION_DATETIME"))
).withColumn(
    "TRANSACTION_DATE",
    F.to_date("TRANSACTION_DATETIME")   
)
### 2.2.2 Sin/Cosine Day of Month Transformation 
from pyspark.sql.functions import col, sin, cos, lit

max_day = 31
pi = 3.141592653589793

df = df.withColumn(
    "dom_angle", 2 * lit(pi) * col("DAY_OF_MONTH") / lit(max_day)
).withColumn(
    "dom_sin", sin(col("dom_angle"))
).withColumn(
    "dom_cos", cos(col("dom_angle"))
).drop("dom_angle")

### 2.2.3 Sin/Cosine Month Transformation  

max_month = 12

df = df.withColumn(
    "month_angle", 2 * lit(pi) * col("TRANSACTION_MONTH") / lit(max_month)
).withColumn(
    "month_sin", sin(col("month_angle"))
).withColumn(
    "month_cos", cos(col("month_angle"))
).drop("month_angle")

### 2.2.4 Is Weekend
df = df.withColumn("is_weekend", F.when(F.dayofweek("TRANSACTION_DATE").isin(1,7), 1).otherwise(0))

## 2.3 Creating Lags & Rolling Statistics

- The goal here is to create the rolling statistics and lags for the accepted fraud amount, where the fraud classification datettime is smaller than the timestamp on that given day

- The rolling statistics include:`rolling_min`, `rolling_max`,`rolling_mean`,`rolling_sum`,`rolling_stddev` for the window of 7 and 14 days

- The lags consist of 1,7,14 days

### 2.3.1 Creating 7-Day Rolling Statistics
# 1. Extract date columns
df = df.withColumn("day", F.to_date("TRANSACTION_DATETIME")) \
       .withColumn("classification_day", F.to_date("fraud_classification_datetime"))

# 2. Select relevant columns for rolling calculations
df_valid = df.select("day", "classification_day", "fraud_amount_accepted")

# 3. Create a DataFrame with unique days renamed as target_day
target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")

# 4. Cross join to combine transactions with all possible target_days
joined = df_valid.crossJoin(target_days) \
    .filter(F.col("day") < F.col("target_day")) \
    .filter(F.col("classification_day") <= F.col("target_day"))

# 5. Aggregate accepted fraud amount per (target_day, day)
daily_agg_per_target = joined.groupBy("target_day", "day").agg(
    F.sum("fraud_amount_accepted").alias("daily_fraud_sum")
)

# 6. Compute days difference and filter to 7-day rolling window
rolling_base = daily_agg_per_target \
    .withColumn("days_diff", F.datediff(F.col("target_day"), F.col("day"))) \
    .filter(F.col("days_diff").between(1, 7))

# 7. Aggregate rolling statistics per target_day
rolling_stats_7 = rolling_base.groupBy("target_day").agg(
    F.mean("daily_fraud_sum").alias("rolling_mean_7"),
    F.max("daily_fraud_sum").alias("rolling_max_7"),
    F.min("daily_fraud_sum").alias("rolling_min_7"),
    F.sum("daily_fraud_sum").alias("rolling_sum_7"),
    F.stddev("daily_fraud_sum").alias("rolling_std_7")
).orderBy("target_day")

### 2.3.2 Creating 14-Day Rolling Statistics
# 8. Rolling Window for 14 days
rolling_base_14= daily_agg_per_target \
    .withColumn("days_diff", F.datediff(F.col("target_day"), F.col("day"))) \
    .filter(F.col("days_diff").between(1, 14))

#9. Aggregate rolling statistics per target_day
rolling_stats_14 = rolling_base.groupBy("target_day").agg(
    F.mean("daily_fraud_sum").alias("rolling_mean_14"),
    F.max("daily_fraud_sum").alias("rolling_max_14"),
    F.min("daily_fraud_sum").alias("rolling_min_14"),
    F.sum("daily_fraud_sum").alias("rolling_sum_14"),
    F.stddev("daily_fraud_sum").alias("rolling_std_14")
).orderBy("target_day")

### 2.3.3 Creating 1-Day Fraud Amount Lagged Features
from pyspark.sql.functions import to_date, col, date_sub, sum as Fsum

# Ensure date columns
df = df.withColumn("day", to_date("TRANSACTION_DATETIME")) \
       .withColumn("classification_day", to_date("fraud_classification_datetime"))

# Get target days
target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")

# Join to original df to compute lag_1
df_lag1 = df.crossJoin(target_days) \
    .filter(col("day") == date_sub(col("target_day"), 1)) \
    .filter(col("classification_day") <= col("target_day"))

# Aggregate to compute lag_1 per target_day
lag1_df = df_lag1.groupBy("target_day").agg(
    Fsum("fraud_amount_accepted").alias("lag_1")
)

### 2.3.4 Creating 7-Day Fraud Amount Lagged Features
from pyspark.sql.functions import to_date, col, date_sub, sum as Fsum

# Ensure date columns
df = df.withColumn("day", to_date("TRANSACTION_DATETIME")) \
       .withColumn("classification_day", to_date("fraud_classification_datetime"))

# Get target days
target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")

# Join to original df to compute lag_1
df_lag7 = df.crossJoin(target_days) \
    .filter(col("day") == date_sub(col("target_day"), 7)) \
    .filter(col("classification_day") <= col("target_day"))

# Aggregate to compute lag_1 per target_day
lag7_df = df_lag7.groupBy("target_day").agg(
    Fsum("fraud_amount_accepted").alias("lag_7")
)

### 2.3.5 Creating 14-Day Fraud Amount Lagged Features
from pyspark.sql.functions import to_date, col, date_sub, sum as Fsum
# Ensure date columns
df = df.withColumn("day", to_date("TRANSACTION_DATETIME")) \
       .withColumn("classification_day", to_date("fraud_classification_datetime"))
# Get target days
target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")
# Join14to original df to compute lag_1
df_lag14 = df.crossJoin(target_days) \
    .filter(col("day") == date_sub(col("target_day"), 14)) \
    .filter(col("classification_day") <= col("target_day"))
# Aggregate to compute lag_1 per target_day
lag14_df = df_lag14.groupBy("target_day").agg(
    Fsum("fraud_amount_accepted").alias("lag_14")
)

## 2.4 Merchant Features

Using the `MCC` and `MERCHANT NAME` column to create features:
- `is_new_merchant`: If a merchant was first seen within the period of training and test data
- `small_merchant`, `big_merchant`: Whether a merchant is "Big" or "Small" Ranked on the Number of Total Transactions
- `fraud_risk_rank`: Risk-Rank for Merchant MCC (the higher the rank the riskier)
- `penalized_operation_amount`: Operation Amount, multiplied by the fraud risk rank of Merchant (e.g. the higher the fraud amount the higher the penalty)
- `is_high_risk_merchant`: Binary feature if a Merchant is Risky, based on `mcc_fraud_rate`, and if it exceeds a certain threshold
- `low_card_merchant_counts`: Binary feature indicating if a Merchant deals with less than 40 distinct cards daily or not

### 2.4.1 Frequency of Merchants
window_spec = Window.partitionBy("MERCHANT_NAME").orderBy("TRANSACTION_DATE")

df = df.withColumn("first_seen_date", F.min("TRANSACTION_DATE").over(window_spec))
df = df.withColumn("is_new_merchant", (F.col("TRANSACTION_DATE") == F.col("first_seen_date")).cast("int"))

### 2.4.2 Risk-Assesment of Merchants
merchant_freq_df = df.groupBy("MERCHANT_NAME").agg(
    F.count("*").alias("tx_count")
)

window_spec = Window.orderBy("tx_count")
merchant_ranked_df = merchant_freq_df.withColumn(
    "percent_rank", F.percent_rank().over(window_spec)
)

merchant_labeled_df = merchant_ranked_df.withColumn(
    "small_merchant", F.when(F.col("percent_rank") <= 0.05, 1).otherwise(0)
).withColumn(
    "big_merchant", F.when(F.col("percent_rank") > 0.05, 1).otherwise(0)
).select("MERCHANT_NAME", "small_merchant", "big_merchant")

df = df.join(merchant_labeled_df, on="MERCHANT_NAME", how="left")

mcc_fraud_pct = training.groupBy("MCC").agg(
    F.round(100 * F.count(F.when(F.col('fraud_label') == 1, 1)) / F.count('*'), 4).alias('mcc_fraud_rate')
)   

window_spec = Window.orderBy(F.col("mcc_fraud_rate").asc())

ranked_mcc = mcc_fraud_pct.withColumn(
    "fraud_rank", F.dense_rank().over(window_spec)
)

ranked_mcc_df_filtered = ranked_mcc.select("MCC", "fraud_rank")

df = df.join(
    ranked_mcc_df_filtered,
    on="MCC",
    how="left"
)

df = df.withColumnRenamed("fraud_rank", "fraud_risk_rank" )

df = df.withColumn("penalized_operation_amount", F.col("OPERATION_AMOUNT") * F.col("fraud_risk_rank"))

fraud_threshold = mcc_fraud_pct.approxQuantile("mcc_fraud_rate", [0.95], 0.01)[0]

high_risk_mcc = mcc_fraud_pct.withColumn(
    "is_high_risk_merchant", F.when(F.col("mcc_fraud_rate") >= fraud_threshold, 1).otherwise(0)
).select("MCC", "is_high_risk_merchant")

df = df.join(high_risk_mcc, on="MCC", how="left").fillna({"is_high_risk_merchant": 0})

### 2.4.3 Selecting Merchants Based on EDA
df = df.withColumn("MERCHANT_NAME_LOWER", F.lower(F.col("MERCHANT_NAME")))


cards_per_merchant_day = df.groupBy("TRANSACTION_DATE", "MERCHANT_NAME").agg(
    F.countDistinct("PAN_ENCRYPTED").alias("distinct_cards")
)

# Flag merchants with < 40 cards
low_card_merchants = cards_per_merchant_day.filter(F.col("distinct_cards") < 40) \
    .groupBy("TRANSACTION_DATE") \
    .agg(F.count("MERCHANT_NAME").alias("low_card_merchants_count"))

### 2.4.4 "IS NEW" Merchant
window_spec = Window.partitionBy("MERCHANT_NAME").orderBy("TRANSACTION_DATE")
df = df.withColumn("first_seen_date", F.min("TRANSACTION_DATE").over(window_spec))
df = df.withColumn("is_new_merchant", (F.col("TRANSACTION_DATE") == F.col("first_seen_date")).cast("int"))

### 2.4.5 Amount of Lower Cards
df = df.withColumn("MERCHANT_NAME_LOWER", F.lower(F.col("MERCHANT_NAME")))


cards_per_merchant_day = df.groupBy("TRANSACTION_DATE", "MERCHANT_NAME").agg(
    F.countDistinct("PAN_ENCRYPTED").alias("distinct_cards")
)

# Flag merchants with < 40 cards
low_card_merchants = cards_per_merchant_day.filter(F.col("distinct_cards") < 40) \
    .groupBy("TRANSACTION_DATE") \
    .agg(F.count("MERCHANT_NAME").alias("low_card_merchants_count"))

## 2.5 Card Features

- Here we are trying to mimic behaviour of fraudsters by identifying if:
   - 1.) `number_tx_exceeded`: Customer has to enter the pin, because the amount of transactions exceeeded
   - 2.) `cash_withdrawal_exceeded`: If the allowed amount of cash, from a Card was exceeded
   - 3.) `card_number_invalid`: If the card number entered was invalid
   - 4.) `invalid_pin`: If the Pin was invalid
   - 5.) `invalid_cvv`: If the cvv was invalid
- Furthermore we are aggregating the number of cancelled cards with `card_cancellations` and aggregated it on a daily basis using `daily_cancellations`

df = df.withColumn("number_tx_exceeded", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 65,1).otherwise(0))
df = df.withColumn("cash_withdrawal_exceeded", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 61,1).otherwise(0))
df = df.withColumn("card_number_invalid", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 14,1).otherwise(0))
df = df.withColumn("invalid_pin", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 55,1).otherwise(0))
df = df.withColumn("invalid_cvv", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 82,1).otherwise(0))

from pyspark.sql.functions import col, to_date

card_status_full = card_status_full.withColumn("TRANSACTION_DATE", to_date(col("TRANSACTION_DATETIME")))
card_cancellations = card_status_full.filter(col("NEW_CARD_STATUS").isin("06", "09"))
daily_cancellations = card_cancellations.groupBy("TRANSACTION_DATE").count().withColumnRenamed("count", "DAILY_CANCELLED_CARD_COUNT")

daily_cancellations = daily_cancellations.withColumnRenamed("DAILY_CANCELLED_CARD_COUNT", "DAILY_CANCELLED_CARD_COUNT_cancellations")
df = df.join(daily_cancellations, on="TRANSACTION_DATE", how="left")
df = df.withColumn("is_remote", F.col("FRAUD_ACCEPTOR_CDE").startswith("REMOTE").cast("int"))

## 2.6 Daily Aggregation

- The following code aggregates the created features on a daily basis `TRANSACTION_DATE`
daily_data = df.groupBy("TRANSACTION_DATE", "is_weekend").agg(
    F.sum(F.when(df["small_merchant"] == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("small_merchant_amount"),
    F.sum(F.when(df["big_merchant"] == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("big_merchant_amount"),
    F.sum("penalized_operation_amount").alias("penalized_total_operation_amount"),
    F.round(F.sum("penalized_operation_amount") / F.countDistinct("PAN_ENCRYPTED"),4).alias('avg_pen_amount_per_card'),
    F.round(F.sum("penalized_operation_amount") / F.sum("OPERATION_AMOUNT"),4).alias('avg_pen_amount_norm'),
    F.sum("DAILY_CANCELLED_CARD_COUNT_cancellations").alias("total_cancellations"),  
    F.count("*").alias("total_tx"),
    F.sum("is_remote").alias("remote_tx"),
    F.countDistinct("PAN_ENCRYPTED").alias("unique_cards"),
    F.countDistinct("MERCHANT_NAME").alias("unique_merchants"),
    F.countDistinct("MCC").alias("mcc_distinct_counts"),
    F.round(F.sum("OPERATION_AMOUNT") / F.countDistinct("PAN_ENCRYPTED"), 4).alias("avg_unique_card_amount"),
    F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)).alias("total_rejected"),
    F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)).alias("total_accepted"),
    F.round(F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0))/ F.countDistinct("MERCHANT_NAME"),4).alias("avg_rejected_tx_merchant"),
    F.sum("fraud_amount_accepted").alias("fraud_amount_accepted"),
    F.sum("OPERATION_AMOUNT").alias("total_operation_amount"),
    F.first("month_sin").alias("month_sin"),
    F.first("month_cos").alias("month_cos"),
    F.first("dom_cos").alias("day_of_month_cos"),
    F.first("dom_sin").alias("day_of_month_sin"),
    F.count(F.col("number_tx_exceeded")).alias("number_tx_exceeded"),
    F.count(F.col("cash_withdrawal_exceeded")).alias("cash_withdrawal_exceeded"),
    F.count(F.col("card_number_invalid")).alias("card_number_invalid"),
    F.count(F.col("invalid_pin")).alias("invalid_pin"),
    F.count(F.col("invalid_cvv")).alias("invalid_cvv"),
    F.countDistinct(F.concat_ws("_", F.col("PAN_ENCRYPTED"), F.col("MERCHANT_NAME"))).alias("unique_merchant_card_pairs"),
    F.sum("is_new_merchant").alias("new_merchant_tx"),
    F.sum(F.when(F.col("is_new_merchant") == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("new_merchant_amount"),
    F.round( F.sum('OPERATION_AMOUNT')/F.count('*'),4).alias('amount_transaction_ratio'),
    F.round( F.count('*')/F.countDistinct("MERCHANT_NAME"),4).alias('avg_transaction_per_merchant'),
    F.round( F.sum("penalized_operation_amount") / F.countDistinct("PAN_ENCRYPTED"),4).alias('avg_pen_amount_per_card'),
    F.round( F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)) / F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)),4).alias('rejected_accepted_ratio'),
    F.round( F.sum("is_remote") / F.count('*'),4).alias('remote_ratio'),
    F.round(F.sum("is_high_risk_merchant")/ F.count('*'), 4).alias('high_risk_tx_ratio'),
    F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("vinted"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("vinted_amount"),
    F.sum(F.when(F.col("MCC") == "6051",1).otherwise(0)).alias("mcc_6051_count"),
    F.sum(F.when(F.col("MCC") == "4829",1).otherwise(0)).alias("mcc_4829_count"),
    F.sum(F.when(F.col("MCC") == "6051", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_6051_amount"),
    F.sum(F.when(F.col("MCC") == "4829", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_4829_amount"),
    F.sum(F.when(F.col("MCC") == "7995", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_7995_amount"),
    F.sum(F.when(F.col("MCC") == "5999", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_5999_amount"),
    F.sum(F.when(F.col("MCC") == "4722", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_4722_amount"),
    F.sum(F.when(F.col("MCC") == "5944", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_5944_amount"),
    F.sum(F.when(F.col("MERCHANT_NAME") == "BETANO PT", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("BETANO_PT_amount"),
    F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("binance.com"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("binance_amount"),
    F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("bifinity"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("bifinity_amount")

).orderBy("TRANSACTION_DATE", ascending = True)

## 2.7 Merging Features and the Daily Data

- We are itteratively converting the Spark-Dataframes into Pandas-Dataframes
- We choose this sequential way for debugging pruposes, as some Spark-Dataframe had an Overflow error

daily_pd = daily_data.toPandas()

rolling_stats_7_pd = rolling_stats_7.toPandas()

rolling_stats_14_pd = rolling_stats_14.toPandas()

lag1_df_pd = lag1_df.toPandas()

lag7_df_pd = lag7_df.toPandas()

lag14_df_pd = lag14_df.toPandas()

df_pd = daily_pd.merge(rolling_stats_7_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
df_pd = df_pd.merge(rolling_stats_14_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
df_pd = df_pd.merge(lag1_df_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
df_pd = df_pd.merge(lag7_df_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
df_pd = df_pd.merge(lag14_df_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])

df_pd = df_pd.bfill() # total_cancellations had a small number of missing values, which we back filled
df_pd = df_pd.sort_values("TRANSACTION_DATE", ascending = True) 
df_pd = df_pd.reset_index(drop = True)

df_pd.to_csv("/home/cdsw/full_final_features.csv")

## 2.8 Creating Fraud Labelling Delay Corrected Target Variable 
   - For model performance evaluation purposes, we are creating a corrected version of the accepted fraud amount time series, accounting for fraud labelling delay
   - Some fraudulent transactions can take up to several months to be labelled
   - We computed the percentage of fraudulent transactions labelled after a given number of days: for instance, suppose the proportion of fraud properly labelled 1 day after the transaction occurred is 1.15% and the proportion of fraud labelled after 10 days is 12%
   - Once the last day of labelling is established, we apply each cummulative labelling percentage to each corresponding lag, thereby achieving a accepted fraud amount value more representative of the actual amount

df_delay =training.select(
    'id',  
    'TRANSACTION_DATETIME',
    'fraud_classification_datetime',
    'fraud_label'
    
).filter('fraud_label == 1')

df_delay = df_delay.withColumn(
    "LABEL_DELAY_DAYS",
    (F.col("fraud_classification_datetime").cast("long") - F.col("TRANSACTION_DATETIME").cast("long"))/86400
).withColumn(
    "delay_days_rounded",
    F.floor(F.col('LABEL_DELAY_DAYS'))
)

counts_by_day = df_delay.groupBy('delay_days_rounded').agg(
    F.count('*').alias('labelled_count')
).orderBy('delay_days_rounded')

total_transactions = df_delay.count()

w = Window.orderBy('delay_days_rounded').rowsBetween(Window.unboundedPreceding, 0)

counts_by_day = counts_by_day.withColumn(
    'cummulative_labelled', 
    F.sum('labelled_count').over(w)
)
    
counts_by_day = counts_by_day.withColumn(
    'pct_labelled',
    (F.col('cummulative_labelled')/F.lit(total_transactions))*100
)
    

label_curve = counts_by_day.select(
    'delay_days_rounded',
    'pct_labelled'
).filter('delay_days_rounded >= 0')

label_curve =label_curve.toPandas()

label_curve.to_csv('/home/cdsw/full_labelling_curve.csv', index = False)

labeling_curve_dict = dict(zip(label_curve['delay_days_rounded'], (label_curve['pct_labelled'])/100))

# Calculate days since transaction
dataset_given_date = pd.Timestamp('2025-06-06')

df_pd['days_since_transaction'] = (dataset_given_date -pd.to_datetime(df_pd['TRANSACTION_DATE'])).dt.days

# Cap days (just in case you go beyond maximum labeling curve)
max_known_day = max(labeling_curve_dict.keys())
df_pd['days_since_transaction'] = df_pd['days_since_transaction'].clip(upper=max_known_day)

# Apply correction
df_pd['correction_factor'] = df_pd['days_since_transaction'].map(labeling_curve_dict)
df_pd['corrected_accepted_fraud_amount'] = ((df_pd['fraud_amount_accepted']/100) / df_pd['correction_factor'])*100

df_pd.to_csv('/home/cdsw/fully_corrected_data.csv', index = False)

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(12, 6))
plt.plot(df_pd['TRANSACTION_DATE'], df_pd['corrected_accepted_fraud_amount'], label='Corrected', color='red')
plt.plot(df_pd['TRANSACTION_DATE'], df_pd['fraud_amount_accepted'], label='Actual')
plt.title("Accepted Fraud Amount: Actual vs Corrected")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 3. Modelling
## 3.1 Training / Test Split

- Training starts at the `2023-06-01` and stops at the `2024-12-31`
- Testing starts from the `2025-01-01` and stops at the `2025-04-30`

df_pd['TRANSACTION_DATE'] = pd.to_datetime(df_pd['TRANSACTION_DATE'])

train_test_data = df_pd[(df_pd['TRANSACTION_DATE'] >= '2023-06-01') & (df_pd['TRANSACTION_DATE'] <= '2024-12-31')]
out_of_sample = df_pd[(df_pd['TRANSACTION_DATE'] >= '2025-01-01') & (df_pd['TRANSACTION_DATE'] <= '2025-04-30')]

## 3.2 Feature Importance Testing and Selection

We are going to select the features, based on the following metrics:
 - MI-Score derived from the Mutual Information Analysis
 - Shapley values derived from the SHAP analysis 
 - Business context

### 3.2.1 Mutual Information Analysis 
from sklearn.feature_selection import mutual_info_regression

X = train_test_data.drop(columns=['TRANSACTION_DATE', 'fraud_amount_accepted','days_since_transaction','correction_factor','corrected_accepted_fraud_amount'], axis=1).fillna(0)
y = train_test_data['fraud_amount_accepted']

mi_scores = mutual_info_regression(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores}).sort_values(by='MI_Score', ascending=False)

mi_df.head(15)

### 3.2.2 SHAP Analysis 
df_train = train_test_data.copy()
df_test = out_of_sample.copy()
df_test = df_test.bfill().fillna(0)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline


# Save and drop date columns
train_dates = df_train["TRANSACTION_DATE"]
test_dates = df_test["TRANSACTION_DATE"]
df_train = df_train.drop(columns=["TRANSACTION_DATE"])
df_test = df_test.drop(columns=["TRANSACTION_DATE"])

# Manual Oversampling: Duplicate peak days (>100K fraud) 5x
threshold = 100000
peak_days = df_train[df_train["corrected_accepted_fraud_amount"] > threshold]
normal_days = df_train[df_train["corrected_accepted_fraud_amount"] <= threshold]
df_train_balanced = pd.concat([normal_days, pd.concat([peak_days]*5)], ignore_index=True)

# Prepare features and target
X_train = df_train_balanced.drop(columns=['fraud_amount_accepted', 'days_since_transaction', 'correction_factor', 'corrected_accepted_fraud_amount'], axis=1)
y_train = df_train_balanced["corrected_accepted_fraud_amount"]

X_test = df_test.drop(columns=['fraud_amount_accepted', 'days_since_transaction', 'correction_factor', 'corrected_accepted_fraud_amount'], axis=1)
y_test = df_test["corrected_accepted_fraud_amount"]

# Random Forest with hyperparameter tuning
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

import shap
import matplotlib.pyplot as plt
shap.initjs()
 
# Step 1: Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(best_rf)
 
# Step 2: Get SHAP values for the test set
shap_values = explainer.shap_values(X_test)
 
# Step 3: Summary plot of feature importances
shap.summary_plot(shap_values, X_test)
 
# Step 4: Detailed view of feature importance for a specific prediction (e.g., first instance in the test set)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

## 3.3 Final Model: Random Forest Regressor 

The final model chosen is a Random Forest Regressor, trained from June 2023 through December 2024 and subsequently evaluated from January 2025 through April 2025. The choice of selected features was based on MI-scores, Shapley values and on the business problem context. The model's optimal hyperparameter combination was fine-tuned through the Grid Search Cross Validation method.
df_train = train_test_data.copy()
df_test = out_of_sample.copy()
df_test = df_test.bfill().fillna(0)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline


# 2. Save and drop date columns
train_dates = df_train["TRANSACTION_DATE"]
test_dates = df_test["TRANSACTION_DATE"]
df_train = df_train.drop(columns=["TRANSACTION_DATE"])
df_test = df_test.drop(columns=["TRANSACTION_DATE"])

# 3. Manual Oversampling: Duplicate peak days (>100K fraud) 5x
threshold = 100000
peak_days = df_train[df_train["corrected_accepted_fraud_amount"] > threshold]
normal_days = df_train[df_train["corrected_accepted_fraud_amount"] <= threshold]
df_train_balanced = pd.concat([normal_days, pd.concat([peak_days]*5)], ignore_index=True)

# 4. Prepare features and target

features =  ['unique_cards', 'total_rejected', 'total_accepted',
       'mcc_4829_count',
       'total_operation_amount', 'total_tx', 'remote_tx', 'unique_merchant_card_pairs',
       'new_merchant_tx', 'total_cancellations', 
       'month_sin', 'month_cos', 
       'is_weekend', 'number_tx_exceeded', 'cash_withdrawal_exceeded',
       'card_number_invalid', 'invalid_pin', 'invalid_cvv',
       'rejected_accepted_ratio', 'remote_ratio', 'amount_transaction_ratio',
       'avg_transaction_per_merchant', 
       'rolling_mean_7', 'rolling_max_7', 'rolling_min_7', 'rolling_sum_7',
       'rolling_std_7', 'rolling_mean_14', 'rolling_max_14', 'rolling_min_14',
       'rolling_sum_14', 'rolling_std_14', 'lag_1', 'lag_7', 'lag_14',
       'day_of_month_cos', 'day_of_month_sin',
       'mcc_6051_amount', 'mcc_4829_amount', 'small_merchant_amount',
       'big_merchant_amount', 'new_merchant_amount', 'mcc_7995_amount',
       'mcc_5999_amount', 'mcc_4722_amount',
       'penalized_total_operation_amount', 'avg_pen_amount_per_card',
       'avg_pen_amount_norm']


X_train = df_train_balanced[features]
y_train = df_train_balanced["corrected_accepted_fraud_amount"]

X_test = df_test[features]
y_test = df_test["corrected_accepted_fraud_amount"]

# 5. Random Forest with hyperparameter tuning
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# 9. Predict on training set (oversampled)
y_train_pred = best_rf.predict(X_train)

# 6. Predict on out-of-sample
y_pred = best_rf.predict(X_test)

# 7. Plot Actual vs. Predicted
plt.figure(figsize=(14, 6))
plt.plot(test_dates.values, y_test.values, label='Actual', linewidth=2)
plt.plot(test_dates.values, y_pred, label='Predicted', linewidth=2)
plt.title("Corrected Actual vs. Predicted Fraud Amount (Out-of-Sample)")
plt.xlabel("Date")
plt.ylabel("Fraud Amount Accepted")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Final Model Evaluation

Model performance will be evaluated on 1, 7 and 30 days, according to:
 - Mean Absolute Error
 - Mean Absolute Percentage Error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

if isinstance(y_pred, np.ndarray):
    y_pred = pd.Series(y_pred, index=y_test.index)

print(f"\nEvaluation on {len(y_pred)} test days:")

if len(y_pred) >= 1:
    y1, p1 = y_test.iloc[:1], y_pred.iloc[:1]
    print(f"1-Day → MAE: {mean_absolute_error(y1, p1):.2f}, "
          f"MAPE: {np.mean(np.abs((y1 - p1) / y1)) * 100:.2f}%, "
          f"Penalized MAE: {penalized_mae_function(y1.values, p1.values):.2f}")

if len(y_pred) >= 7:
    y7, p7 = y_test.iloc[:7], y_pred.iloc[:7]
    print(f"7-Day → MAE: {mean_absolute_error(y7, p7):.2f}, "
          f"MAPE: {np.mean(np.abs((y7 - p7) / y7)) * 100:.2f}%, "
          f"Penalized MAE: {penalized_mae_function(y7.values, p7.values):.2f}")

if len(y_pred) >= 30:
    y30, p30 = y_test.iloc[:30], y_pred.iloc[:30]
    print(f"30-Day → MAE: {mean_absolute_error(y30, p30):.2f}, "
          f"MAPE: {np.mean(np.abs((y30 - p30) / y30)) * 100:.2f}%, "
          f"Penalized MAE: {penalized_mae_function(y30.values, p30.values):.2f}")
