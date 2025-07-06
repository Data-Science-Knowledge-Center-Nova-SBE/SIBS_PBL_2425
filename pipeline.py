"""
Fraud Business Impact Forecasting: NOVA x SIBS Pipeline

This pipeline processes transaction data to create features for fraud amount prediction.
The pipeline consists of three main phases:
1. Spark-based feature engineering
2. Data transformation and aggregation
3. Machine learning model training and evaluation
"""

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col, avg, when, sin, cos, lit, to_date, date_sub, sum as Fsum
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import shap


def initialize_spark_session(session_name="fraud_pipeline"):
    """
    Initialize Spark session with optimized configurations for the fraud detection pipeline.
    
    Args:
        session_name (str): Name for the Spark session
        
    Returns:
        SparkSession: Configured Spark session
    """
    python_packages_path = "/home/cdsw/ml_new.tar.gz#environment"
    executor_memory = "10g"
    driver_memory = "8g"
    executor_cores = "3"
    maxExecutors = 15
    memory_overhead = "2g"
    n_partitions = 200
    driver_memory_overhead = 8000

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
            .config("spark.yarn.driver.memoryOverhead", driver_memory_overhead)\
            .config("spark.yarn.queue", "root.nova_sbe")\
            .appName(session_name)\
            .getOrCreate()
    
    return spark


def load_data(spark):
    """
    Load training, out-of-sample, and card status data from Spark tables.
    
    Args:
        spark (SparkSession): Active Spark session
        
    Returns:
        tuple: (training_df, oos_df, card_status_full_df) - loaded DataFrames
    """
    # Load training and out-of-sample datasets
    training = spark.table("nova_sbe.raw_202306_202412")
    card_status = spark.table("nova_sbe.card_status_202306_202412")
    oos = spark.table("nova_sbe.sample_202412")
    card_status_outsample = spark.table("nova_sbe.card_status_202501_202504")
    
    # Ensure out-of-sample has same columns as training
    oos = oos[training.columns]
    
    # Concatenate card status DataFrames
    card_status_full = card_status.union(card_status_outsample)
    
    return training, oos, card_status_full


def create_target_variable(df, training_df):
    """
    Create fraud-related target variables and labels.
    
    Args:
        df (DataFrame): Main DataFrame to add target variables to
        training_df (DataFrame): Training DataFrame for creating fraud labels
        
    Returns:
        DataFrame: DataFrame with added target variables
    """
    # Create fraud label in training dataset to avoid data leakage
    training_df = training_df.withColumn(
        "fraud_label",
        F.when(F.col('fraud_sp') > 0, 1)  # Confirmed fraud
        .when(F.col('fraud_sp') <= 0, -1)  # Confirmed genuine
        .otherwise(0)  # Unknown
    )
    
    # Create fraud label in main dataset
    df = df.withColumn(
        "fraud_label",
        F.when(F.col('fraud_sp') > 0, 1)
        .when(F.col('fraud_sp') <= 0, -1)
        .otherwise(0)
    )
    
    # Create fraud amount (amount when transaction is fraud)
    df = df.withColumn(
        "fraud_amount",
        when(col("fraud_label") == 1, col("OPERATION_AMOUNT")).otherwise(0)
    )
    
    # Create transaction accepted flag
    df = df.withColumn(
        "TRANSACTION_ACCEPTED",
        F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)
    )
    
    # Create fraud amount accepted (target variable)
    df = df.withColumn(
        "fraud_amount_accepted",
        when(col("TRANSACTION_ACCEPTED") == 1, col("fraud_amount")).otherwise(0)
    )
    
    return df


def create_time_features(df):
    """
    Create time-based features including cyclical transformations.
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: DataFrame with added time features
    """
    # Extract basic time components
    df = df.withColumn("TRANSACTION_MONTH", F.month(F.col("TRANSACTION_DATETIME")))\
           .withColumn("HOUR_OF_DAY", F.hour(F.col("TRANSACTION_DATETIME")))\
           .withColumn("DAY_OF_WEEK", F.dayofweek(F.col("TRANSACTION_DATETIME")))\
           .withColumn("DAY_OF_MONTH", F.dayofmonth(F.col("TRANSACTION_DATETIME")))\
           .withColumn("TRANSACTION_DATE", F.to_date("TRANSACTION_DATETIME"))
    
    # Sin/Cosine transformations for cyclical features
    max_day = 31
    max_month = 12
    pi = 3.141592653589793
    
    # Day of month cyclical transformation
    df = df.withColumn("dom_angle", 2 * lit(pi) * col("DAY_OF_MONTH") / lit(max_day))\
           .withColumn("dom_sin", sin(col("dom_angle")))\
           .withColumn("dom_cos", cos(col("dom_angle")))\
           .drop("dom_angle")
    
    # Month cyclical transformation
    df = df.withColumn("month_angle", 2 * lit(pi) * col("TRANSACTION_MONTH") / lit(max_month))\
           .withColumn("month_sin", sin(col("month_angle")))\
           .withColumn("month_cos", cos(col("month_angle")))\
           .drop("month_angle")
    
    # Weekend indicator
    df = df.withColumn("is_weekend", F.when(F.dayofweek("TRANSACTION_DATE").isin(1, 7), 1).otherwise(0))
    
    return df


def create_rolling_features(df):
    """
    Create rolling statistics and lag features for fraud amounts.
    
    Args:
        df (DataFrame): Input DataFrame with date columns
        
    Returns:
        tuple: (df, rolling_stats_7, rolling_stats_14, lag1_df, lag7_df, lag14_df)
    """
    # Ensure date columns exist
    df = df.withColumn("day", F.to_date("TRANSACTION_DATETIME"))\
           .withColumn("classification_day", F.to_date("fraud_classification_datetime"))
    
    # Prepare data for rolling calculations
    df_valid = df.select("day", "classification_day", "fraud_amount_accepted")
    target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")
    
    # Cross join and filter for valid combinations
    joined = df_valid.crossJoin(target_days)\
        .filter(F.col("day") < F.col("target_day"))\
        .filter(F.col("classification_day") <= F.col("target_day"))
    
    # Aggregate daily fraud amounts
    daily_agg_per_target = joined.groupBy("target_day", "day").agg(
        F.sum("fraud_amount_accepted").alias("daily_fraud_sum")
    )
    
    # 7-day rolling statistics
    rolling_base_7 = daily_agg_per_target\
        .withColumn("days_diff", F.datediff(F.col("target_day"), F.col("day")))\
        .filter(F.col("days_diff").between(1, 7))
    
    rolling_stats_7 = rolling_base_7.groupBy("target_day").agg(
        F.mean("daily_fraud_sum").alias("rolling_mean_7"),
        F.max("daily_fraud_sum").alias("rolling_max_7"),
        F.min("daily_fraud_sum").alias("rolling_min_7"),
        F.sum("daily_fraud_sum").alias("rolling_sum_7"),
        F.stddev("daily_fraud_sum").alias("rolling_std_7")
    ).orderBy("target_day")
    
    # 14-day rolling statistics
    rolling_base_14 = daily_agg_per_target\
        .withColumn("days_diff", F.datediff(F.col("target_day"), F.col("day")))\
        .filter(F.col("days_diff").between(1, 14))
    
    rolling_stats_14 = rolling_base_14.groupBy("target_day").agg(
        F.mean("daily_fraud_sum").alias("rolling_mean_14"),
        F.max("daily_fraud_sum").alias("rolling_max_14"),
        F.min("daily_fraud_sum").alias("rolling_min_14"),
        F.sum("daily_fraud_sum").alias("rolling_sum_14"),
        F.stddev("daily_fraud_sum").alias("rolling_std_14")
    ).orderBy("target_day")
    
    # Create lag features
    target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")
    
    # 1-day lag
    df_lag1 = df.crossJoin(target_days)\
        .filter(col("day") == date_sub(col("target_day"), 1))\
        .filter(col("classification_day") <= col("target_day"))
    lag1_df = df_lag1.groupBy("target_day").agg(Fsum("fraud_amount_accepted").alias("lag_1"))
    
    # 7-day lag
    df_lag7 = df.crossJoin(target_days)\
        .filter(col("day") == date_sub(col("target_day"), 7))\
        .filter(col("classification_day") <= col("target_day"))
    lag7_df = df_lag7.groupBy("target_day").agg(Fsum("fraud_amount_accepted").alias("lag_7"))
    
    # 14-day lag
    df_lag14 = df.crossJoin(target_days)\
        .filter(col("day") == date_sub(col("target_day"), 14))\
        .filter(col("classification_day") <= col("target_day"))
    lag14_df = df_lag14.groupBy("target_day").agg(Fsum("fraud_amount_accepted").alias("lag_14"))
    
    return df, rolling_stats_7, rolling_stats_14, lag1_df, lag7_df, lag14_df


def create_merchant_features(df, training_df):
    """
    Create merchant-related features including risk assessment and categorization.
    
    Args:
        df (DataFrame): Main DataFrame
        training_df (DataFrame): Training DataFrame for risk calculations
        
    Returns:
        DataFrame: DataFrame with added merchant features
    """
    # Merchant frequency and new merchant detection
    window_spec = Window.partitionBy("MERCHANT_NAME").orderBy("TRANSACTION_DATE")
    df = df.withColumn("first_seen_date", F.min("TRANSACTION_DATE").over(window_spec))
    df = df.withColumn("is_new_merchant", (F.col("TRANSACTION_DATE") == F.col("first_seen_date")).cast("int"))
    
    # Merchant size categorization (small vs big merchants)
    merchant_freq_df = df.groupBy("MERCHANT_NAME").agg(F.count("*").alias("tx_count"))
    window_spec = Window.orderBy("tx_count")
    merchant_ranked_df = merchant_freq_df.withColumn("percent_rank", F.percent_rank().over(window_spec))
    
    merchant_labeled_df = merchant_ranked_df.withColumn(
        "small_merchant", F.when(F.col("percent_rank") <= 0.05, 1).otherwise(0)
    ).withColumn(
        "big_merchant", F.when(F.col("percent_rank") > 0.05, 1).otherwise(0)
    ).select("MERCHANT_NAME", "small_merchant", "big_merchant")
    
    df = df.join(merchant_labeled_df, on="MERCHANT_NAME", how="left")
    
    # MCC fraud risk ranking
    mcc_fraud_pct = training_df.groupBy("MCC").agg(
        F.round(100 * F.count(F.when(F.col('fraud_label') == 1, 1)) / F.count('*'), 4).alias('mcc_fraud_rate')
    )
    
    window_spec = Window.orderBy(F.col("mcc_fraud_rate").asc())
    ranked_mcc = mcc_fraud_pct.withColumn("fraud_rank", F.dense_rank().over(window_spec))
    ranked_mcc_df_filtered = ranked_mcc.select("MCC", "fraud_rank")
    
    df = df.join(ranked_mcc_df_filtered, on="MCC", how="left")
    df = df.withColumnRenamed("fraud_rank", "fraud_risk_rank")
    
    # Penalized operation amount based on risk
    df = df.withColumn("penalized_operation_amount", F.col("OPERATION_AMOUNT") * F.col("fraud_risk_rank"))
    
    # High-risk merchant identification
    fraud_threshold = mcc_fraud_pct.approxQuantile("mcc_fraud_rate", [0.95], 0.01)[0]
    high_risk_mcc = mcc_fraud_pct.withColumn(
        "is_high_risk_merchant", F.when(F.col("mcc_fraud_rate") >= fraud_threshold, 1).otherwise(0)
    ).select("MCC", "is_high_risk_merchant")
    
    df = df.join(high_risk_mcc, on="MCC", how="left").fillna({"is_high_risk_merchant": 0})
    
    # Merchant name processing
    df = df.withColumn("MERCHANT_NAME_LOWER", F.lower(F.col("MERCHANT_NAME")))
    
    return df


def create_card_features(df, card_status_full):
    """
    Create card-related features including authorization response codes and cancellations.
    
    Args:
        df (DataFrame): Main DataFrame
        card_status_full (DataFrame): Card status DataFrame
        
    Returns:
        DataFrame: DataFrame with added card features
    """
    # Authorization response code features
    df = df.withColumn("number_tx_exceeded", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 65, 1).otherwise(0))
    df = df.withColumn("cash_withdrawal_exceeded", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 61, 1).otherwise(0))
    df = df.withColumn("card_number_invalid", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 14, 1).otherwise(0))
    df = df.withColumn("invalid_pin", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 55, 1).otherwise(0))
    df = df.withColumn("invalid_cvv", F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 82, 1).otherwise(0))
    
    # Card cancellations
    card_status_full = card_status_full.withColumn("TRANSACTION_DATE", to_date(col("TRANSACTION_DATETIME")))
    card_cancellations = card_status_full.filter(col("NEW_CARD_STATUS").isin("06", "09"))
    daily_cancellations = card_cancellations.groupBy("TRANSACTION_DATE").count().withColumnRenamed("count", "DAILY_CANCELLED_CARD_COUNT")
    daily_cancellations = daily_cancellations.withColumnRenamed("DAILY_CANCELLED_CARD_COUNT", "DAILY_CANCELLED_CARD_COUNT_cancellations")
    
    df = df.join(daily_cancellations, on="TRANSACTION_DATE", how="left")
    
    # Remote transaction indicator
    df = df.withColumn("is_remote", F.col("FRAUD_ACCEPTOR_CDE").startswith("REMOTE").cast("int"))
    
    return df


def aggregate_daily_features(df):
    """
    Aggregate all features on a daily basis.
    
    Args:
        df (DataFrame): DataFrame with all features
        
    Returns:
        DataFrame: Daily aggregated DataFrame
    """
    daily_data = df.groupBy("TRANSACTION_DATE", "is_weekend").agg(
        F.sum(F.when(df["small_merchant"] == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("small_merchant_amount"),
        F.sum(F.when(df["big_merchant"] == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("big_merchant_amount"),
        F.sum("penalized_operation_amount").alias("penalized_total_operation_amount"),
        F.round(F.sum("penalized_operation_amount") / F.countDistinct("PAN_ENCRYPTED"), 4).alias('avg_pen_amount_per_card'),
        F.round(F.sum("penalized_operation_amount") / F.sum("OPERATION_AMOUNT"), 4).alias('avg_pen_amount_norm'),
        F.sum("DAILY_CANCELLED_CARD_COUNT_cancellations").alias("total_cancellations"),
        F.count("*").alias("total_tx"),
        F.sum("is_remote").alias("remote_tx"),
        F.countDistinct("PAN_ENCRYPTED").alias("unique_cards"),
        F.countDistinct("MERCHANT_NAME").alias("unique_merchants"),
        F.countDistinct("MCC").alias("mcc_distinct_counts"),
        F.round(F.sum("OPERATION_AMOUNT") / F.countDistinct("PAN_ENCRYPTED"), 4).alias("avg_unique_card_amount"),
        F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)).alias("total_rejected"),
        F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)).alias("total_accepted"),
        F.round(F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)) / F.countDistinct("MERCHANT_NAME"), 4).alias("avg_rejected_tx_merchant"),
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
        F.round(F.sum('OPERATION_AMOUNT') / F.count('*'), 4).alias('amount_transaction_ratio'),
        F.round(F.count('*') / F.countDistinct("MERCHANT_NAME"), 4).alias('avg_transaction_per_merchant'),
        F.round(F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)) / F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)), 4).alias('rejected_accepted_ratio'),
        F.round(F.sum("is_remote") / F.count('*'), 4).alias('remote_ratio'),
        F.round(F.sum("is_high_risk_merchant") / F.count('*'), 4).alias('high_risk_tx_ratio'),
        F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("vinted"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("vinted_amount"),
        F.sum(F.when(F.col("MCC") == "6051", 1).otherwise(0)).alias("mcc_6051_count"),
        F.sum(F.when(F.col("MCC") == "4829", 1).otherwise(0)).alias("mcc_4829_count"),
        F.sum(F.when(F.col("MCC") == "6051", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_6051_amount"),
        F.sum(F.when(F.col("MCC") == "4829", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_4829_amount"),
        F.sum(F.when(F.col("MCC") == "7995", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_7995_amount"),
        F.sum(F.when(F.col("MCC") == "5999", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_5999_amount"),
        F.sum(F.when(F.col("MCC") == "4722", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_4722_amount"),
        F.sum(F.when(F.col("MCC") == "5944", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_5944_amount"),
        F.sum(F.when(F.col("MERCHANT_NAME") == "BETANO PT", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("BETANO_PT_amount"),
        F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("binance.com"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("binance_amount"),
        F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("bifinity"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("bifinity_amount")
    ).orderBy("TRANSACTION_DATE", ascending=True)
    
    return daily_data


def convert_to_pandas(daily_data, rolling_stats_7, rolling_stats_14, lag1_df, lag7_df, lag14_df):
    """
    Convert Spark DataFrames to Pandas and merge all features.
    
    Args:
        daily_data, rolling_stats_7, rolling_stats_14, lag1_df, lag7_df, lag14_df: Spark DataFrames
        
    Returns:
        pd.DataFrame: Merged Pandas DataFrame with all features
    """
    # Convert to Pandas
    daily_pd = daily_data.toPandas()
    rolling_stats_7_pd = rolling_stats_7.toPandas()
    rolling_stats_14_pd = rolling_stats_14.toPandas()
    lag1_df_pd = lag1_df.toPandas()
    lag7_df_pd = lag7_df.toPandas()
    lag14_df_pd = lag14_df.toPandas()
    
    # Merge all features
    df_pd = daily_pd.merge(rolling_stats_7_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    df_pd = df_pd.merge(rolling_stats_14_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    df_pd = df_pd.merge(lag1_df_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    df_pd = df_pd.merge(lag7_df_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    df_pd = df_pd.merge(lag14_df_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    
    # Handle missing values
    df_pd = df_pd.bfill()
    df_pd = df_pd.sort_values("TRANSACTION_DATE", ascending=True)
    df_pd = df_pd.reset_index(drop=True)
    
    return df_pd


def create_corrected_target(df_pd, training_df):
    """
    Create fraud labelling delay corrected target variable.
    
    Args:
        df_pd (pd.DataFrame): Main DataFrame
        training_df (DataFrame): Training Spark DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with corrected target variable
    """
    # Calculate fraud labelling delay
    df_delay = training_df.select(
        'id', 'TRANSACTION_DATETIME', 'fraud_classification_datetime', 'fraud_label'
    ).filter('fraud_label == 1')
    
    df_delay = df_delay.withColumn(
        "LABEL_DELAY_DAYS",
        (F.col("fraud_classification_datetime").cast("long") - F.col("TRANSACTION_DATETIME").cast("long")) / 86400
    ).withColumn(
        "delay_days_rounded",
        F.floor(F.col('LABEL_DELAY_DAYS'))
    )
    
    # Calculate labelling curve
    counts_by_day = df_delay.groupBy('delay_days_rounded').agg(
        F.count('*').alias('labelled_count')
    ).orderBy('delay_days_rounded')
    
    total_transactions = df_delay.count()
    w = Window.orderBy('delay_days_rounded').rowsBetween(Window.unboundedPreceding, 0)
    
    counts_by_day = counts_by_day.withColumn(
        'cummulative_labelled', F.sum('labelled_count').over(w)
    ).withColumn(
        'pct_labelled', (F.col('cummulative_labelled') / F.lit(total_transactions)) * 100
    )
    
    label_curve = counts_by_day.select('delay_days_rounded', 'pct_labelled').filter('delay_days_rounded >= 0')
    label_curve_pd = label_curve.toPandas()
    
    # Apply correction
    labeling_curve_dict = dict(zip(label_curve_pd['delay_days_rounded'], (label_curve_pd['pct_labelled']) / 100))
    dataset_given_date = pd.Timestamp('2025-06-06')
    
    df_pd['days_since_transaction'] = (dataset_given_date - pd.to_datetime(df_pd['TRANSACTION_DATE'])).dt.days
    max_known_day = max(labeling_curve_dict.keys())
    df_pd['days_since_transaction'] = df_pd['days_since_transaction'].clip(upper=max_known_day)
    
    df_pd['correction_factor'] = df_pd['days_since_transaction'].map(labeling_curve_dict)
    df_pd['corrected_accepted_fraud_amount'] = ((df_pd['fraud_amount_accepted'] / 100) / df_pd['correction_factor']) * 100
    
    return df_pd


def prepare_modeling_data(df_pd):
    """
    Prepare data for modeling by creating training and test splits.
    
    Args:
        df_pd (pd.DataFrame): Complete DataFrame with all features
        
    Returns:
        tuple: (train_data, test_data) - training and testing DataFrames
    """
    df_pd['TRANSACTION_DATE'] = pd.to_datetime(df_pd['TRANSACTION_DATE'])
    
    train_data = df_pd[(df_pd['TRANSACTION_DATE'] >= '2023-06-01') & (df_pd['TRANSACTION_DATE'] <= '2024-12-31')]
    test_data = df_pd[(df_pd['TRANSACTION_DATE'] >= '2025-01-01') & (df_pd['TRANSACTION_DATE'] <= '2025-04-30')]
    
    return train_data, test_data


def feature_selection_analysis(train_data):
    """
    Perform feature selection analysis using Mutual Information and SHAP.
    
    Args:
        train_data (pd.DataFrame): Training data
        
    Returns:
        tuple: (mi_df, shap_values) - MI scores and SHAP analysis results
    """
    # Mutual Information Analysis
    X = train_data.drop(columns=['TRANSACTION_DATE', 'fraud_amount_accepted', 'days_since_transaction', 'correction_factor', 'corrected_accepted_fraud_amount'], axis=1).fillna(0)
    y = train_data['fraud_amount_accepted']
    
    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores}).sort_values(by='MI_Score', ascending=False)
    
    print("Top 15 Features by Mutual Information Score:")
    print(mi_df.head(15))
    
    return mi_df


def train_and_evaluate_model(train_data, test_data):
    """
    Train Random Forest model and evaluate performance.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Testing data
        
    Returns:
        tuple: (best_model, predictions, test_dates) - trained model, predictions, and test dates
    """
    # Prepare data
    train_dates = train_data["TRANSACTION_DATE"]
    test_dates = test_data["TRANSACTION_DATE"]
    
    df_train = train_data.drop(columns=["TRANSACTION_DATE"])
    df_test = test_data.drop(columns=["TRANSACTION_DATE"]).bfill().fillna(0)
    
    # Manual oversampling for imbalanced data
    threshold = 100000
    peak_days = df_train[df_train["corrected_accepted_fraud_amount"] > threshold]
    normal_days = df_train[df_train["corrected_accepted_fraud_amount"] <= threshold]
    df_train_balanced = pd.concat([normal_days, pd.concat([peak_days] * 5)], ignore_index=True)
    
    # Feature selection based on analysis
    features = ['unique_cards', 'total_rejected', 'total_accepted', 'mcc_4829_count',
                'total_operation_amount', 'total_tx', 'remote_tx', 'unique_merchant_card_pairs',
                'new_merchant_tx', 'total_cancellations', 'month_sin', 'month_cos',
                'is_weekend', 'number_tx_exceeded', 'cash_withdrawal_exceeded',
                'card_number_invalid', 'invalid_pin', 'invalid_cvv',
                'rejected_accepted_ratio', 'remote_ratio', 'amount_transaction_ratio',
                'avg_transaction_per_merchant', 'rolling_mean_7', 'rolling_max_7',
                'rolling_min_7', 'rolling_sum_7', 'rolling_std_7', 'rolling_mean_14',
                'rolling_max_14', 'rolling_min_14', 'rolling_sum_14', 'rolling_std_14',
                'lag_1', 'lag_7', 'lag_14', 'day_of_month_cos', 'day_of_month_sin',
                'mcc_6051_amount', 'mcc_4829_amount', 'small_merchant_amount',
                'big_merchant_amount', 'new_merchant_amount', 'mcc_7995_amount',
                'mcc_5999_amount', 'mcc_4722_amount', 'penalized_total_operation_amount',
                'avg_pen_amount_per_card', 'avg_pen_amount_norm']
    
    X_train = df_train_balanced[features]
    y_train = df_train_balanced["corrected_accepted_fraud_amount"]
    
    X_test = df_test[features]
    y_test = df_test["corrected_accepted_fraud_amount"]
    
    # Hyperparameter tuning
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
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    
    # Visualization
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
    
    # Model evaluation
    def penalized_mae_function(y_true, y_pred):
        """Custom penalized MAE function"""
        return np.mean(np.abs(y_true - y_pred))
    
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
    
    return best_rf, y_pred, test_dates


def run_complete_pipeline():
    """
    Execute the complete fraud detection pipeline from data loading to model evaluation.
    
    Returns:
        tuple: (model, predictions, final_data) - Final model, predictions, and processed data
    """
    print("Starting Fraud Detection Pipeline...")
    
    # 1. Initialize Spark session
    print("1. Initializing Spark session...")
    spark = initialize_spark_session()
    
    # 2. Load data
    print("2. Loading data...")
    training, oos, card_status_full = load_data(spark)
    
    # Combine training and out-of-sample for feature engineering
    df = training.union(oos)
    
    # 3. Create target variables
    print("3. Creating target variables...")
    df = create_target_variable(df, training)
    
    # 4. Create time features
    print("4. Creating time features...")
    df = create_time_features(df)
    
    # 5. Create rolling and lag features
    print("5. Creating rolling and lag features...")
    df, rolling_stats_7, rolling_stats_14, lag1_df, lag7_df, lag14_df = create_rolling_features(df)
    
    # 6. Create merchant features
    print("6. Creating merchant features...")
    df = create_merchant_features(df, training)
    
    # 7. Create card features
    print("7. Creating card features...")
    df = create_card_features(df, card_status_full)
    
    # 8. Daily aggregation
    print("8. Aggregating features daily...")
    daily_data = aggregate_daily_features(df)
    
    # 9. Convert to Pandas and merge
    print("9. Converting to Pandas and merging features...")
    df_pd = convert_to_pandas(daily_data, rolling_stats_7, rolling_stats_14, lag1_df, lag7_df, lag14_df)
    
    # 10. Create corrected target
    print("10. Creating corrected target variable...")
    df_pd = create_corrected_target(df_pd, training)
    
    # 11. Prepare modeling data
    print("11. Preparing modeling data...")
    train_data, test_data = prepare_modeling_data(df_pd)
    
    # 12. Feature selection analysis
    print("12. Performing feature selection analysis...")
    mi_df = feature_selection_analysis(train_data)
    
    # 13. Train and evaluate model
    print("13. Training and evaluating model...")
    model, predictions, test_dates = train_and_evaluate_model(train_data, test_data)
    
    print("Pipeline completed successfully!")
    
    return model, predictions, df_pd


# Execute the pipeline if this script is run directly
if __name__ == "__main__":
    model, predictions, final_data = run_complete_pipeline()
