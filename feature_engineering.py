"""
Feature engineering functions for fraud detection pipeline
"""
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, when, sin, cos, lit, to_date, date_sub, sum as Fsum
from typing import Tuple, Dict, Any
import numpy as np
from .config import HIGH_RISK_CONFIG

def create_fraud_labels(df: SparkDataFrame) -> SparkDataFrame:
    """
    Create fraud labels and target variables
    
    Args:
        df: Input Spark DataFrame
    
    Returns:
        DataFrame with fraud labels and target variables
    """
    # Create fraud label
    df = df.withColumn(
        "fraud_label",
        F.when(F.col('fraud_sp') > 0, 1)  # Confirmed fraud
        .when(F.col('fraud_sp') <= 0, -1)  # Confirmed genuine
        .otherwise(0)  # Unknown
    )
    
    # Create fraud amount
    df = df.withColumn(
        "fraud_amount",
        when(col("fraud_label") == 1, col("OPERATION_AMOUNT")).otherwise(0)
    )
    
    # Create transaction accepted flag
    df = df.withColumn(
        "TRANSACTION_ACCEPTED",
        F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)
    )
    
    # Create target variable: fraud_amount_accepted
    df = df.withColumn(
        "fraud_amount_accepted",
        when(col("TRANSACTION_ACCEPTED") == 1, col("fraud_amount")).otherwise(0)
    )
    
    return df

def create_time_features(df: SparkDataFrame) -> SparkDataFrame:
    """
    Create time-based features
    
    Args:
        df: Input Spark DataFrame
    
    Returns:
        DataFrame with time features
    """
    # Basic time features
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
    
    # Cyclical transformations
    max_day = 31
    max_month = 12
    pi = 3.141592653589793
    
    # Day of month sin/cos
    df = df.withColumn(
        "dom_angle", 2 * lit(pi) * col("DAY_OF_MONTH") / lit(max_day)
    ).withColumn(
        "dom_sin", sin(col("dom_angle"))
    ).withColumn(
        "dom_cos", cos(col("dom_angle"))
    ).drop("dom_angle")
    
    # Month sin/cos
    df = df.withColumn(
        "month_angle", 2 * lit(pi) * col("TRANSACTION_MONTH") / lit(max_month)
    ).withColumn(
        "month_sin", sin(col("month_angle"))
    ).withColumn(
        "month_cos", cos(col("month_angle"))
    ).drop("month_angle")
    
    # Weekend indicator
    df = df.withColumn(
        "is_weekend", 
        F.when(F.dayofweek("TRANSACTION_DATE").isin(1, 7), 1).otherwise(0)
    )
    
    return df

def create_rolling_features(df: SparkDataFrame, window_days: int = 7) -> SparkDataFrame:
    """
    Create rolling statistics features
    
    Args:
        df: Input Spark DataFrame
        window_days: Number of days for rolling window
    
    Returns:
        DataFrame with rolling features
    """
    # Extract date columns
    df = df.withColumn("day", F.to_date("TRANSACTION_DATETIME")) \
           .withColumn("classification_day", F.to_date("fraud_classification_datetime"))
    
    # Select relevant columns for rolling calculations
    df_valid = df.select("day", "classification_day", "fraud_amount_accepted")
    
    # Create target days
    target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")
    
    # Cross join and filter
    joined = df_valid.crossJoin(target_days) \
        .filter(F.col("day") < F.col("target_day")) \
        .filter(F.col("classification_day") <= F.col("target_day"))
    
    # Aggregate per day
    daily_agg_per_target = joined.groupBy("target_day", "day").agg(
        F.sum("fraud_amount_accepted").alias("daily_fraud_sum")
    )
    
    # Calculate rolling statistics
    rolling_base = daily_agg_per_target \
        .withColumn("days_diff", F.datediff(F.col("target_day"), F.col("day"))) \
        .filter(F.col("days_diff").between(1, window_days))
    
    rolling_stats = rolling_base.groupBy("target_day").agg(
        F.mean("daily_fraud_sum").alias(f"rolling_mean_{window_days}"),
        F.max("daily_fraud_sum").alias(f"rolling_max_{window_days}"),
        F.min("daily_fraud_sum").alias(f"rolling_min_{window_days}"),
        F.sum("daily_fraud_sum").alias(f"rolling_sum_{window_days}"),
        F.stddev("daily_fraud_sum").alias(f"rolling_std_{window_days}")
    ).orderBy("target_day")
    
    return rolling_stats

def create_lag_features(df: SparkDataFrame, lag_days: int = 1) -> SparkDataFrame:
    """
    Create lagged features
    
    Args:
        df: Input Spark DataFrame
        lag_days: Number of days to lag
    
    Returns:
        DataFrame with lag features
    """
    # Ensure date columns
    df = df.withColumn("day", to_date("TRANSACTION_DATETIME")) \
           .withColumn("classification_day", to_date("fraud_classification_datetime"))
    
    # Get target days
    target_days = df.select("day").distinct().withColumnRenamed("day", "target_day")
    
    # Join to compute lag
    df_lag = df.crossJoin(target_days) \
        .filter(col("day") == date_sub(col("target_day"), lag_days)) \
        .filter(col("classification_day") <= col("target_day"))
    
    # Aggregate lag features
    lag_df = df_lag.groupBy("target_day").agg(
        Fsum("fraud_amount_accepted").alias(f"lag_{lag_days}")
    )
    
    return lag_df

def create_merchant_features(df: SparkDataFrame, training_df: SparkDataFrame) -> SparkDataFrame:
    """
    Create merchant-related features
    
    Args:
        df: Input Spark DataFrame
        training_df: Training DataFrame for computing risk metrics
    
    Returns:
        DataFrame with merchant features
    """
    # New merchant detection
    window_spec = Window.partitionBy("MERCHANT_NAME").orderBy("TRANSACTION_DATE")
    df = df.withColumn("first_seen_date", F.min("TRANSACTION_DATE").over(window_spec))
    df = df.withColumn("is_new_merchant", 
                      (F.col("TRANSACTION_DATE") == F.col("first_seen_date")).cast("int"))
    
    # Merchant size classification
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
    
    # MCC fraud risk ranking
    mcc_fraud_pct = training_df.groupBy("MCC").agg(
        F.round(100 * F.count(F.when(F.col('fraud_label') == 1, 1)) / F.count('*'), 4).alias('mcc_fraud_rate')
    )
    
    window_spec = Window.orderBy(F.col("mcc_fraud_rate").asc())
    ranked_mcc = mcc_fraud_pct.withColumn(
        "fraud_rank", F.dense_rank().over(window_spec)
    )
    
    df = df.join(ranked_mcc.select("MCC", "fraud_rank"), on="MCC", how="left")
    df = df.withColumnRenamed("fraud_rank", "fraud_risk_rank")
    
    # Penalized operation amount
    df = df.withColumn("penalized_operation_amount", 
                      F.col("OPERATION_AMOUNT") * F.col("fraud_risk_rank"))
    
    # High-risk merchant classification
    fraud_threshold = mcc_fraud_pct.approxQuantile("mcc_fraud_rate", [HIGH_RISK_CONFIG["fraud_threshold_percentile"]], 0.01)[0]
    high_risk_mcc = mcc_fraud_pct.withColumn(
        "is_high_risk_merchant", 
        F.when(F.col("mcc_fraud_rate") >= fraud_threshold, 1).otherwise(0)
    ).select("MCC", "is_high_risk_merchant")
    
    df = df.join(high_risk_mcc, on="MCC", how="left").fillna({"is_high_risk_merchant": 0})
    
    # Merchant name processing
    df = df.withColumn("MERCHANT_NAME_LOWER", F.lower(F.col("MERCHANT_NAME")))
    
    return df

def create_card_features(df: SparkDataFrame, card_status_df: SparkDataFrame) -> SparkDataFrame:
    """
    Create card-related features
    
    Args:
        df: Input Spark DataFrame
        card_status_df: Card status DataFrame
    
    Returns:
        DataFrame with card features
    """
    # Authorization response code features
    df = df.withColumn("number_tx_exceeded", 
                      F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 65, 1).otherwise(0))
    df = df.withColumn("cash_withdrawal_exceeded", 
                      F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 61, 1).otherwise(0))
    df = df.withColumn("card_number_invalid", 
                      F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 14, 1).otherwise(0))
    df = df.withColumn("invalid_pin", 
                      F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 55, 1).otherwise(0))
    df = df.withColumn("invalid_cvv", 
                      F.when(F.col("AUTHORISATION_RESPONSE_CDE") == 82, 1).otherwise(0))
    
    # Card cancellations
    card_status_df = card_status_df.withColumn("TRANSACTION_DATE", to_date(col("TRANSACTION_DATETIME")))
    card_cancellations = card_status_df.filter(col("NEW_CARD_STATUS").isin("06", "09"))
    daily_cancellations = card_cancellations.groupBy("TRANSACTION_DATE").count().withColumnRenamed("count", "DAILY_CANCELLED_CARD_COUNT")
    
    df = df.join(daily_cancellations, on="TRANSACTION_DATE", how="left")
    
    # Remote transaction indicator
    df = df.withColumn("is_remote", F.col("FRAUD_ACCEPTOR_CDE").startswith("REMOTE").cast("int"))
    
    return df

def create_daily_aggregation(df: SparkDataFrame) -> SparkDataFrame:
    """
    Create daily aggregated features
    
    Args:
        df: Input Spark DataFrame with all features
    
    Returns:
        DataFrame with daily aggregated features
    """
    daily_data = df.groupBy("TRANSACTION_DATE", "is_weekend").agg(
        # Merchant amounts
        F.sum(F.when(df["small_merchant"] == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("small_merchant_amount"),
        F.sum(F.when(df["big_merchant"] == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("big_merchant_amount"),
        F.sum("penalized_operation_amount").alias("penalized_total_operation_amount"),
        
        # Averages and ratios
        F.round(F.sum("penalized_operation_amount") / F.countDistinct("PAN_ENCRYPTED"), 4).alias('avg_pen_amount_per_card'),
        F.round(F.sum("penalized_operation_amount") / F.sum("OPERATION_AMOUNT"), 4).alias('avg_pen_amount_norm'),
        F.round(F.sum("OPERATION_AMOUNT") / F.countDistinct("PAN_ENCRYPTED"), 4).alias("avg_unique_card_amount"),
        F.round(F.sum("OPERATION_AMOUNT") / F.count('*'), 4).alias('amount_transaction_ratio'),
        F.round(F.count('*') / F.countDistinct("MERCHANT_NAME"), 4).alias('avg_transaction_per_merchant'),
        
        # Counts and volumes
        F.sum("DAILY_CANCELLED_CARD_COUNT").alias("total_cancellations"),
        F.count("*").alias("total_tx"),
        F.sum("is_remote").alias("remote_tx"),
        F.countDistinct("PAN_ENCRYPTED").alias("unique_cards"),
        F.countDistinct("MERCHANT_NAME").alias("unique_merchants"),
        F.countDistinct("MCC").alias("mcc_distinct_counts"),
        F.countDistinct(F.concat_ws("_", F.col("PAN_ENCRYPTED"), F.col("MERCHANT_NAME"))).alias("unique_merchant_card_pairs"),
        
        # Response statuses
        F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)).alias("total_rejected"),
        F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)).alias("total_accepted"),
        F.round(F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)) / F.countDistinct("MERCHANT_NAME"), 4).alias("avg_rejected_tx_merchant"),
        F.round(F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "RJCT", 1).otherwise(0)) / F.sum(F.when(F.col("RESPONSE_STATUS_CDE") == "ACCP", 1).otherwise(0)), 4).alias('rejected_accepted_ratio'),
        
        # Target and amounts
        F.sum("fraud_amount_accepted").alias("fraud_amount_accepted"),
        F.sum("OPERATION_AMOUNT").alias("total_operation_amount"),
        
        # Time features
        F.first("month_sin").alias("month_sin"),
        F.first("month_cos").alias("month_cos"),
        F.first("dom_cos").alias("day_of_month_cos"),
        F.first("dom_sin").alias("day_of_month_sin"),
        
        # Authorization errors
        F.sum("number_tx_exceeded").alias("number_tx_exceeded"),
        F.sum("cash_withdrawal_exceeded").alias("cash_withdrawal_exceeded"),
        F.sum("card_number_invalid").alias("card_number_invalid"),
        F.sum("invalid_pin").alias("invalid_pin"),
        F.sum("invalid_cvv").alias("invalid_cvv"),
        
        # New merchant features
        F.sum("is_new_merchant").alias("new_merchant_tx"),
        F.sum(F.when(F.col("is_new_merchant") == 1, F.col("OPERATION_AMOUNT")).otherwise(0)).alias("new_merchant_amount"),
        
        # Ratios
        F.round(F.sum("is_remote") / F.count('*'), 4).alias('remote_ratio'),
        F.round(F.sum("is_high_risk_merchant") / F.count('*'), 4).alias('high_risk_tx_ratio'),
        
        # Specific MCC amounts
        F.sum(F.when(F.col("MCC") == "6051", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_6051_amount"),
        F.sum(F.when(F.col("MCC") == "4829", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_4829_amount"),
        F.sum(F.when(F.col("MCC") == "7995", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_7995_amount"),
        F.sum(F.when(F.col("MCC") == "5999", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_5999_amount"),
        F.sum(F.when(F.col("MCC") == "4722", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_4722_amount"),
        F.sum(F.when(F.col("MCC") == "5944", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("mcc_5944_amount"),
        
        # Specific MCC counts
        F.sum(F.when(F.col("MCC") == "6051", 1).otherwise(0)).alias("mcc_6051_count"),
        F.sum(F.when(F.col("MCC") == "4829", 1).otherwise(0)).alias("mcc_4829_count"),
        
        # Specific merchants
        F.sum(F.when(F.col("MERCHANT_NAME") == "BETANO PT", F.col("OPERATION_AMOUNT")).otherwise(0)).alias("BETANO_PT_amount"),
        F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("binance.com"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("binance_amount"),
        F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("bifinity"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("bifinity_amount"),
        F.sum(F.when(F.col("MERCHANT_NAME_LOWER").contains("vinted"), F.col("OPERATION_AMOUNT")).otherwise(0)).alias("vinted_amount")
        
    ).orderBy("TRANSACTION_DATE", ascending=True)
    
    return daily_data

def create_labeling_delay_correction(training_df: SparkDataFrame, 
                                   daily_df: pd.DataFrame, 
                                   reference_date: str = '2025-06-06') -> pd.DataFrame:
    """
    Create fraud labeling delay correction
    
    Args:
        training_df: Training Spark DataFrame
        daily_df: Daily aggregated pandas DataFrame
        reference_date: Reference date for correction calculation
    
    Returns:
        DataFrame with corrected fraud amounts
    """
    # Calculate labeling delay curve
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
    
    counts_by_day = df_delay.groupBy('delay_days_rounded').agg(
        F.count('*').alias('labelled_count')
    ).orderBy('delay_days_rounded')
    
    total_transactions = df_delay.count()
    
    w = Window.orderBy('delay_days_rounded').rowsBetween(Window.unboundedPreceding, 0)
    
    counts_by_day = counts_by_day.withColumn(
        'cummulative_labelled', 
        F.sum('labelled_count').over(w)
    ).withColumn(
        'pct_labelled',
        (F.col('cummulative_labelled') / F.lit(total_transactions)) * 100
    )
    
    label_curve = counts_by_day.select(
        'delay_days_rounded', 'pct_labelled'
    ).filter('delay_days_rounded >= 0').toPandas()
    
    # Apply correction to daily data
    labeling_curve_dict = dict(zip(label_curve['delay_days_rounded'], (label_curve['pct_labelled']) / 100))
    
    dataset_given_date = pd.Timestamp(reference_date)
    daily_df['days_since_transaction'] = (dataset_given_date - pd.to_datetime(daily_df['TRANSACTION_DATE'])).dt.days
    
    # Cap days
    max_known_day = max(labeling_curve_dict.keys())
    daily_df['days_since_transaction'] = daily_df['days_since_transaction'].clip(upper=max_known_day)
    
    # Apply correction
    daily_df['correction_factor'] = daily_df['days_since_transaction'].map(labeling_curve_dict)
    daily_df['corrected_accepted_fraud_amount'] = (daily_df['fraud_amount_accepted'] / daily_df['correction_factor'])
    
    return daily_df

def generate_all_features(df: SparkDataFrame, 
                         training_df: SparkDataFrame,
                         card_status_df: SparkDataFrame,
                         apply_correction: bool = True) -> pd.DataFrame:
    """
    Generate all features for the fraud detection pipeline
    
    Args:
        df: Input Spark DataFrame
        training_df: Training DataFrame for risk calculations
        card_status_df: Card status DataFrame
        apply_correction: Whether to apply labeling delay correction
    
    Returns:
        Pandas DataFrame with all features
    """
    # Step 1: Create fraud labels and target
    df = create_fraud_labels(df)
    
    # Step 2: Create time features
    df = create_time_features(df)
    
    # Step 3: Create merchant features
    df = create_merchant_features(df, training_df)
    
    # Step 4: Create card features
    df = create_card_features(df, card_status_df)
    
    # Step 5: Create daily aggregation
    daily_data = create_daily_aggregation(df)
    
    # Step 6: Create rolling and lag features
    rolling_7 = create_rolling_features(df, 7)
    rolling_14 = create_rolling_features(df, 14)
    lag_1 = create_lag_features(df, 1)
    lag_7 = create_lag_features(df, 7)
    lag_14 = create_lag_features(df, 14)
    
    # Convert to pandas for merging
    daily_pd = daily_data.toPandas()
    rolling_7_pd = rolling_7.toPandas()
    rolling_14_pd = rolling_14.toPandas()
    lag_1_pd = lag_1.toPandas()
    lag_7_pd = lag_7.toPandas()
    lag_14_pd = lag_14.toPandas()
    
    # Merge all features
    final_df = daily_pd.merge(rolling_7_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    final_df = final_df.merge(rolling_14_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    final_df = final_df.merge(lag_1_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    final_df = final_df.merge(lag_7_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    final_df = final_df.merge(lag_14_pd, left_on="TRANSACTION_DATE", right_on="target_day", how="left").drop(columns=["target_day"])
    
    # Fill missing values
    final_df = final_df.bfill().fillna(0)
    final_df = final_df.sort_values("TRANSACTION_DATE", ascending=True).reset_index(drop=True)
    
    # Step 7: Apply labeling delay correction if requested
    if apply_correction:
        final_df = create_labeling_delay_correction(training_df, final_df)
    
    return final_df 