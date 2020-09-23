from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession, DataFrame
import os
import datetime
from consts import (RAW_DATA_DIR, TRAIN_DATASET_PATH, TEST_DATASET_PATH,
                    CSV_SEPARATOR, SELECTED_FEATURES, NUMERICAL_DATA_INDEX, TEST_SET_SIZE_PERCENTAGE,
                    TRAIN_SET_SIZE_PERCENTAGE)


def main():
    data_set = None
    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith('.csv'):
            print(file_name)
            raw_data_path = os.path.join(RAW_DATA_DIR, file_name)
            df = prepare_data(raw_data_path)
            if data_set:
                data_set = data_set.union(df)
            else:
                data_set = df
    # Split dataset to training set and test set
    data_sets = data_set.randomSplit([TRAIN_SET_SIZE_PERCENTAGE, TEST_SET_SIZE_PERCENTAGE],
                                     datetime.datetime.now().microsecond)
    data_sets[0].write.csv(TRAIN_DATASET_PATH, sep=CSV_SEPARATOR, encoding='utf-8')
    data_sets[1].write.csv(TEST_DATASET_PATH, sep=CSV_SEPARATOR, encoding='utf-8')


def normalize_df(df: DataFrame, cols):
    df.show(5)
    unlist = udf(lambda x: round(float(list(x)[0]), 2), DoubleType())
    for i in cols:
        # VectorAssembler Transformation - Converting column to vector type
        assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")
        # MinMaxScaler Transformation
        scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")
        # Pipeline of VectorAssembler and MinMaxScaler
        pipeline = Pipeline(stages=[assembler, scaler])

        # Fitting pipeline on dataframe
        df = pipeline.fit(df).transform(df).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect").drop(i)
    return df


def prepare_data(raw_data_file) -> DataFrame:
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(raw_data_file, header=True, mode='DROPMALFORMED', inferSchema=True)
    selected_features_df = df.select(SELECTED_FEATURES)
    normalized_df = normalize_df(selected_features_df, SELECTED_FEATURES[NUMERICAL_DATA_INDEX:])
    normalized_df.show()
    return normalized_df


if __name__ == '__main__':
    main()
