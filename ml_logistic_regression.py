from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithLBFGS
from pyspark import SparkContext, RDD

import os
from consts import MODEL_PATH, NUM_CLASSES, LABEL_INDEX, NUMERICAL_DATA_INDEX, VALUE_TO_LABEL


def create_or_load_model(sc: SparkContext, train_dataset_path: str) -> LogisticRegressionModel:
    if not os.path.exists(MODEL_PATH):
        print('training model...')
        dataset_rdd = sc.textFile(train_dataset_path)
        table_rdd = dataset_rdd.map(lambda line: line.split(','))
        labeled_features = rdd_to_feature(table_rdd)
        # labeled_features.foreach(lambda lp: print(lp))
        labeled_features.cache()
        model = LogisticRegressionWithLBFGS.train(labeled_features,
                                                  numClasses=NUM_CLASSES)
        model.setThreshold(0.5)
        model.save(sc, MODEL_PATH)
        return model
    else:
        model = LogisticRegressionModel.load(sc, MODEL_PATH)
        return model


def rdd_to_feature(rdd: RDD) -> RDD:
    return rdd.map(lambda row:
                   LabeledPoint(VALUE_TO_LABEL[row[LABEL_INDEX]],
                                [float(feature) for feature in row[NUMERICAL_DATA_INDEX:]]))


