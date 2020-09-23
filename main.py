from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

import ml_logistic_regression
from consts import TRAIN_DATASET_PATH, TEST_DATASET_PATH


def main():
    sc = SparkContext.getOrCreate()
    model = ml_logistic_regression.create_or_load_model(sc, TRAIN_DATASET_PATH)
    test(sc, model, TEST_DATASET_PATH)


def logged_comparison(model: LogisticRegressionModel, labeled_point: LabeledPoint):
    predication = model.predict(labeled_point.features)
    # print(f'prediction = {predication}', end=', ')
    return predication == labeled_point.label


def test(sc: SparkContext, model: LogisticRegressionModel, test_dataset_path: str):
    lines = sc.textFile(test_dataset_path)
    labeled_test_set = ml_logistic_regression.rdd_to_feature(lines.map(lambda line: line.split(',')))
    success_count = labeled_test_set.filter(lambda labeled_point: logged_comparison(model, labeled_point)).count()

    total_count = lines.count()
    print(f'Model Accuracy for {test_dataset_path}: {100 * (success_count / total_count)} %')
    print(f'from {total_count} test cases')


if __name__ == '__main__':
    main()

