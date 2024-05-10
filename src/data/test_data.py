import pandas as pd
import os
import sys
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues, \
  TestNumberOfConstantColumns, TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType, \
  TestNumberOfDriftedColumns

current_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(current_dir, '..', '..', 'reports')
data_dir = os.path.join(current_dir, '..', '..', 'data')
merged_data_path = os.path.join(data_dir, 'merged')

current_dataset = pd.read_csv(os.path.join(merged_data_path, 'current_data.csv'))
reference_dataset = pd.read_csv(os.path.join(merged_data_path, 'reference_data.csv'))

current_dataset['datetime'] = pd.to_datetime(current_dataset['datetime'])
reference_dataset['datetime'] = pd.to_datetime(reference_dataset['datetime'])

# Run stability tests
tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference_dataset, current_data=current_dataset)
test_results = tests.as_dict()

# Check if any test failed
if test_results['summary']['failed_tests'] > 0:
    print("Some tests failed:")
    print(test_results['summary']['failed_tests'])
    sys.exit(1)
else:
    print("All tests passed!")


# Run data drift analysis
report = Report(metrics=[
    DataDriftPreset(), 
])

report.run(reference_data=reference_dataset, current_data=current_dataset)
report.save_html(os.path.join(reports_dir, 'figures', "DataDrift_report.html"))
print(f"Data drift report successfully saved!")