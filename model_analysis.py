# Author - Muhammad Usman - 19/10/2022

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


# Reading csv files
def read_csv():
    ground_truth = pd.read_csv('ground_truth.csv')
    document_entity = pd.read_csv('document_entity.csv')

    # Extracting date and removing datetime from the processed_at column
    document_entity['processed_at'] = pd.to_datetime(document_entity['processed_at']).dt.date

    # Converting data type of processed_at column from object to datetime64
    document_entity['processed_at'] = pd.to_datetime(document_entity['processed_at'])

    return ground_truth, document_entity


# Pivot document entity dataframe to look exactly like ground truth table.
def pivot(document_entity):
    document_entity_pivot = document_entity.pivot(index=['document_id', 'processed_at'],
                                                  columns='label', values='prediction_value')

    # Resetting index so that document and processed_at can be used as column
    document_entity_pivot = document_entity_pivot.reset_index()

    # Converting datatype of column total_gross from object to float
    document_entity_pivot['totals_gross'] = document_entity_pivot['totals_gross'].astype(np.float64)
    return document_entity_pivot


# Join the Ground truth table and Document entity table on base of document_id
def joining_dataframe(document_entity_pivot, ground_truth):
    return document_entity_pivot.merge(ground_truth, on=['document_id'])


# this function will find the overall accuracy of all entity and STP
def find_overall_accuracy(merged, entity):
    return round(merged[entity].sum() / merged[entity].size * 100, 2)


# gives 1 if value matches otherwise zero. This function will create the drive (flagged) columns for all entity
def create_entity_match_flag(merged, column_x, column_y):
    return np.where(merged[column_x] == merged[column_y], 1, 0)


def create_resultant_dataframe(merged):
    resultant = merged[
        ['processed_at', 'document_id', 'currency_flag', 'issued_at_flag', 'totals_gross_flag', 'vendor_id_flag',
         'stp_flag']].copy()
    return resultant


def create_analysis_dataframe(resultant):
    analysis = resultant.groupby([pd.Grouper(key='processed_at', freq='W-MON')]).agg(
        document_volume=('document_id', 'count'),
        currency_count=('currency_flag', 'sum'),
        issued_at_count=('issued_at_flag', 'sum'),
        totals_gross_count=('totals_gross_flag', 'sum'),
        vendor_id_count=('vendor_id_flag', 'sum'),
        stp_count=('stp_flag', 'sum')).round(2)
    return analysis


# Calculate the percentage of the entity and STP
def cal_accuracy_percentage(analysis, column_x, column_y):
    return round(analysis[column_x] / analysis[column_y] * 100, 2)


if __name__ == '__main__':
    ground_truth_dataframe, document_entity_dataframe = read_csv()

    # Pivot document entity column based on label.
    document_entity_dataframe_pivot = pivot(document_entity_dataframe)

    # Join the Ground truth table and Document entity table on base of document_id
    merged_dataframe = joining_dataframe(document_entity_dataframe_pivot, ground_truth_dataframe)

    # created derived column as flag column for all entity and STP
    merged_dataframe['currency_flag'] = create_entity_match_flag(merged_dataframe, 'currency_x', 'currency_y')
    merged_dataframe['issued_at_flag'] = create_entity_match_flag(merged_dataframe, 'issued_at_x', 'issued_at_y')
    merged_dataframe['totals_gross_flag'] = create_entity_match_flag(merged_dataframe, 'totals_gross_x',
                                                                     'totals_gross_y')
    merged_dataframe['vendor_id_flag'] = create_entity_match_flag(merged_dataframe, 'vendor_vendor_id_x',
                                                                  'vendor_vendor_id_y')

    # Condition for STP
    merged_dataframe['stp_flag'] = np.where(
        ((merged_dataframe['currency_x'] == merged_dataframe['currency_y']) | (merged_dataframe['currency_x'].isna())) &
        ((merged_dataframe['issued_at_x'] == merged_dataframe['issued_at_y']) | (
            merged_dataframe['currency_x'].isna())) &
        ((merged_dataframe['totals_gross_x'] == merged_dataframe['totals_gross_y']) | (
            merged_dataframe['currency_x'].isna())) &
        ((merged_dataframe['vendor_vendor_id_x'] == merged_dataframe['vendor_vendor_id_y']) | (
            merged_dataframe['currency_x'].isna())), 1, 0)

    resultant_dataframe = create_resultant_dataframe(merged_dataframe)

    analysis_dataframe = create_analysis_dataframe(resultant_dataframe)

    analysis_dataframe['currency_perc'] = cal_accuracy_percentage(analysis_dataframe, 'currency_count',
                                                                  'document_volume')
    analysis_dataframe['issued_at_perc'] = cal_accuracy_percentage(analysis_dataframe, 'issued_at_count',
                                                                   'document_volume')
    analysis_dataframe['totals_gross_perc'] = cal_accuracy_percentage(analysis_dataframe, 'totals_gross_count',
                                                                      'document_volume')
    analysis_dataframe['vendor_id_perc'] = cal_accuracy_percentage(analysis_dataframe, 'vendor_id_count',
                                                                   'document_volume')
    analysis_dataframe['stp_perc'] = cal_accuracy_percentage(analysis_dataframe, 'stp_count', 'document_volume')

    # Currency Entity Accuracy
    currency_accuracy = find_overall_accuracy(merged_dataframe, 'currency_flag')
    print("The Currency Entity Accuracy is: ", currency_accuracy)

    # issued_at Entity Accuracy
    issued_at_accuracy = find_overall_accuracy(merged_dataframe, 'issued_at_flag')
    print("The issued_at Entity Accuracy is: ", issued_at_accuracy)
    #
    # totals_gross Entity Accuracy
    totals_gross_accuracy = find_overall_accuracy(merged_dataframe, 'totals_gross_flag')
    print("The totals_gross Entity Accuracy is: ", totals_gross_accuracy)
    #
    # vendor_id Entity Accuracy
    vendor_id_accuracy = find_overall_accuracy(merged_dataframe, 'vendor_id_flag')
    print("The vendor_id Entity Accuracy is: ", vendor_id_accuracy)

    # STP
    STP = find_overall_accuracy(merged_dataframe, 'stp_flag')
    print("The STP of the Model is: ", STP)