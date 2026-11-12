# DATA CARD – QUESTION 2

## Dataset Name
Rwanda Road Traffic Accident Severity Synthetic Dataset

## Scenario
Road traffic accident severity preparation in Rwanda.

## Stakeholder
Rwanda Road Safety and Traffic Management Unit.

## Purpose
The dataset is prepared for future classification of simulated accident severity based on road, weather, vehicle and accident characteristics.

## Data Source
Group-generated synthetic dataset using a reproducible Python generator.

## Dataset Size
1,608 records before cleaning and duplicate removal.

## Target
`severity`

Classes:
- Minor
- Serious
- Fatal

## Features
District, road type, weather, road surface, lighting, time of day, vehicle type, speed limit, vehicles involved, road users affected and response time.

## Data Preparation
The group investigated missing values, duplicates, inconsistent categories, invalid values and outliers.

Preprocessing included imputation, encoding, scaling, feature engineering, feature selection and splitting.

## Limitations
All records and target labels are simulated. The dataset does not represent official Rwanda accident statistics and is unsuitable for real-world operational decisions.

## Reproducibility
The dataset generator uses a fixed random seed. The notebooks and preprocessing code are included in the project repository.

## Ethical Considerations
No personal identifiers are included. The dataset is synthetic and must not be represented as genuine accident records.

## Intended Use
Educational machine-learning dataset preparation.
