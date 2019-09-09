import numpy as np
import pandas as pd

# df = pd.read_csv('Ecommerce Purchases')
# df = df.head(100)
# print(df.info())
# Data columns (total 14 columns):
# Address             100 non-null object
# Lot                 100 non-null object
# AM or PM            100 non-null object
# Browser Info        100 non-null object
# Company             100 non-null object
# Credit Card         100 non-null int64
# CC Exp Date         100 non-null object
# CC Security Code    100 non-null int64
# CC Provider         100 non-null object
# Email               100 non-null object
# Job                 100 non-null object
# IP Address          100 non-null object
# Language            100 non-null object
# Purchase Price      100 non-null float64

# print(df.head())

# print(df['Email'].apply(lambda x : x.split('@')[1]).value_counts().head(5))

# print(df[df['CC Exp Date'].apply(lambda x:int(x[3::])) == 25]['Email'].count())

# print(df[(df['CC Provider'] == 'American Express') & (df['Purchase Price'] > 95)]['Email'].count())

# print(df[df['Credit Card'] == 4926535242672853]['Email'])

# print(df[df['Lot'] == '90 WT']['Purchase Price'].reset_index()['Purchase Price'].iloc[0])

# print(df['Job'].value_counts().head(5))

# print(df['AM or PM'].value_counts())

# print(df.groupby('AM or PM')['Email'].count())

# print(df[df['Job'] == 'Lawyer']['Job'].count())

# print(df[df['Language'] == 'en']['Language'].count())

# print(df['Purchase Price'].max(), df['Purchase Price'].min(), df['Purchase Price'].mean())





###########################################################################
# df = pd.read_csv('Salaries.csv', header = None)
# print(df.head())
#    Id       EmployeeName                                        JobTitle    BasePay  OvertimePay  ...  TotalPayBenefits  Year  Notes         Agency  Status
# 0   1     NATHANIEL FORD  GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY  167411.18         0.00  ...         567595.43  2011    NaN  San Francisco     NaN
# 1   2       GARY JIMENEZ                 CAPTAIN III (POLICE DEPARTMENT)  155966.02    245131.88  ...         538909.28  2011    NaN  San Francisco     NaN
# 2   3     ALBERT PARDINI                 CAPTAIN III (POLICE DEPARTMENT)  212739.13    106088.18  ...         335279.91  2011    NaN  San Francisco     NaN
# 3   4  CHRISTOPHER CHONG            WIRE ROPE CABLE MAINTENANCE MECHANIC   77916.00     56120.71  ...         332343.61  2011    NaN  San Francisco     NaN
# 4   5    PATRICK GARDNER    DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)  134401.60      9737.00  ...         326373.19  2011    NaN  San Francisco     NaN
# print(df.info())
# Data columns (total 13 columns):
# Id                  148654 non-null int64
# EmployeeName        148654 non-null object
# JobTitle            148654 non-null object
# BasePay             148045 non-null float64
# OvertimePay         148650 non-null float64
# OtherPay            148650 non-null float64
# Benefits            112491 non-null float64
# TotalPay            148654 non-null float64
# TotalPayBenefits    148654 non-null float64
# Year                148654 non-null int64
# Notes               0 non-null float64
# Agency              148654 non-null object
# Status              0 non-null float64
# dtypes: float64(8), int64(2), object(3)
# memory usage: 14.7+ MB

# df['JobTitle'] = df['JobTitle'].apply(len)
# df['length'] = df['JobTitle'].apply(len)
# print(df.head())
# print(df.drop('length', axis = 1).head())


# df.drop('length', inplace = True)
# print(df.info())
# print(df[['JobTitle', 'BasePay']].corr())

# print(df[df['JobTitle'].apply(lambda x: 'chief' in x.lower())]['JobTitle'].count())
# print(sum(df['JobTitle'].apply(lambda x: 'chief' in x.lower())))

# job_title = df[df['Year'] == 2013]['JobTitle'].reset_index()
# job_title2 = job_title['JobTitle'].value_counts().reset_index()
# print(job_title2[job_title2['JobTitle'] == 1]['index'].count())


# print(df['JobTitle'].value_counts().reset_index().sort_values(by = 'JobTitle', ascending = False).head(5))
# print(df['JobTitle'].nunique())
# print(df.groupby('Year').mean()['BasePay'].reset_index())
# print(df[df['TotalPayBenefits'] == df['TotalPayBenefits'].min()])
# print(df[df['TotalPayBenefits'] == df['TotalPayBenefits'].max()]['EmployeeName'].reset_index()['EmployeeName'].iloc[0])

# print(df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits'].reset_index()['TotalPayBenefits'].iloc[0])
# print(df[df['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'].reset_index()['JobTitle'].iloc[0])

# s = pd.Series([1,2,3], name = 'val', index = ['a','b','c'])
# print(s)
# print(s.reset_index()['val'].iloc[2])

# print(df['OvertimePay'].max())

# print(df['BasePay'].sum() / df['BasePay'].count())
# print(df['BasePay'].mean())

