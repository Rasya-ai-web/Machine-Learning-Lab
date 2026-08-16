import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



df = pd.read_csv("student_performance_updated_1000(1).csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)



print("\nMissing Values Before Handling:")
print(df.isnull().sum())


numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


df['Online Classes Taken'] = df['Online Classes Taken'].fillna(
    df['Online Classes Taken'].mode()[0]
)

print("\nMissing Values After Handling:")
print(df.isnull().sum())



print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)


le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])


le_parental = LabelEncoder()
df['ParentalSupport'] = le_parental.fit_transform(
    df['ParentalSupport']
)


le_online = LabelEncoder()
df['Online Classes Taken'] = le_online.fit_transform(
    df['Online Classes Taken'].astype(str)
)

print("\nDataset After Encoding:")
print(df.head())



y = df['FinalGrade']

X = df.drop(columns=['FinalGrade', 'Name', 'StudentID'])

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("\nScaled Training Features:")
print(X_train)

print("\nScaled Testing Features:")
print(X_test)

print("\nTraining Target:")
print(y_train.head())

print("\nTesting Target:")
print(y_test.head())

print("\nPreprocessing Completed Successfully!")
