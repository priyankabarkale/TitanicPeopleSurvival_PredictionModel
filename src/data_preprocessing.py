from config import Config
# Import Machine Learning Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler,LabelEncoder
from imblearn.over_sampling import SMOTE


def data_preprocessing(df):

    df= pd.read_csv(Config.filepath)

    df.drop(columns = ['PassengerId','CabinDeck','Ticket','Title','AgeGroup'],inplace=True,axis =1)

    ''' 
    1. Split the Dataset into X and y
    2. Split the Dataset into Train and Test Sets
    3. Use Encoding Techniques on Categorical Features
    4. Use Scaling Techniques on Numerical Features
    5. Use SMOTE
    '''
    # 1. Split the Dataset into X and y
    X = df.drop('Survived',axis=1)    
    y = df['Survived']
    # 2. Split the Dataset into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.3,
                                                        random_state = 1)
    # 3. Use Encoding Techniques on Categorical Features

    # Segregate Categorical and Numerical Features
    numerical_col = X.select_dtypes(exclude = 'object').columns
    categorical_col = X.select_dtypes(include = 'object').columns
    # Apply Label Encoding on Categorical Features
    le = LabelEncoder()
    for col in categorical_col:
        X_train[col] = le.fit_transform(X_train[col])  # Seen Data
        X_test[col] = le.transform(X_test[col])           # Unseen Data
    # 4. Use Scaling Techniques on Numerical Features
    sc = RobustScaler()
    X_train[numerical_col] = sc.fit_transform(X_train[numerical_col])   # Seen Data
    X_test[numerical_col] = sc.transform(X_test[numerical_col])   # Unseen Data

    # 5. Use SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)  # Seen Data

    return X_train,X_test,y_train,y_test