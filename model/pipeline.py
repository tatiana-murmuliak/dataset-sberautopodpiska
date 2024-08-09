import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from datetime import datetime


def balance_data(df):
    import pandas as pd

    df = df.copy()

    df_1 = df[df['target_action'] == 1]
    df_0 = df[df['target_action'] == 0].iloc[:50000]
    df_balanced = pd.concat([df_1, df_0], axis=0).sample(frac=1).reset_index(drop=True)

    return df_balanced


def edit_geodata(df):
    df = df.copy()

    most_popular_cities = [
        'Moscow', 'Saint Petersburg', 'Yekaterinburg',
        'Krasnodar', 'Kazan', 'Samara', 'Nizhny Novgorod',
        'Ufa', 'Novosibirsk', 'Krasnoyarsk'
    ]

    df['geo_city'] = df.geo_city.apply(lambda x: x if x in most_popular_cities else '(not set)')

    return df


def edit_datetime(df):
    import pandas as pd

    df = df.copy()

    df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S')
    df['visit_time'] = df['visit_time'].dt.hour

    df['visit_date'] = pd.to_datetime(df['visit_date'], format='%Y-%m-%d')
    df['visit_month'] = df['visit_date'].dt.month
    df['visit_dayofweek'] = df['visit_date'].dt.dayofweek

    return df


def filter_data(df):
    columns_to_drop = [
        'session_id',
        'client_id',
        'device_model',
        'utm_keyword',
        'device_os',
        'utm_adcontent',
        'utm_campaign',
        'device_screen_resolution',
        'geo_country',
        'visit_date'
    ]

    return df.drop(columns_to_drop, axis=1)


def main():
    print('Target Action Prediction Pipeline')

    df_merged = pd.read_csv('data/ga_merged.csv')
    df_balanced = balance_data(df_merged)

    X = df_balanced.drop('target_action', axis=1)
    y = df_balanced['target_action']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('geodata_editor', FunctionTransformer(edit_geodata)),
        ('datetime_editor', FunctionTransformer(edit_datetime)),
        ('filter', FunctionTransformer(filter_data)),
        ('column_transformer', column_transformer)
    ])

    models = (
        LogisticRegression(penalty='l2', solver='newton-cg', max_iter=400),
        RandomForestClassifier(max_depth=20, min_samples_leaf=2,
                               min_samples_split=10, n_estimators=900),
        MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,20), activation='tanh')
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('sber_auto_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target action prediction model',
                'author': 'Tatiana Murmuliak',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


