from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

#
# Dataset fields
#
numeric_fields = ["if"+str(i) for i in range(1,14)]
categorical_fields = ["cf"+str(i) for i in range(1,27)] + ["day_number"]


fields = ["id", "label"] + numeric_fields + categorical_fields

#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
#numeric_features = ['CLEANLINESS', 'ROOM', 'SERVICE', 'LOCATION']
numeric_features = fields[2:15]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#categorical_features = ['city', 'country']
#так будет плохо кажется
categorical_features = fields[15:]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LinearRegression())
])




