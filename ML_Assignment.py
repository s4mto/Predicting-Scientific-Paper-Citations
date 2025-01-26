import pandas as pd
import numpy as np
import logging
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def main():
    # Path to the dataset folder
    train = pd.DataFrame.from_records(json.load(open('train.json')))
    test = pd.DataFrame.from_records(json.load(open('test.json')))


    # Feature Engineering for train
    # Number of authors (count how many authors are in the list)
    train['num_authors'] = train['authors'].apply(lambda x: len(x.split(',')))

    # Number of references
    train['num_references'] = train['references'].apply(lambda x: len(x))

    # Age of the paper
    train['paper_age'] = 2024 - train['year']

    # Title word count
    train['title_word_count'] = train['title'].apply(lambda x: len(x.split()))

    # Encode 'venue' using LabelEncoder
    venue_encoder = LabelEncoder()
    train['venue_encoded'] = venue_encoder.fit_transform(train['venue'])


    # Feature Engineering for test
    # Number of authors (count how many authors are in the list)
    test['num_authors'] = test['authors'].apply(lambda x: len(x.split(',')))

    # Number of references
    test['num_references'] = test['references'].apply(lambda x: len(x))

    # Age of the paper
    test['paper_age'] = 2024 - test['year']

    # Title word count
    test['title_word_count'] = test['title'].apply(lambda x: len(x.split()))

    # Handle encoding for 'venue' in test
    # Create a mapping from the fitted LabelEncoder
    venue_mapping = {venue: idx for idx, venue in enumerate(venue_encoder.classes_)}

    # Function to encode 'venue' in the test set
    def encode_venue(venue):
        return venue_mapping.get(venue, -1)  # Map unseen labels to -1

    # Apply encoding to the test set
    test['venue_encoded'] = test['venue'].apply(encode_venue)

    
    # Split the dataset into train and validation sets
    train, validation = train_test_split(train, test_size=1/3, random_state=123)
    

    # Define the feature transformation pipeline
    featurizer = ColumnTransformer(
        transformers=[
        ("year", 'passthrough', ["year"]),  # Pass numerical features directly
        ("num_authors", 'passthrough', ["num_authors"]),  # Pass numerical features directly
        ("num_references", 'passthrough', ["num_references"]),  # Pass numerical features directly
        ("paper_age", 'passthrough', ["paper_age"]),  # Pass numerical features directly
        ("title_word_count", 'passthrough', ["title_word_count"]),  # Pass numerical features directly
        ("venue_encoded", 'passthrough', ["venue_encoded"]),  # Pass numerical features directly
        ("authors", TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), "authors"),  # Text feature
        ("abstract", TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000), "abstract"),  # Text feature
        ("title", TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=3000), "title"),  # Text feature
        ("venue", TfidfVectorizer(analyzer='word', ngram_range=(1, 1)), "venue"),  # Text feature
    ],
    remainder='drop'  # Drop any columns not explicitly transformed
        )
    
    # Define models
    ridge = make_pipeline(featurizer, Ridge(alpha=1))
    gradient = make_pipeline(featurizer, GradientBoostingRegressor(random_state=123))  
    lgbm_model = make_pipeline(featurizer, LGBMRegressor(random_state=123))  
    scaled_featurizer = make_pipeline(featurizer, MaxAbsScaler())
    xgb_model = make_pipeline(scaled_featurizer, XGBRegressor(
        objective='reg:squarederror',
        random_state=123  
    ))

    
    # Define the label
    label = 'n_citation'
    
    # GridSearch for GradientBoostingRegressor
    param_grid_gradient = {
        'gradientboostingregressor__n_estimators': [100, 200],
        'gradientboostingregressor__learning_rate': [0.05, 0.1],
        'gradientboostingregressor__max_depth': [3, 5],
    }
    grid_search_gradient = GridSearchCV(gradient, param_grid_gradient, cv=3, scoring='neg_mean_absolute_error')
    grid_search_gradient.fit(train.drop([label], axis=1), np.log1p(train[label].values))

    # RandomizedSearch for LightGBM
    param_grid_lgbm = {
        "lgbmregressor__n_estimators": [100, 200],
        "lgbmregressor__learning_rate": [0.05, 0.1],
        "lgbmregressor__max_depth": [3, 5],
        "lgbmregressor__num_leaves": [15, 31]
    }
    random_search_lgbm = RandomizedSearchCV(
        lgbm_model,
        param_distributions=param_grid_lgbm,
        n_iter=10,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    random_search_lgbm.fit(train.drop([label], axis=1), np.log1p(train[label].values))

    # RandomizedSearch for XGBRegressor
    param_grid_xgb = {
        "xgbregressor__n_estimators": [100, 200],
        "xgbregressor__learning_rate": [0.05, 0.1],
        "xgbregressor__max_depth": [3, 5],
        "xgbregressor__subsample": [0.8, 1.0],
        "xgbregressor__colsample_bytree": [0.8, 1.0]
    }
    random_search_xgb = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid_xgb,
        n_iter=10,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    random_search_xgb.fit(train.drop([label], axis=1), np.log1p(train[label].values))

    # Evaluate XGB model
    xgb_scores = cross_val_score(random_search_xgb.best_estimator_, train.drop([label], axis=1), np.log1p(train[label].values), cv=10, scoring='neg_mean_absolute_error')
    logging.info(f"XGBoost Regression cross-validation MAE scores: {-xgb_scores}")
    logging.info(f"XGBoost Regression average MAE: {-np.mean(xgb_scores):.2f}")


    # Evaluate models and choose the best one
    validation_scores = {}
    for model_name, model in [
        ("ridge", ridge),
        ("GradientBoosting", grid_search_gradient.best_estimator_),
        ("LightGBM", random_search_lgbm.best_estimator_),
        ("XGB", random_search_xgb.best_estimator_),
    ]:
        validation_pred = np.expm1(model.predict(validation.drop([label], axis=1)))
        validation_mae = mean_absolute_error(validation[label], validation_pred)
        validation_scores[model_name] = validation_mae
        logging.info(f"{model_name} validation MAE: {validation_mae:.2f}")

    # Select the best model based on MAE
    best_model_name = min(validation_scores, key=validation_scores.get)
    best_model = {
        "ridge": ridge,
        "GradientBoosting": grid_search_gradient.best_estimator_,
        "LightGBM": random_search_lgbm.best_estimator_,
        "XGB": random_search_xgb.best_estimator_,
    }[best_model_name]
    logging.info(f"Best model is {best_model_name} with MAE: {validation_scores[best_model_name]:.2f}")

    # Generate predictions for the best model
    test['n_citation'] = np.expm1(best_model.predict(test))

    # Save the predictions to a JSON file
    json.dump(test[['n_citation']].to_dict(orient='records'), open(f'predicted_{best_model_name}.json', 'w'),indent=2)
    logging.info(f"Predictions for the best model '{best_model_name}' saved to 'predicted_{best_model_name}.json'")

# Set logging level
logging.getLogger().setLevel(logging.INFO)

# Run the main function
main()