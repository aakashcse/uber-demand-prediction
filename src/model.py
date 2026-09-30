"""Small helpers shared by training and the app, so both prepare inputs identically."""
import numpy as np
import pandas as pd

from src.features import FEATURES


def prepare_X(df, zone_categories):
    """Select model features and mark zone_id as a categorical column."""
    X = df[FEATURES].copy()
    X["zone_id"] = pd.Categorical(X["zone_id"], categories=zone_categories)
    return X


def predict(bundle, df):
    """Predict pickups for the rows in df (never negative)."""
    X = prepare_X(df, bundle["zone_categories"])
    return np.clip(bundle["model"].predict(X), 0, None)
