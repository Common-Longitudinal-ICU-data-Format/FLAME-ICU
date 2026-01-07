"""
FLAIR Benchmark Models

XGBoost and ElasticNet models for tasks 5, 6, 7.
"""

from code.models.xgboost_model import XGBoostModel
from code.models.elasticnet_model import ElasticNetModel
from code.models.evaluation import TaskEvaluator

__all__ = ["XGBoostModel", "ElasticNetModel", "TaskEvaluator"]
