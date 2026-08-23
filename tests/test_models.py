"""
Unit Tests for Core Neural Learning Models and Datasets
"""

import unittest
import numpy as np
import pandas as pd
from core.datasets import generate_dataset, process_uploaded_dataframe, AVAILABLE_DATASETS
from core.models import (
    PerceptronModel,
    AdalineModel,
    HebbianModel,
    LogisticRegressionModel,
    MLPModel,
    CompetitiveLearningModel,
    AVAILABLE_MODELS
)


class TestNeuralModels(unittest.TestCase):

    def test_dataset_generation_all(self):
        for dataset_name in AVAILABLE_DATASETS:
            with self.subTest(dataset=dataset_name):
                X, y = generate_dataset(dataset_name, n_samples=60, noise=0.1, random_state=42)
                self.assertEqual(X.shape, (60, 2))
                self.assertEqual(y.shape, (60,))
                self.assertTrue(set(np.unique(y)).issubset({-1.0, 1.0}))
                self.assertFalse(np.isnan(X).any())
                self.assertFalse(np.isnan(y).any())

    def test_process_uploaded_dataframe(self):
        df = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "f3": np.random.randn(50),
            "label": ["cat"] * 25 + ["dog"] * 25
        })
        X, y, feats, err = process_uploaded_dataframe(df, target_column="label")
        self.assertIsNone(err)
        self.assertEqual(X.shape, (50, 2))
        self.assertEqual(y.shape, (50,))
        self.assertEqual(len(feats), 2)

    def test_perceptron_convergence_on_separable(self):
        X, y = generate_dataset("Linearly Separable", n_samples=80, noise=0.0, random_state=42)
        model = PerceptronModel()
        model.fit(X, y, epochs=40, lr=0.1)
        
        preds = model.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)
        self.assertEqual(len(model.loss_history), 40)
        self.assertEqual(len(model.weight_history), 40)
        self.assertIn("weights", model.export_weights())

    def test_adaline_lms(self):
        X, y = generate_dataset("Linearly Separable", n_samples=80, noise=0.0, random_state=42)
        model = AdalineModel()
        model.fit(X, y, epochs=40, lr=0.05)
        
        preds = model.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.85)
        self.assertLess(model.loss_history[-1], model.loss_history[0])

    def test_hebbian_learning(self):
        X, y = generate_dataset("Linearly Separable", n_samples=60, noise=0.0, random_state=42)
        model = HebbianModel(mode="Normalized (Oja)")
        model.fit(X, y, epochs=30, lr=0.05)
        
        preds = model.predict(X)
        self.assertEqual(len(preds), 60)
        self.assertEqual(len(model.weight_history), 30)

    def test_logistic_regression(self):
        X, y = generate_dataset("Linearly Separable", n_samples=80, noise=0.0, random_state=42)
        model = LogisticRegressionModel()
        model.fit(X, y, epochs=50, lr=0.2)
        
        preds = model.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)
        self.assertLess(model.loss_history[-1], model.loss_history[0])

    def test_mlp_xor_solution(self):
        X, y = generate_dataset("XOR Problem", n_samples=100, noise=0.0, random_state=42)
        model = MLPModel(hidden_dim=8, activation="tanh")
        model.fit(X, y, epochs=120, lr=0.3, random_state=42)
        
        preds = model.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.80)
        self.assertEqual(len(model.w_snapshots), 120)

    def test_competitive_learning(self):
        X, _ = generate_dataset("Linearly Separable", n_samples=80, noise=0.0, random_state=42)
        model = CompetitiveLearningModel(n_clusters=2)
        model.fit(X, epochs=30, lr=0.2, random_state=42)
        
        preds = model.predict(X)
        self.assertEqual(len(preds), 80)
        self.assertTrue(set(np.unique(preds)).issubset({-1, 1}))
        self.assertEqual(len(model.prototype_history), 30)


if __name__ == "__main__":
    unittest.main()
