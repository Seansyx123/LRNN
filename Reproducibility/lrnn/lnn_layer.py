"""
lnn_layer.py

Logic-Neural Network (LNN) layer.

This layer integrates rule-based knowledge into neural representations
by computing rule violation penalties and adjusting feature values
accordingly.
"""

import numpy as np


class LNNLayer:
    """
    LNNLayer integrates logic rules as soft constraints.

    Each rule contributes a penalty when violated.
    The final output features are adjusted based on the aggregated
    rule violation penalties.
    """

    def __init__(self, penalty_weight=1.0):
        """
        Initialize the LNN layer.

        Args:
            penalty_weight (float): Weight controlling the influence
                                    of rule penalties.
        """
        self.penalty_weight = penalty_weight

    def _rule_violation(self, feature_vector, rule):
        """
        Compute the violation score of a single rule.

        Args:
            feature_vector (np.ndarray): Feature vector of one instance.
            rule (callable): A rule function returning 0 (satisfied)
                             or a positive value (violation degree).

        Returns:
            float: Rule violation score.
        """
        try:
            return max(0.0, rule(feature_vector))
        except Exception:
            # If rule evaluation fails, treat as no violation
            return 0.0

    def apply_penalty(self, features, rules):
        """
        Apply logic-rule penalties to input features.

        Args:
            features (np.ndarray): Input feature matrix of shape (N, D).
            rules (list): A list of rule functions.

        Returns:
            np.ndarray: Penalized feature matrix.
        """
        if rules is None or len(rules) == 0:
            return features

        adjusted_features = features.copy()

        for i in range(features.shape[0]):
            total_penalty = 0.0
            for rule in rules:
                total_penalty += self._rule_violation(
                    features[i], rule
                )

            # Soft constraint: penalize features proportionally
            adjusted_features[i] -= (
                self.penalty_weight * total_penalty
            )

        return adjusted_features
