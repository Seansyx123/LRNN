"""
rule_utils.py

Utilities for loading and parsing logic rules.
"""

import re


def load_rules(rule_path):
    """
    Load rule functions from a rule definition file.

    Args:
        rule_path (str): Path to rule text file.

    Returns:
        list: A list of callable rule functions.
    """
    with open(rule_path, "r") as f:
        lines = f.readlines()

    return parse_rules(lines)


def parse_rules(lines):
    """
    Parse rule definitions into executable rule functions.

    Args:
        lines (list): Lines from rule file.

    Returns:
        list: List of rule functions.
    """
    rules = []
    current_conditions = []

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("IF"):
            current_conditions = extract_conditions(line)

        elif line.startswith("THEN"):
            rule_fn = build_rule_function(current_conditions)
            rules.append(rule_fn)
            current_conditions = []

    return rules


def extract_conditions(if_line):
    """
    Extract atomic conditions from an IF clause.

    Example:
        IF (LOC > 300) AND (Cyclomatic_Complexity > 10)

    Returns:
        list of (feature, operator, threshold)
    """
    condition_pattern = r"\((.*?)\)"
    matches = re.findall(condition_pattern, if_line)

    conditions = []
    for m in matches:
        tokens = m.split()
        if len(tokens) == 3:
            feature, op, value = tokens
            conditions.append(
                (feature, op, float(value))
            )

    return conditions


def build_rule_function(conditions):
    """
    Build a rule violation function from atomic conditions.

    Args:
        conditions (list): List of (feature, operator, threshold)

    Returns:
        function: rule(feature_vector) -> violation score
    """

    def rule(feature_vector):
        violation = 0.0
        for idx, (feature, op, threshold) in enumerate(conditions):
            # Feature index mapping assumed to be external
            try:
                value = feature_vector[idx]
            except IndexError:
                continue

            if op == ">":
                if value <= threshold:
                    violation += (threshold - value)
            elif op == "<":
                if value >= threshold:
                    violation += (value - threshold)
            elif op == ">=":
                if value < threshold:
                    violation += (threshold - value)
            elif op == "<=":
                if value > threshold:
                    violation += (value - threshold)

        return violation

    return rule
