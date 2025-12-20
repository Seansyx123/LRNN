
"""
Generate logic rules for LRNN using RuleKit.
"""
import argparse
import pandas as pd
from rulekit.classification import RuleClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training CSV file")
    parser.add_argument("--label", default="Defective", help="Label column name")
    parser.add_argument("--out", required=True, help="Output rules file")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.label])
    y = df[args.label]

    clf = RuleClassifier(
        min_rule_coverage=5,
        max_rule_length=3,
        pruning_enabled=True
    )
    clf.fit(X, y)

    with open(args.out, "w") as f:
        for i, rule in enumerate(clf.model_.rules, 1):
            f.write(f"Rule_{i}: {rule}\n")

if __name__ == "__main__":
    main()
