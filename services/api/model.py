def fraud_score(features: dict):
    """
    Simple baseline fraud logic
    """

    txn_count = int(features.get("txn_count_1min", 0))
    txn_sum = float(features.get("txn_sum_1min", 0))

    score = 0

    # Rule 1: Too many transactions
    if txn_count > 5:
        score += 0.5

    # Rule 2: High spending
    if txn_sum > 3000:
        score += 0.5

    # Final decision
    is_fraud = score >= 0.5

    return {
        "score": score,
        "is_fraud": is_fraud
    }