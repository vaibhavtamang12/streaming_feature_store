from feast import FeatureStore

store = None

def init_store():
    global store
    store = FeatureStore(repo_path="/app/feature_repo")


def get_features(user_id: int):
    features = store.get_online_features(
        features=[
            "user_features:txn_count_1min",
            "user_features:txn_sum_1min",
        ],
        entity_rows=[{"user_id": user_id}],
    ).to_dict()

    return {
        "txn_count_1min": features["txn_count_1min"][0],
        "txn_sum_1min": features["txn_sum_1min"][0],
    }