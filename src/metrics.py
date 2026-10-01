import numpy as np

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''
    Compute precision and recall at k for each user.
    predictions: list of tuples (user_id, item_id, true_r, est_r)
    '''
    user_est_true = {}
    for uid, _, true_r, est in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0.0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0.0

    mean_precision = float(np.mean(list(precisions.values())))
    mean_recall = float(np.mean(list(recalls.values())))
    return mean_precision, mean_recall

if __name__ == "__main__":
    dummy_preds = [
        (1, 101, 4.0, 4.2),
        (1, 102, 2.0, 2.5),
        (2, 101, 5.0, 4.8),
        (2, 103, 3.0, 3.1),
    ]
    y_t = [p[2] for p in dummy_preds]
    y_p = [p[3] for p in dummy_preds]
    print(f"Test RMSE: {compute_rmse(y_t, y_p):.4f}")
    print(f"Test MAE: {compute_mae(y_t, y_p):.4f}")
    p_k, r_k = precision_recall_at_k(dummy_preds, k=2, threshold=3.5)
    print(f"Precision@2: {p_k:.4f}, Recall@2: {r_k:.4f}")
