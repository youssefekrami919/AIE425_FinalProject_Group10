# ============================================================================ 
# Group: 19
# Team Members:
# - Youssef Ekrami Elsayed 
# - Abdelrahman Mohamed Negm
# - Hagar Mohamed Badawy
# - Dareen Ashraf Mosa
# ============================================================================





# ============================================================================ 
# HELPER FUNCTION: Construct User Profile
# ============================================================================
def build_user_profile(user_id, user_item_df, item_features, item_ids, avg_profile):
    user_data = user_item_df[user_item_df['user_id'] == user_id]
    if user_data.empty:
        return avg_profile
    ratings = user_data['rating'].values.reshape(-1, 1)
    indices = [np.where(item_ids == iid)[0][0] for iid in user_data['item_id']]
    user_profile = (item_features[indices].T @ ratings).flatten() / ratings.sum()
    return user_profile

# ============================================================================ 
# HELPER FUNCTION: Content-Based Top-N Recommendation
# ============================================================================
def content_based_recommendation(user_id, user_profiles, item_features, item_ids, item_info_dict, top_n=10):
    user_vec = user_profiles.get(user_id, item_features.mean(axis=0).A1).reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, item_features).flatten()
    rated_items = set(df[df['user_id'] == user_id]['item_id'].values)
    items_scores = [(iid, score) for iid, score in zip(item_ids, sim_scores) if iid not in rated_items]
    
    # Keep unique items sorted by score
    seen_items = set()
    unique_items = []
    for iid, score in sorted(items_scores, key=lambda x: x[1], reverse=True):
        if iid not in seen_items:
            seen_items.add(iid)
            unique_items.append((iid, score))
        if len(unique_items) >= top_n:
            break

    recommendations = []
    for iid, score in unique_items[:top_n]:
        info = item_info_dict[iid]
        recommendations.append({
            'user_id': user_id,
            'item_id': iid,
            'score': score,
            'title': info['title'],
            'primary_topic': info['primary_topic'],
            'subtopic': info['subtopic'],
            'difficulty': info['difficulty'],
            'content_type': info['content_type']
        })
    return recommendations

# ============================================================================ 
# HELPER FUNCTION: Item-Based k-NN Prediction
# ============================================================================
def knn_predict_rating(user_id, iid, user_item_df, knn_results, avg_rating=0):
    user_data = user_item_df[user_item_df['user_id'] == user_id]
    if iid in user_data['item_id'].values:
        return None  # skip already rated
    sim_items = knn_results[20][iid]  # using k=20
    weighted_sum, sim_sum = 0, 0
    for sim_iid, sim_score in sim_items:
        row = user_data[user_data['item_id'] == sim_iid]
        if not row.empty:
            weighted_sum += row['rating'].values[0] * sim_score
            sim_sum += sim_score
    return weighted_sum / sim_sum if sim_sum > 0 else avg_rating






# ============================================================================ 
# HELPER FUNCTION: Item-Based Collaborative Filtering Recommendation
# ============================================================================
def item_based_recommendation(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    scores = {}

    for item in rated_items:
        similar_items = item_similarity_df[item]

        for sim_item, sim_score in similar_items.items():
            if sim_item not in rated_items:
                if sim_item not in scores:
                    scores[sim_item] = 0
                scores[sim_item] += sim_score * user_ratings[item]

    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return scores[:top_n]

# ============================================================================ 
# HELPER FUNCTION: SVD-Based Recommendation
# ============================================================================
def svd_recommendation(user_id, top_n=5):
    if user_id not in predicted_df.index:
        return []

    user_predictions = predicted_df.loc[user_id]
    user_actual = user_item_matrix.loc[user_id]

    # Recommend items not yet rated
    recommendations = user_predictions[user_actual.isna()]

    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(top_n)




