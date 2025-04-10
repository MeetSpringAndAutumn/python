import json
import math

import numpy as np
from scipy.stats import pearsonr

# def load_data(json_data):
#     return json.loads(json_data)
import os


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")

    with open(filepath, 'r', encoding='utf-8') as file:
        json_data = file.read()

    if not json_data.strip():  # 检查文件是否为空
        raise ValueError("File is empty")

    return json.loads(json_data)


import math
from scipy.stats import pearsonr

def sort_movie_ratings_desc(data):
    sorted_data = {user: dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True))
                   for user, ratings in data.items()}
    return sorted_data

import numpy as np

def euclidean_distance(user1_ratings, user2_ratings):
    common_movies = set(user1_ratings.keys()).intersection(set(user2_ratings.keys()))
    if not common_movies:
        return float('inf')  # No common movies, distance is infinite

    squared_differences = []
    for movie in common_movies:
        squared_differences.append(np.square(user1_ratings[movie] - user2_ratings[movie]))
    return 1 / (1 + np.sqrt(np.sum(squared_differences)))



def pearson_correlation(user1_ratings, user2_ratings):
    # 找到共同评分的项目
    common_movies = set(user1_ratings.keys()).intersection(set(user2_ratings.keys()))
    if len(common_movies) < 2:
        return 0  # 如果共同评分项少于2，返回0表示无效相关性

    # 提取共同评分项的评分
    user1_scores = [user1_ratings[movie] for movie in common_movies]
    user2_scores = [user2_ratings[movie] for movie in common_movies]

    # 检查是否为常量数组
    if np.all(np.array(user1_scores) == user1_scores[0]) or np.all(np.array(user2_scores) == user2_scores[0]):
        return 0  # 如果数组是常量数组，返回0表示无效相关性

    # 计算并返回 Pearson 相关系数
    correlation, _ = pearsonr(user1_scores, user2_scores)
    return correlation

def find_most_similar_users(data, target_user, k=1, method='euclidean'):
    distances = []
    for user in data:
        if user != target_user:
            if method == 'euclidean':
                distance = euclidean_distance(data[target_user], data[user])
            elif method == 'pearson':
                distance = pearson_correlation(data[target_user], data[user])
            distances.append((user, distance))

    if method == 'euclidean':
        # For Euclidean distance, the smaller the distance, the more similar the users
        distances.sort(key=lambda x: x[1])
    elif method == 'pearson':
        # For Pearson correlation, the larger the correlation, the more similar the users
        distances.sort(key=lambda x: x[1], reverse=True)

    # Return the top k most similar users
    return distances[:k]


def recommend_movies(data, target_user, k, method='euclidean', num_recommendations=2):
    # print(k)
    similar_users = find_most_similar_users(data, target_user, k, method)
    # print(similar_users)
    similar_user_names = [user for user, _ in similar_users]

    movie_recommendations = {}
    for user in similar_user_names:
        for movie, rating in data[user].items():
            if movie not in data[target_user]:
                if movie not in movie_recommendations:
                    movie_recommendations[movie] = []
                movie_recommendations[movie].append(rating)

    for movie in movie_recommendations:
        movie_recommendations[movie] = np.mean(movie_recommendations[movie])

    recommended_movies = sorted(movie_recommendations.items(), key=lambda x: x[1], reverse=True)

    return [name for name,_ in recommended_movies[:num_recommendations]]


def print_similiar_users(data, target_user, k=2,method='euclidean'):
    similar_users = find_most_similar_users(data, target_user, k,method)
    # 取出前 k 个最相似用户的名称
    top_k_similar_users = [user for user, _ in similar_users[:k]]
    print(f"{target_user}的最相似的两个用户：{top_k_similar_users}")


# Example usage
if __name__ == "__main__":
    # 测试加载 JSON 数据
    try:
        data = load_data('movie_ratings.json')
        # print(data)
    except Exception as e:
        print(f"An error occurred: {e}")
    data=sort_movie_ratings_desc(data)
    # print(data)
    user1 = 'John Carson'
    user2 = 'Michael Henry'

    # Calculate Euclidean distance
    euclidean_dist = euclidean_distance(data[user1], data[user2])
    if euclidean_dist != float('inf'):
        print(f"Euclidean Distance between {user1} and {user2}: {euclidean_dist}")
        print_similiar_users(data, user1)
        print_similiar_users(data, user2)
        print()
    # Calculate Pearson correlation
    pearson_corr = pearson_correlation(data[user1], data[user2])
    if pearson_corr != 0:
        print(f"Pearson Correlation between {user1} and {user2}: {pearson_corr}")
        print_similiar_users(data, user1,method='pearson')
        print_similiar_users(data, user2,method='pearson')
        print()

    # Generate movie recommendations
    target_user = user2
    recommendations = recommend_movies(data, target_user,k=len(data),method='euclidean')
    print(f"Recommendations for {target_user} using Euclidean method:")
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{i}: {recommendation}")
