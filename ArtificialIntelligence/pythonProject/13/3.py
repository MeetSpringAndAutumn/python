# import pandas as pd
#
import os

import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr


# 读取 CSV 数据
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df


from scipy.stats import pearsonr


def sort_movie_ratings_desc(data):
    sorted_data = {user: dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True))
                   for user, ratings in data.items()}
    return sorted_data



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

    return [name for name, _ in recommended_movies[:num_recommendations]]


def print_similiar_users(data, target_user, k=2, method='euclidean'):
    similar_users = find_most_similar_users(data, target_user, k, method)
    # 取出前 k 个最相似用户的名称
    top_k_similar_users = [user for user, _ in similar_users[:k]]
    print(f"{target_user}的最相似的两个用户：{top_k_similar_users}")
def load_data_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")

    with open(filepath, 'r', encoding='utf-8') as file:
        json_data = file.read()

    if not json_data.strip():  # 检查文件是否为空
        raise ValueError("File is empty")

    return json.loads(json_data)


# 将 CSV 数据转换为用户-电影评分的字典格式
def convert_to_dict(df):
    user_ratings = {}
    for user_id, group in df.groupby('userId'):
        user_ratings[user_id] = group.set_index('title')['rating'].to_dict()
    return user_ratings


# 示例代码
if __name__ == "__main__":
    # # 读取movies.csv和ratings.csv文件
    # movies_df = pd.read_csv('movies.csv')
    # ratings_df = pd.read_csv('ratings.csv')
    #
    # # 合并两个DataFrame，使用movieId作为关联键
    # merged_df = pd.merge(ratings_df, movies_df, on='movieId')
    #
    # # 将合并后的DataFrame保存为新的CSV文件
    # merged_df.to_csv('merged_movies_ratings.csv', index=False)
    # csv_file = 'merged_movies_ratings.csv'
    # data = load_data(csv_file)
    # user_ratings = convert_to_dict(data)
    # # 将字典数据写入JSON文件
    # filename = 'data.json'
    # with open(filename, 'w', encoding='utf-8') as file:
    #     json.dump(user_ratings, file, ensure_ascii=False, indent=4)
    # print(user_ratings)
    # data=load_data_json('data.json')
    # data = sort_movie_ratings_desc(data)
    # with open(filename, 'w', encoding='utf-8') as file:
    #     json.dump(data, file, ensure_ascii=False, indent=4)

    data = load_data_json('data.json')

    user1 = "1"
    user2 = "7"

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
        print_similiar_users(data, user1, method='pearson')
        print_similiar_users(data, user2, method='pearson')
        print()

    # Generate movie recommendations
    target_user = user2
    recommendations = recommend_movies(data, target_user, k=12, method='euclidean',num_recommendations=2)
    print(f"Recommendations for {target_user} using Euclidean method:")
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{i}: {recommendation}")

