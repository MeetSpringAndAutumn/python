import numpy as np

# 计算欧几里得距离


def euclidean_distance(x, y):

    # 计算点差
    diff = x - y

    # 计算平方和
    square_sum = np.sum(diff ** 2, axis=1)

    # 计算欧几里得距离
    distance = np.sqrt(square_sum)

    return distance

# 对距离进行排序


def sort_distance(distance):

    # 使用 argsort 函数获取排序后的索引
    sorted_idx = np.argsort(distance)

    # 使用 sorted_idx 对距离进行排序
    sorted_distance = distance[sorted_idx]

    return sorted_distance


# 测试代码
x = np.array([[1, 3]])
y = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])

distance = euclidean_distance(x, y)
sorted_distance = sort_distance(distance)

print("欧几里得距离：", distance)
print("排序后的欧几里得距离：", sorted_distance)
