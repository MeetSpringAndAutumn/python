alien_0 = {'color': 'green', 'points': 5}
print(f"you just enarned {alien_0['points']} points")
alien_0['x_position'] = 0
alien_0['y_position'] = 25
print(alien_0)
alien_0['color'] = 'yellow'
print(alien_0['color'])
del alien_0['points']
print(alien_0)
# del alien_0
# 使用get解决键不存在的问题
point_value = alien_0.get('points', 'No point value assigned')
print(point_value)
point_value = alien_0.get('points')
print(point_value)
# 遍历字典
for key, value in alien_0.items():  # 变量名可以任取
    print(f"{key}:{value}")
# 遍历键
for key in alien_0.keys():
    print(key)
for key in alien_0:
    print(key)
for key in sorted(alien_0.keys(), reverse=True):
    print(key)
favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'phil': 'python',
}
# 去除重复项
for language in set(favorite_languages.values()):
    print(language)
# 使用花括号创建集合
lan = {'chinese', 'english', 'english'}
for i in lan:
    print(i)
# 字典列表
aliens = []
for alien_number in range(30):
    new_alien = {'color': 'green', 'points': 5, 'speed': 'slow'}
    aliens.append(new_alien)
for alien in aliens[:3]:
    if alien['color'] == 'green':
        alien['points'] = 10
for alien in aliens[:5]:
    print(alien)
# 在字典中存储列表
pizza = {'toppings': ['mushrooms', 'extra cheese'], 'crust': 'thick', }
