# 遍历列表
magicians = ['alice', 'david', 'carolina']
for magician in magicians:
    print(magician)
# 使用函数range()生成数值
for value in range(1, 5):
    print(value)
numbers = list(range(1, 6))
print(numbers)
numbers = list(range(1, 11, 2))
squares = []
for value in range(1, 11):
    square = value**2
    squares.append(square)
print(squares)
# 列表解析
squares = [value**2 for value in range(1, 5)]
print(squares)
players = ['charles', 'martina', 'michael', 'florence', 'eli']
print(players)
print(players[1:3])
print(players[1:])
print(players[::2])
# 赋值给列表
first = [1, 3]
second = first
second.append(4)
first.append(2)
print(first)
print(second)
first = [1, 3]
second = first[:]
second.append(4)
first.append(2)
print(first)
print(second)
# 元组
dimentions = (200,)  # 想要改变元组的值需要给这个元组重新赋值，不能直接更改
