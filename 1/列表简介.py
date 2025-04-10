bicycles = ["trek", 'cannondale', 'redline', 'specialized']
print(bicycles)
print(bicycles[0])
print(bicycles[-1])
# 修改列表元素
bicycles[0] = 'ducati'
print(bicycles[0])
# 列表尾部添加元素
bicycles.append('aito')
print(bicycles)
# 列表插入元素
bicycles.insert(1, 'byd')
print(bicycles)
# 从列表中删除元素
del bicycles[0]
print(bicycles)
lastCar = bicycles.pop()
print(bicycles)
print(lastCar)
theDeleteCar = bicycles.pop(1)
print(bicycles)
print(theDeleteCar)
bicycles.remove('byd')  # remove只删除第一个指定的值
# 使用sort对列表永久排序
bicycles = ["trek", 'cannondale', 'redline', 'specialized']
print(bicycles)
bicycles.sort()
print(bicycles)
bicycles.sort(reverse=True)  # 按照与字母顺序相反的顺序排列
# 使用sorted对列表进行临时排序
bicycles = ["trek", 'cannondale', 'redline', 'specialized']
print(bicycles)
print(sorted(bicycles))
print(sorted(bicycles, reverse=True))
# 反转列表元素
bicycles.reverse()
print(bicycles)
# 获取列表的长度
print(len(bicycles))
