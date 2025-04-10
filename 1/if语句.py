cars = ['audi', 'bmw', 'subaru', 'toyota']
for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())
# 使用and检查多个条件
age_0 = 1
age_1 = 2
if age_0 == 1 and age_1 != 5:
    print(True)
# 使用or检查多个条件
if age_1 == 5 or age_0 == 1:
    print(True)
# 检查特定值是否包含在列表中
print('audi' in cars)
print('bmw' not in cars)
age = 12
if age < 4:
    print("free")
elif age < 18:
    print("cost $25")
else:
    print("cost $40")
# 判断列表为空
tops = []
if not tops:
    print(tops)
cars = ['audi', 'bmw', 'subaru', 'toyota']
for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())
# 使用and检查多个条件
age_0 = 1
age_1 = 2
if age_0 == 1 and age_1 != 5:
    print(True)
# 使用or检查多个条件
if age_1 == 5 or age_0 == 1:
    print(True)
