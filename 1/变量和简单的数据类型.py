name = "zhengShuai shan"
# 将字符串首字母大写，其他字符都小写
print(name.title())
# 字符串转大写/小写
print(name.upper())
print(name.lower())
# python默认换行输出，想要不换行输出需要这么做：
print(name, end="")
print(name, end="\n\n")
# 在字符串中使用变量的值：f字符串
firstname = "zhengshuai"
lastname = "shan"
myname = f"{firstname} {lastname}"
print(f"{ myname.title() },你好！")
py = " python "
print(py+'123')
# 输出时删除左边空白
print(py.lstrip()+'123')
# 输出时删除右边的空白
print(py.rstrip()+'123')
# 输出时删除两边的空白
print(py.strip()+'123')
print(py+'123')
# 永久删除可以这样干：
py = py.strip()
print(py + "123")
# 用数中的下划线来给数字分组
universe_age = 14_000_000_000
print(universe_age)
# 同时给多个变量赋值
x, y, z = 0, 1, 2
print(f"{x}+{y}+{z}")
