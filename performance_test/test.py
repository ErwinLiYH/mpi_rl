import timeit

n = 500000  # 要删除的元素数量

# 方案 1: 切片赋值
time_slice = timeit.timeit(
    stmt="lst1 = lst1[n:]",
    setup="lst1 = list(range(10**6)); n = 500000",
    number=10
)

# 方案 2: del 关键字
time_del = timeit.timeit(
    stmt="del lst2[:n]",
    setup="lst2 = list(range(10**6)); n = 500000",
    number=10
)

print(f"切片赋值 (lst = lst[n:]) 耗时: {time_slice:.6f} 秒")
print(f"del (del lst[:n]) 耗时: {time_del:.6f} 秒")