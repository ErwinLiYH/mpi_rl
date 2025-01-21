import timeit

episode = [{"name": "Alice"}] * 10**6
episode_id = 1001

# 方法 1: 使用 map
time_map = timeit.timeit(lambda: list(map(lambda x: dict(x, episode_id=episode_id), episode)), number=10)

# 方法 2: 使用列表推导式
time_list_comp = timeit.timeit(lambda: [{**x, "episode_id": episode_id} for x in episode], number=10)

print(f"map + dict() 耗时: {time_map:.6f} 秒")
print(f"列表推导式 耗时: {time_list_comp:.6f} 秒")