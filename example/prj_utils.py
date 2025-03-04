import os

def path_allocator(base_namepath: str, linker:str = "_", is_directory: bool = True, start:int = 1) -> str:
    """
    generate a new directory {base_namepath}{linker}{i} where i is the smallest integer that makes the directory name unique.
    if is_directory is True, create the directory and return the path, else just return the path.
    """
    i = start
    while True:
        path = f"{base_namepath}{linker}{i}"
        if not os.path.exists(path):
            if is_directory:
                os.makedirs(path)
            return path
        else:
            i += 1
