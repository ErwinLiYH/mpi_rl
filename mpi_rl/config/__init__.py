_use_mpi = True  # Whether running in MPI multi-rank env, should not be changed directly
gpu_awareness = False

def get_use_mpi():
    return _use_mpi