_use_mpi = True  # Whether running in MPI multi-rank env, should not be changed directly
gpu_awareness = False

eval_env_in_another_process = True
# if True, the evaluation environment will be created in another process in sequential training evaluation mode

def get_use_mpi():
    return _use_mpi