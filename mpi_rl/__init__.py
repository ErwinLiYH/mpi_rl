from . import config

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        if MPI.COMM_WORLD.Get_rank() == 0:
            # print basic information about the MPI environment
            print("MPI environment detected.")
            print(f"Running in MPI mode with {MPI.COMM_WORLD.Get_size()} processes.")
            # print thread information
            print(f"Thread support: {MPI.Query_thread()}")
            config.use_mpi = True
    else:
        config.use_mpi = False
        print("Running in single process mode, but MPI is available.")
except ImportError:
    config.use_mpi = False
    print("Running in single process mode, MPI is not available.")        
