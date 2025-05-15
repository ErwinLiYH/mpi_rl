from mpi4py import MPI

def allocate_list_resource_by_num(L:list, num_each:int, comm:MPI.Comm)->list:
    """
    Allocate a list resource to each process in MPI, each process gets num_each elements.
    if len(L) > comm_size * num_each, the rest elements are discarded.
    if len(L) < comm_size * num_each, the rest processes get empty list or resource num is less than num_each.
    """
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    L0 = L[:comm_size * num_each]
    L0 = [L0[i:i + num_each] for i in range(0, len(L0), num_each)]
    if rank < len(L0):
        return L0[rank]
    else:
        raise Exception(f"Resource is out of range, {len(L)} resource, {comm_size} ranks requests {num_each} each.")
    
def run_on_one_rank(comm:MPI.Comm, rank:int, func, *args, **kwargs) -> None:
    """
    Run a function on one rank, and broadcast the result to all ranks.
    """
    rank = comm.Get_rank()
    if rank == rank:
        func(*args, **kwargs)
    else:
        return