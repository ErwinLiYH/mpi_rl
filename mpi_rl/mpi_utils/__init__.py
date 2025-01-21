from mpi4py import MPI

# for type hint only
from mpi_rl.utils.replay_buffer import Episode


def mpi_broadcast_model_params():
    pass

def mpi_gather_episodes_to_main(
        rank: int,
        comm: MPI.Comm,
        role: str,
        episode: Episode,
        ):
    pass