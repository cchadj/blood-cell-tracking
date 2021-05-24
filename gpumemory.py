import torch


def print_memory_cuda_stats():
    t = torch.cuda.get_device_properties(0).total_memory
    print('Total memory', t)
    c = torch.cuda.memory_cached(0)
    print('Cached memory', c)
    a = torch.cuda.memory_allocated(0)
    print('Allocated memory', a)
    f = c-a  # free inside cache
    print('Available memory', f)


if __name__ == '__main__':
    print_memory_cuda_stats()

