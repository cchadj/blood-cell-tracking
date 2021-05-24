import torch


def test_cuda():
    print("Is cuda available?: ", torch.cuda.is_available())
    print("Cuda device: ", torch.cuda.current_device())
    print("Cuda device count: ", torch.cuda.device_count())
    print("Cuda device name: ", torch.cuda.get_device_name())


if __name__ == "__main__":
    test_cuda()

