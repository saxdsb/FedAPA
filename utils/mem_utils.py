import math
import gc
from collections import defaultdict
from typing import Optional, Tuple, List
from collections import OrderedDict
import torch
  
from math import isnan
from calmsize import size as calmsize
import sys
def readable_size(num_bytes: int) -> str:
    return '' if isnan(num_bytes) else '{:.2f}'.format(calmsize(num_bytes))

LEN = 79

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2 ** 9
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE = 2 ** 20

class MemReporter():
    """A memory reporter that collects tensors and memory usages

    Parameters:
        - model: an extra nn.Module can be passed to infer the name
        of Tensors

    """
    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.tensor_name = {}
        self.device_mapping = defaultdict(list)
        self.device_tensor_stat = {}
        # to numbering the unknown tensors
        self.name_idx = 0
        self.total_upload_size = 0
        self.total_download_size = 0

        tensor_names = defaultdict(list)
        if model is not None:
            assert isinstance(model, torch.nn.Module)
            # for model with tying weight, multiple parameters may share
            # the same underlying tensor
            for name, param in model.named_parameters():
                tensor_names[param].append(name)

        for param, name in tensor_names.items():
            self.tensor_name[id(param)] = '+'.join(name)

    def _get_tensor_name(self, tensor: torch.Tensor) -> str:
        tensor_id = id(tensor)
        if tensor_id in self.tensor_name:
            name = self.tensor_name[tensor_id]
        # use numbering if no name can be inferred
        else:
            name = type(tensor).__name__ + str(self.name_idx)
            self.tensor_name[tensor_id] = name
            self.name_idx += 1
        return name

    def collect_tensor(self):
        """Collect all tensor objects tracked by python

        NOTICE:
            - the buffers for backward which is implemented in C++ are
            not tracked by python's reference counting.
            - the gradients(.grad) of Parameters is not collected, and
            I don't know why.
        """
        #FIXME: make the grad tensor collected by gc
        objects = gc.get_objects()
        tensors = [obj for obj in objects if isinstance(obj, torch.Tensor)]
        for t in tensors:
            self.device_mapping[t.device].append(t)

    def get_stats(self):
        """Get the memory stat of tensors and then release them

        As a memory profiler, we cannot hold the reference to any tensors, which
        causes possibly inaccurate memory usage stats, so we delete the tensors after
        getting required stats"""
        visited_data = {}
        self.device_tensor_stat.clear()

        def get_tensor_stat(tensor: torch.Tensor) -> List[Tuple[str, int, int, int]]:
            """Get the stat of a single tensor

            Returns:
                - stat: a tuple containing (tensor_name, tensor_size,
            tensor_numel, tensor_memory)
            """
            assert isinstance(tensor, torch.Tensor)

            name = self._get_tensor_name(tensor)
            if tensor.is_sparse:
                indices_stat = get_tensor_stat(tensor._indices())
                values_stat = get_tensor_stat(tensor._values())
                return indices_stat + values_stat

            numel = tensor.numel()
            element_size = tensor.element_size()
            fact_numel = tensor.storage().size()
            fact_memory_size = fact_numel * element_size
            # since pytorch allocate at least 512 Bytes for any tensor, round
            # up to a multiple of 512
            memory_size = math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) \
                    * PYTORCH_MIN_ALLOCATE

            # tensor.storage should be the actual object related to memory
            # allocation
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                name = '{}(->{})'.format(
                    name,
                    visited_data[data_ptr],
                )
                # don't count the memory for reusing same underlying storage
                memory_size = 0
            else:
                visited_data[data_ptr] = name

            size = tuple(tensor.size())
            # torch scalar has empty size
            if not size:
                size = (1,)

            return [(name, size, numel, memory_size)]

        for device, tensors in self.device_mapping.items():
            tensor_stats = []
            for tensor in tensors:

                if tensor.numel() == 0:
                    continue
                stat = get_tensor_stat(tensor)  # (name, shape, numel, memory_size)
                tensor_stats += stat
                if isinstance(tensor, torch.nn.Parameter):
                    if tensor.grad is not None:
                        # manually specify the name of gradient tensor
                        self.tensor_name[id(tensor.grad)] = '{}.grad'.format(
                            self._get_tensor_name(tensor)
                        )
                        stat = get_tensor_stat(tensor.grad)
                        tensor_stats += stat

            self.device_tensor_stat[device] = tensor_stats

        self.device_mapping.clear()

    def print_stats(self, verbose: bool = False, target_device: Optional[torch.device] = None) -> None:
        # header
        # show_reuse = verbose
        # template_format = '{:<40s}{:>20s}{:>10s}'
        # print(template_format.format('Element type', 'Size', 'Used MEM') )
        for device, tensor_stats in self.device_tensor_stat.items():
            # By default, if the target_device is not specified,
            # print tensors on all devices
            if target_device is not None and device != target_device:
                continue
            # print('-' * LEN)
            print('\nStorage on {}'.format(device))
            total_mem = 0
            total_numel = 0
            for stat in tensor_stats:
                name, size, numel, mem = stat
                # if not show_reuse:
                #     name = name.split('(')[0]
                # print(template_format.format(
                #     str(name),
                #     str(size),
                #     readable_size(mem),
                # ))
                total_mem += mem
                total_numel += numel

            print('-'*LEN)
            print('Total Tensors: {} \tUsed Memory: {}'.format(
                total_numel, readable_size(total_mem),
            ))

            if device != torch.device('cpu'):
                with torch.cuda.device(device):
                    memory_allocated = torch.cuda.memory_allocated()
                print('The allocated memory on {}: {}'.format(
                    device, readable_size(memory_allocated),
                ))
                if memory_allocated != total_mem:
                    print('Memory differs due to the matrix alignment or'
                          ' invisible gradient buffer tensors')
            print('-'*LEN)

    def report(self, verbose: bool = False, device: Optional[torch.device] = None) -> None:
        """Interface for end-users to directly print the memory usage

        args:
            - verbose: flag to show tensor.storage reuse information
            - device: `torch.device` object, specify the target device
            to report detailed memory usage. It will print memory usage
            on all devices if not specified. Usually we only want to
            print the memory usage on CUDA devices.

        """
        self.collect_tensor()
        self.get_stats()
        self.print_stats(verbose, target_device=device)

    def track_upload(self, tensor_or_stat):
        """Track the size of the tensor being uploaded (e.g., gradients or model parameters)."""

        if isinstance(tensor_or_stat, torch.Tensor):
            upload_size = tensor_or_stat.numel() * tensor_or_stat.element_size()
        elif isinstance(tensor_or_stat, int):
            upload_size = sys.getsizeof(tensor_or_stat)  # 如果是int，直接使用它
        elif isinstance(tensor_or_stat, list):
            # 如果是 list，遍历每个 tensor 并计算总大小
            upload_size = sum(tensor.numel() * tensor.element_size() for tensor in tensor_or_stat if isinstance(tensor, torch.Tensor))
        elif isinstance(tensor_or_stat, torch.nn.Module):
            # 如果是模型（torch.nn.Module），遍历模型的所有参数计算大小
            upload_size = sum(param.numel() * param.element_size() for param in tensor_or_stat.parameters())
        elif isinstance(tensor_or_stat, OrderedDict):
            # 如果是 OrderedDict，遍历其中的每个值，并计算上传大小
            upload_size = sum(value.numel() * value.element_size() for value in tensor_or_stat.values() if
                              isinstance(value, torch.Tensor))
        elif isinstance(tensor_or_stat, dict):
            upload_size=0
            # 如果是 OrderedDict，遍历其中的每个值，并计算上传大小
            # upload_size = sum(value.numel() * value.element_size() for value in tensor_or_stat.values() if
            #                   isinstance(value, torch.Tensor))
            for key, value in tensor_or_stat.items():
                if isinstance(value, list):
                    upload_size += sum(tensor.numel() * tensor.element_size() for tensor in value if
                                        isinstance(tensor, torch.Tensor))
                elif isinstance(value, torch.Tensor):
                    upload_size += value.numel() * value.element_size()

        self.total_upload_size += upload_size
        print(f"Upload size: {self.readable_size(upload_size)}")

    def track_download(self, tensor_or_stat):
        """Track the size of the tensor being downloaded (e.g., global model)."""
        if isinstance(tensor_or_stat, torch.Tensor):
            download_size = tensor_or_stat.numel() * tensor_or_stat.element_size()
        elif isinstance(tensor_or_stat, int):
            download_size = sys.getsizeof(tensor_or_stat)  # 如果是int，直接使用它
        elif isinstance(tensor_or_stat, list):
            # 如果是 list，遍历每个 tensor 并计算总大小
            download_size = sum(tensor.numel() * tensor.element_size() for tensor in tensor_or_stat if isinstance(tensor, torch.Tensor))
        elif isinstance(tensor_or_stat, torch.nn.Module):
            # 如果是模型（torch.nn.Module），遍历模型的所有参数计算大小
            download_size = sum(param.numel() * param.element_size() for param in tensor_or_stat.parameters())
        elif isinstance(tensor_or_stat, OrderedDict):
            # 如果是 OrderedDict，遍历其中的每个值，并计算上传大小
            download_size = sum(value.numel() * value.element_size() for value in tensor_or_stat.values() if
                              isinstance(value, torch.Tensor))
        elif isinstance(tensor_or_stat, dict):
            download_size=0
            # 如果是 OrderedDict，遍历其中的每个值，并计算上传大小
            # download_size = sum(value.numel() * value.element_size() for value in tensor_or_stat.values() if
            #                   isinstance(value, torch.Tensor))
            for key, value in tensor_or_stat.items():
                if isinstance(value, list):
                    # 如果字典中的值是 list，则递归处理
                    download_size+= sum(tensor.numel() * tensor.element_size() for tensor in value if
                                        isinstance(tensor, torch.Tensor))
                elif isinstance(value, torch.Tensor):
                    download_size+= value.numel() * value.element_size()

        # download_size = tensor.numel() * tensor.element_size()
        self.total_download_size += download_size
        print(f"Download size: {self.readable_size(download_size)}")

    def print_communication_stats(self) ->int:
        print(f"Total upload size: {self.readable_size(self.total_upload_size)}")
        print(f"Total download size: {self.readable_size(self.total_download_size)}")
        return self.total_upload_size,self.total_download_size

    def readable_size(self, num_bytes: int) -> str:
        """Convert bytes to a readable format."""
        if num_bytes < 1024:
            return f"{num_bytes} bytes"
        elif num_bytes < 1024 ** 2:
            return f"{num_bytes / 1024:.2f} KB"
        elif num_bytes < 1024 ** 3:
            return f"{num_bytes / (1024 ** 2):.2f} MB"
        else:
            return f"{num_bytes / (1024 ** 3):.2f} GB"