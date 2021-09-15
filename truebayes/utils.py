import torch

def numpy2cuda(array, single=True):
  ## Create a tensor from numpy array
  array = torch.from_numpy(array)
  
  if single:
    array = array.float()

  ## Returns a bool indicating if CUDA is currently available
  if torch.cuda.is_available():
    print('cuda is available')
    # map_location = torch.device('cpu')
    array = array.cuda()
    
  return array


def cuda2numpy(tensor):
  ## creates a tensor that shares storage with tensor that does not require grad.
  ## detach() = cut computational graph.
  ## cpu() =allocate tensor in RAM
  ## numpy() = port tensor to numpy
  return tensor.detach().cpu().numpy()
