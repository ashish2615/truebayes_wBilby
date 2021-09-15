import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim


def makenet(dims, softmax=True, single=True):

  """Make a fully connected Deep Neural Network with layer widths described by `dims`.

  CUDA is always enabled, and double precision is set with `single=False`.
  The output layer applies a softmax transformation, disabled by setting `softmax=False`."""
  print('dimesions are', dims)
  ndims = len(dims)
  print('ndims is ',ndims)

  ## Base class for all neural network modules.
  ## A kind of Tensor that is to be considered a module parameter.
  ## Modules can also contain other Modules, allowing to nest them in a tree structure.

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # the weights must be set explicitly as attributes in the class
      # (i.e., we can't collect them in a single list)
      for l in range(ndims - 1):
        ## Applies a linear transformation to the incoming data: y = xA^T + b
        layer = nn.Linear(dims[l], dims[l+1])
        print('layer',layer)
        
        if not single:
          layer = layer.double()
        
        if torch.cuda.is_available():
          print('cuda is available')
          layer = layer.cuda()
        
        setattr(self, f'fc{l}', layer)
                
    def forward(self, x):
      # per Alvin's recipe, apply relu everywhere but last layer
      for l in range(ndims - 2):
        ## LeakyRelu gives x if x > 0 otherwise it will give negative_slope * x.
        x = F.leaky_relu(getattr(self, f'fc{l}')(x), negative_slope=0.2)

      x = getattr(self, f'fc{ndims - 2}')(x)

      if softmax:
        return F.softmax(x, dim=1)
      else:
        return x
  
  return Net


def makenetbn(dims, softmax=True, single=True):
  """A batch-normalizing version of makenet. Experimental."""

  ndims = len(dims)

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # the weights must be set explicitly as attributes in the class
      # (i.e., we can't collect them in a single list)
      for l in range(ndims - 1):
        layer = nn.Linear(dims[l], dims[l+1])
        bn = nn.BatchNorm1d(num_features=dims[l+1])
        
        if not single:
          layer = layer.double()
          bn = bn.double()
        
        if torch.cuda.is_available():
          print('cuda is available')
          layer = layer.cuda()
          bn = bn.cuda()
        
        setattr(self, f'fc{l}', layer)
        setattr(self, f'bn{l}', bn)
                
    def forward(self, x):
      # per Alvin's recipe, apply relu everywhere but last layer
      for l in range(ndims - 2):
        x = getattr(self, f'bn{l}')(F.leaky_relu(getattr(self, f'fc{l}')(x), negative_slope=0.2))

      x = getattr(self, f'fc{ndims - 2}')(x)

      if softmax:
        return F.softmax(x, dim=1)
      else:
        return x
  
  return Net
