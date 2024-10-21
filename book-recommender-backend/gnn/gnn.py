import torch
from torch import tensor
import os
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# torch-2.4.0+cu121
def test():
    print("test")