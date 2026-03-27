import datetime
import os
from warnings import simplefilter
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from dataload import load_data
import random
import scipy.io as sio
from args import parameter_parser
from train import train
simplefilter(action='ignore', category=FutureWarning)


args = parameter_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.enabled = True


train(args,device)

