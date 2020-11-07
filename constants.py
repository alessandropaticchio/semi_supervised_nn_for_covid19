import torch
import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float

data_start = datetime.strptime('22/1/2020', '%d/%m/%Y')

green = '#009B07'
orange = '#FE9441'
red = '#D61919'
blue = '#0B6AD4'
purple = '#4B0082'
dark_orange = '#FF4E00'
magenta = '#FF0062'
ticksize = 22
labelsize = 25
legendsize = 25
mylinewidth = 15