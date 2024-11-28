# package needed to install
#pip install datasets
#pip install -U transformers
#pip install transformers peft torch
#pip install --upgrade peft
# gym
#pip install gymnasium
#pip install omegaconf
#pip show mujoco
#pip install --upgrade gymnasium[mujoco]

import gymnasium as gym

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import pandas as pd
from torchvision import transforms
import glob
from tqdm import tqdm
from urllib.request import urlopen
from PIL import Image
import json
import collections
from torch.utils.data import DataLoader
from IPython.display import Video

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

import time
import random

#custome lib
import First_stage
import Second_stage
import Third_stage
import Visualization

from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

#seed 고정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    First_stage.reinforcement_learning()
    Second_stage.transformer_build()
    Third_stage.transformer_rl_build()
    Visualization.visualization()



