import pandas as pd
import argparse
import pandas as pd
import torch
# from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np
# import torch.nn as nn
# import torch.optim as optim

from matplotlib import pyplot as plt

# from transformers import EncoderDecoderModel, BertTokenizer
from sklearn.model_selection import train_test_split

def main():
    ###### SETUP ############################
    from datasets import load_dataset
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

if __name__ == '__main__':
    main()