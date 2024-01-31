import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch
import random
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup

import sys
sys.path.append('./src')
from utils import AverageMeter
from custom_dataset import DAIGTDataset
from custom_model import DAIGTModel


def main():
    """
    Main function for training the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    with open('./data/train_neg_list.pickle', 'rb') as f:
        neg_list = pickle.load(f)
    with open('./data/train_pos_list.pickle', 'rb') as f:
        pos_list = pickle.load(f)
    text_list = neg_list + pos_list
    label_list = [0]*len(neg_list) + [1]*len(pos_list)
    print(len(neg_list), len(pos_list))

    # hyperparameters
    learning_rate = 1e-5
    max_len = 256
    batch_size = 128
    num_epoch = 1
    model_path = 'distilroberta-base'

    # build model
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DAIGTModel(model_path, config, tokenizer, pretrained=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_train_steps = int(len(text_list)/(batch_size*3)*num_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    # training
    train_datagen = DAIGTDataset(text_list, tokenizer, max_len, label_list)
    train_generator = DataLoader(dataset=train_datagen,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=False)

    start_time = time.time()

    scaler = GradScaler()
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        progress_bar = tqdm(range(len(train_generator)))
        for j, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(train_generator):
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.float().to(device)

            with autocast():
                logits = model(batch_input_ids, batch_attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), batch_labels)

            losses.update(loss.item(), batch_input_ids.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            progress_bar.update(1)
            print('\r',end='',flush=True)
            message = '%s %5.1f %6.1f    |     %0.3f     |' % ("train",j/len(train_generator)+ep,ep,losses.avg)
            print(message , end='',flush=True)

        print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    out_dir = 'outputs/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model.state_dict(), out_dir+'weights_pre_trained_ep{}'.format(ep))

    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
