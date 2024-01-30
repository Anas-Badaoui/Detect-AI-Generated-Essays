from custom_dataset import DAIGTDataset
from custom_model import DAIGTModel
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch
from utils import AverageMeter
import time
import os
import numpy as np

def get_dataloader(ds, tokenizer, max_len, batch_size, split='train'):
    ds.set_format("pandas")
    text_list = ds[split]['text'].tolist()
    label_list = ds[split]['label'].tolist()

    train_datagen = DAIGTDataset(text_list, tokenizer, max_len, label_list)
    train_generator = DataLoader(dataset=train_datagen,
                                 batch_size=batch_size,
                                 pin_memory=False)
    return train_generator

def get_model_for_training(model_path, pretrained_model_path):    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DAIGTModel(model_path, config, tokenizer, pretrained=True)
    # Load weights from the custom pretrained model
    model.load_state_dict(torch.load(pretrained_model_path))
    return model

def launch_training(model, num_epoch, train_generator, device, optimizer, scheduler, out_dir):
    """
    Function to launch the training process.

    Args:
        model (torch.nn.Module): The model to be trained.
        num_epoch (int): The number of training epochs.
        train_generator (torch.utils.data.DataLoader): The data generator for training.
        device (torch.device): The device to be used for training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        out_dir (str): The output directory to save the trained model.

    Returns:
        None
    """
    start_time = time.time()

    scaler = GradScaler()
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()

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

            print('\r',end='',flush=True)
            message = '%s %5.1f %6.1f    |     %0.3f     |' % ("train",j/len(train_generator)+ep,ep,losses.avg)
            print(message , end='',flush=True)

        
        print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model.state_dict(), out_dir+'weights_finetuned_ep{}'.format(ep))

    end_time = time.time()
    print(end_time-start_time)

    
def evaluate_model(model, test_generator, batch_size):
    """
    Evaluates the given model on the test data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_generator (torch.utils.data.DataLoader): The data generator for the test data.
        batch_size (int): The batch size for evaluation.

    Returns:
        pred_labels (numpy.ndarray): The predicted labels for the test data.
        pred_prob (numpy.ndarray): The predicted probabilities for the test data.
    """
    model.eval()
    pred_prob = []

    for j, (batch_input_ids, batch_attention_mask, batch_label) in enumerate(test_generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(test_generator)-1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask)            
            pred_prob.append(logits.sigmoid().cpu().data.numpy().squeeze())

    pred_prob = np.concatenate(pred_prob)
    pred_labels = np.where(pred_prob > 0.5, 1, 0)
    
    return pred_labels, pred_prob

