import numpy as np
import argparse, time

from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import get_linear_schedule_with_warmup
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
)

from tqdm import tqdm

from datasets import load_metric


from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)

from dataset import Cred_Dataset

from utils import save_cp, format_time, compute_metrics, print_result


def get_args():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default="1")
    parser.add_argument('--output_dir', type=str, default='./output_models/{}/cluster_{}/{}/')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_path", type=str, default="./data/{}/cluster_{}/{}.json")
    parser.add_argument("--ver", type=str, default="ver5.1")
    parser.add_argument("--cluster_num", type=str, default="0")
        
    # dataset related
    parser.add_argument("--task_name", type=str, default="shallow")
    parser.add_argument("--prediction_type", type=str, default="std")
    parser.add_argument('--num_labels', type=int, default=11)
    
    # model related
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    #parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--saving_steps', type=int, default=2000)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, default=(0.9, 0.98), nargs='+')

    # trainer related
    parser.add_argument("--load_from_checkpoint", type=str, default="")
    parser.add_argument('--debug', action='store_true')
    
    
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument('--fast_dev_run', action='store_true', default=True)
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--do_test", action="store_true", default=True)
   
    

    return parser.parse_args()



def main(args):

    print(args)
    set_seed(args.seed)

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    writer = SummaryWriter(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_type = args.task_name + '_' + args.prediction_type
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        cache_dir=args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        cache_dir=args.cache_dir,
        num_labels=args.num_labels,
    )
    model.cuda()
    
    # Prepare data
    train_dataset = Cred_Dataset(
        args = args,
        mode = 'train',
        cluster_num = args.cluster_num,
        task_type=task_type,
        tokenizer=tokenizer,
    )

    val_dataset = Cred_Dataset(
        args = args,
        mode = 'valid',
        task_type=task_type,
        tokenizer=tokenizer,
    )

    # Load Data
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    valid_dl = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    num_training_steps = args.epochs * (train_dataset.num_data / args.batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )


    # Training starts
    total_train_step = 0
    total_valid_step = 0

    for epoch_i in range(0,args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        
        t0 = time.time()

        # train starts
        print('Training...')
        all_preds = []
        all_labels = []
        total_loss = 0.0
        loss_for_logging = 0.0

        for step, data in enumerate(tqdm(train_dl, desc='train', mininterval=0.01, leave=True), 0):
            
            inputs = {
                    "input_ids": data['input_ids'].to(device),
                    "attention_mask": data['attention_mask'].to(device),
                    #"token_type_ids":data['token_type_ids'].to(device),
                }
            labels = data['labels'].to(device)
            
            
            # foward
            outputs = model(**inputs, labels=labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss_for_logging += loss.item()
            logits = outputs[1]
            preds = logits.argmax(-1)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            
            # logging
            if step % args.logging_steps == 0 and not step==0:
                writer.add_scalar('Train/loss', (loss_for_logging/args.logging_steps), total_train_step)
                loss_for_logging = 0

                elapsed = format_time(time.time() - t0)
                #print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}'.format(step, len(train_dl), elapsed, loss))


            if len(all_preds)==0:
                all_preds = preds.detach().cpu().clone().numpy()
                all_labels = labels.detach().cpu().clone().numpy()
            else:
                all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
                all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())

            
            # save checkpoint
            if total_train_step % args.saving_steps == 0 and not total_train_step==0:
                save_dir_path = args.output_dir.format(args.ver, args.cluster_num, task_type)
                save_cp(args, 
                    save_dir_path, args.batch_size, epoch_i,
                    total_train_step,
                    model, 
                    optimizer, 
                    scheduler, 
                    tokenizer
                )
            

            # when step ends
            total_train_step+=1
            model.zero_grad()

           

        # epoch ends
        # print results
        print("total train loss: {}".format(total_loss / len(train_dl)))
        train_result = compute_metrics(labels=all_labels, preds=all_preds)
        print_result(train_result)
        print("  Train epoch took: {:}".format(format_time(time.time() - t0)))

        # save checkpoint
        """
        save_dir_path = args.output_dir.format(args.ver, args.cluster_num, task_type)
        save_cp(args, 
            save_dir_path, args.batch_size, epoch_i,
            total_train_step,
            model, 
            optimizer, 
            scheduler, 
            tokenizer
        )
        """



        # validation starts
        if args.do_eval == False:
            continue

        print("")
        print("Running Validation...")
        all_preds = []
        all_labels = []
        loss_for_logging = 0
        for step, data in enumerate(tqdm(valid_dl, desc='valid', mininterval=0.01, leave=True),0):
            inputs = {
                    "input_ids": data['input_ids'].to(device),
                    "attention_mask": data['attention_mask'].to(device),
                    #"token_type_ids":data['token_type_ids'].to(device),
                }
            labels = data['labels'].to(device)

            with torch.no_grad():
                outputs = model(**inputs, labels=labels)

            logits = outputs[1]
            preds = logits.argmax(-1)
            loss = outputs[0]
            loss_for_logging+=loss


            # logging
            if step % args.logging_steps == 0 and not step==0:
                writer.add_scalar('Valid/loss', (loss_for_logging/args.logging_steps), total_valid_step)
                loss_for_logging = 0

            if len(all_preds)==0:
                all_preds = preds.detach().cpu().clone().numpy()
                all_labels = labels.detach().cpu().clone().numpy()
            else:
                all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
                all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())


        print("")
        val_result = compute_metrics(labels=all_labels, preds=all_preds)
        print_result(val_result)
        print("  Validation epoch took: {:}".format(format_time(time.time() - t0)))


    print("")
    print("Training complete")

    


if __name__ == '__main__':
    from train import get_args
    args = get_args()
    main(args)
