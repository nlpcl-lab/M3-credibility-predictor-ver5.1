import numpy as np
import argparse, time

import torch
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
)

from dataset import Cred_Dataset
from tqdm import tqdm

from utils import compute_metrics, print_result, load_tokenizer, load_model, load_optimizer, load_scheduler, format_time


def get_args():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default="1")
    parser.add_argument('--output_dir', type=str, default='./output_models/{}/cluster_{}/{}/')
    parser.add_argument('--model_path', type=str, default='./output_models/{}/cluster_{}/{}/checkpoint-16_1_2000/')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_path", type=str, default="./data/{}/cluster_{}/{}.json")
    parser.add_argument("--ver", type=str, default="ver7.1")
    parser.add_argument("--cluster_num", type=str, default="7")
    parser.add_argument('--num_labels', type=int, default=11)
    
    # model related
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--logging_steps', type=int, default=100)

    return parser.parse_args()




def main(args):

    print(args)
    set_seed(args.seed)

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cluster_nums = ['0','1','2','3']
    task_types = ['shallow_mean', 'shallow_std', 'deep_mean', 'deep_std']
    #task_types = ['shallow_std']


    # results: [0(shallow_mean), 0(shallow_std), 0(deep_mean),  0(deep_std)] * clusters
    all_results = []
    

    # Iterate tests for each cluster: 0 ~ 3
    for cluster_idx, cluster_num in enumerate(cluster_nums):
        results = [0]*len(task_types)

        # Iterate tests for each task type: ['shallow_mean', 'shallow_std', 'deep_mean', 'deep_std']
        for task_idx, task_type in enumerate(task_types):

            

            # load model
            model_path = args.model_path.format(args.ver, cluster_num, task_type)
            tokenizer = load_tokenizer(model_path)
            model = load_model(model_path)

            model.cuda()

            # load data
            test_dataset = Cred_Dataset(
                args = args,
                mode = 'test',
                cluster_num = cluster_num,
                task_type=task_type,
                tokenizer=tokenizer,
            )


            test_dl = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
            )


            model.eval()
            all_preds = []
            all_labels = []
            t0 = time.time()

            for step, data in enumerate(tqdm(test_dl, desc='test', mininterval=0.01, leave=True), 0):
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

                if len(all_preds)==0:
                    all_preds = preds.detach().cpu().clone().numpy()
                    all_labels = labels.detach().cpu().clone().numpy()
                else:
                    all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
                    all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())


            test_result = compute_metrics(labels=all_labels, preds=all_preds)
            
            results[task_idx] = round(test_result['acc'], 2)
        all_results.append(results)



    # calculate final results
    final_result = np.mean(all_results, axis=0)

    # print result
    print(all_results)
    for task_idx, task_type in enumerate(task_types):
        print('*** Averaged {}: {}'.format(task_type, str(round(final_result[task_idx],2))))
   



if __name__ == '__main__':
    from eval import get_args
    args = get_args()
    main(args)
