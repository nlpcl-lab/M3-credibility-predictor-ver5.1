import time, datetime, random, os, json
import numpy as np

from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, auc, roc_curve)
from transformers import get_linear_schedule_with_warmup
import torch

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label, features=[]):
        self.guid = guid
        self.label = label
        self.text_a = text_a
        self.text_b = text_b
        self.features = features

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



def generate_examples(mode, contexts, texts, labels):
    examples = []
    for idx in range(len(texts)):
        guid = "%s-%s" % (mode, idx)
        text_a = texts[idx]
        if contexts =='':
            text_b = None
        else:
            text_b = contexts[idx]
        label = labels[idx]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples



def f1_pre_rec_scalar(labels, preds, main_label=1):
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=main_label)#roc_curve(np.sort(labels), np.sort(preds), pos_label=main_label)
    return {
        "acc": simple_accuracy(labels, preds),
    }


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec_scalar(labels, preds)
    

def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def print_result(test_result):
	for name, value in test_result.items():
		print('   Average '+name, value)



def save_cp(args, save_dir_path, batch_size, epochs, steps, model, optimizer, scheduler, tokenizer):
    save_dir_path = save_dir_path+'checkpoint-{}_{}_{}/'.format(batch_size, epochs, steps)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    print('*** Saving checkpoints at {}'.format(save_dir_path))
    torch.save(model, save_dir_path+'model.bin')
    torch.save(optimizer, save_dir_path+'optimizer.pt')
    torch.save(scheduler.state_dict(), save_dir_path+'scheduler.pt')
    torch.save(tokenizer, save_dir_path+'tokenizer.json')



# load model
def load_tokenizer(path):
    return torch.load(path+'tokenizer.json')

def load_model(path):
    return torch.load(path+'model.bin')

def load_optimizer(path):
    return torch.load(path+'optimizer.pt')

def load_scheduler(path, optimzer, warmup_steps, num_training_steps):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    scheduler.load_state_dict(torch.load(path+'scheduler.pt'))
    
    return scheduler




