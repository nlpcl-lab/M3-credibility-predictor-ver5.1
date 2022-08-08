import os, json, time

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
)

from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizer



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



def generate_examples(mode, texts, evidence, labels):
    examples = []
    for idx in range(len(texts)):
        guid = "%s-%s" % (mode, idx)
        text_a = texts[idx]
        if evidence is not None:
            text_b = evidence[idx]
        else:
            text_b = None
        label = labels[idx]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples



class Cred_Dataset(Dataset):
    def __init__(self, args, mode, cluster_num, task_type, tokenizer=PreTrainedTokenizer):
        self.args=args
        self.data_path = args.data_path
        self.mode = mode
        self.cluster_num = cluster_num
        self.label_list = [str(i) for i in range(11)]

        cached_features_file = os.path.join(
            args.cache_dir if args.cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}_{}".format(
                self.args.ver,
                self.cluster_num,
                task_type,
                self.mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_len),
            ),
        )

        
        if os.path.exists(cached_features_file):
            print("*** Loading features from cached file {}".format(cached_features_file))
            (self.features, self.datas, self.idx2docid) = torch.load(cached_features_file)
            self.num_data=len(self.features['labels'])

        else:

            # data load: "./dataset/{}.json".format(mode) : train, valid, test --> "./dataset/train.json"
            with open(self.data_path.format(self.args.ver, self.cluster_num, mode), 'r') as fp:
                self.datas = json.load(fp)

            texts=[]
            evidence=[]
            labels=[]
            self.idx2docid = dict()

            for idx, data in enumerate(self.datas):
                self.idx2docid[str(idx)] = data
                texts.append(self.datas[data]['title'])
                if task_type=='shallow_mean' or task_type=='shallow_std':
                    evidence.append(None)
                else:
                    evidence.append(self.datas[data]['evidence'])
                labels.append(str(self.datas[data][task_type]))

            assert len(texts) == len(evidence) == len(labels)
            self.num_data=len(texts)

            examples = generate_examples(self.mode, texts, evidence, labels)

            output_mode = "classification"
            num_labels = args.num_labels
            label_map = {label: i for i, label in enumerate(self.label_list)}
            def label_from_example(label):
                if output_mode == "classification":
                    return label_map[label]
                elif output_mode == "regression":
                    return float(label)
                raise KeyError(output_mode)
            self.labels = [label_from_example(example.label) for example in examples]

            self.encodings = tokenizer.batch_encode_plus(
                [(example.text_a, example.text_b) if example.text_b else example.text_a for example in examples],
                max_length=args.max_seq_len,
                padding='max_length',
                truncation='longest_first',
                return_tensors="pt",
            )

            
            self.features = self.encodings
            self.features['labels'] = torch.tensor(self.labels)
            print("*** Saving features into cached file {}".format(cached_features_file))
            torch.save((self.features, self.datas, self.idx2docid), cached_features_file)



    
    def __len__(self):
        return len(self.features['labels'])
    

    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.features.items()}
        return item

    def get_labels(self):
        return self.features['labels']





if __name__ == '__main__':
    from train import get_args
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    ds = Cred_Dataset(
        args = args,
        mode='train',
        task_type='shallow_mean',
        tokenizer=tokenizer,
    )
    dl = DataLoader(ds, batch_size=args.batch_size)
    d = next(iter(dl))
    import IPython; IPython.embed(); exit(1)