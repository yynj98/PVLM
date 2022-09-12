import csv
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms


class MSADataset(Dataset):
    def __init__(self, args, processor, mode='train', max_seq_length=128):
        self.dataset = args.dataset
        self.no_img = args.no_img
        
        self.data_dir = args.data_dir
        self.img_dir = args.img_dir

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name, local_files_only=True)
        self.vocab = self.tokenizer.get_vocab()

        self.prompt_token = args.prompt_token
        self.prompt_token_id = self.vocab[args.prompt_token]
        self.prompt_shape = [int(i) for i in args.prompt_shape.split('-')[0]]
        self.prompt_img_len = int(args.prompt_shape[-1])

        self.img_token_id = self.vocab[args.img_token]
        self.img_token_len = args.img_token_len
        self.max_seq_length = max_seq_length
        
        self.template = args.template
        label_list, label_map, self.template_dict = processor(args.template)
        self.label_id_list = [self.vocab[token] for token in label_list]
        self.label_id_map = {key: self.vocab[label_map[key]] for key in label_map.keys()}

        if mode == 'train':
            print("[#] Looking At {}".format(os.path.join(self.data_dir, args.few_shot_file)))
            self.lines = self._read_tsv(os.path.join(self.data_dir, args.few_shot_file))
        else:
            print("[#] Looking At {}".format(os.path.join(self.data_dir, f"{mode}.tsv")))
            self.lines = self._read_tsv(os.path.join(self.data_dir, f"{mode}.tsv"))

        if not args.no_img:
            print("[|] Reading imgs...")
            self.img_dict = self._read_imgs()
        
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            lines.pop(0)  # remove the header row
            return lines

    def _read_imgs(self):
        img_dict = {}
        for line in self.lines:
            img_id = line[2]
            img = Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')
            img = transforms.Resize([224, 224])(img)
            img = transforms.ToTensor()(img)  # (3,224,224)
            img_dict[img_id] = img
        return img_dict

    def _supervised_encode(self, tokens_x, tokens_a, label_id):
        if len(tokens_x) > self.max_seq_length:
            tokens_x = tokens_x[:self.max_seq_length]

        # textual template
        # [CLS] for " [a] " , the sentence " [x] " has [bad/no/good] emotion [SEP]
        # [CLS] for " [a] " , the sentence " [x] " presents a [negative/neutral/positive] sentiment [SEP]
        # [CLS] [negative/neutral/positive] [p] [a] [p] [x] [p]
        p_idx = 0
        input_tokens = []
        for i in self.template_dict['map']:
            if i == 'a':
                input_tokens.extend(tokens_a)
            elif i == 'x':
                input_tokens.extend(tokens_x)
            elif i == 'p':
                input_tokens.extend([self.prompt_token] * self.prompt_shape[p_idx])
                p_idx += 1
            else:
                input_tokens.extend(self.tokenizer.tokenize(self.template_dict['content'][i]))
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(input_ids)
        for i, id in enumerate(input_ids):
            if id == self.tokenizer.mask_token_id:
                labels[i] = label_id

        assert len(input_ids) == len(attention_mask) == len(labels)
        return input_ids, attention_mask, labels

    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        line = self.lines[idx]
        label_id = self.label_id_map[line[1]]
        img_id = line[2]
        text_x = line[3].lower()
        text_a = line[4].lower()
        if self.dataset in ['t2015', 't2017']:
            text_x = text_x.replace('$t$', text_a)

        tokens_x = self.tokenizer.tokenize(text_x)
        tokens_a = self.tokenizer.tokenize(text_a)
        
        input_ids, attention_mask, labels = self._supervised_encode(tokens_x, tokens_a, label_id)

        if self.no_img:
            img = None
        else:
            img = self.img_dict[img_id]
            if self.template == 1 or self.template == 2:
                # [CLS]             [textual template] [SEP]
                # [CLS] [Img] [SEP] [textual template] [SEP]
                addon = [self.img_token_id] * self.img_token_len + [self.tokenizer.sep_token_id]
                input_ids = [input_ids[0]] + addon + input_ids[1:]
                attention_mask = [1] * (self.img_token_len + 1) + attention_mask
                labels = [-100] * (self.img_token_len + 1) + labels
            elif self.template == 3:
                # [CLS]      [M][P][A][P][C][SEP]
                # [CLS][I][P][M][P][A][P][C][SEP]
                addon = [self.img_token_id] * self.img_token_len + [self.prompt_token_id] * self.prompt_img_len
                input_ids = [input_ids[0]] + addon + input_ids[1:]
                attention_mask = [1] * (self.img_token_len + self.prompt_img_len) + attention_mask
                labels = [-100] * (self.img_token_len + self.prompt_img_len) + labels

            assert len(input_ids) == len(attention_mask) == len(labels)
            
        return {
            'img_id': img_id,
            'img': img,
            "text_x": text_x,
            "text_a": text_a,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def collate_fn(self, batch):
        input_ids = [torch.tensor(instance["input_ids"]) for instance in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask = [torch.tensor(instance["attention_mask"]) for instance in batch]
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        labels = [torch.LongTensor(instance["labels"]) for instance in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        returns = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if not self.no_img:
            imgs = [instance['img'] for instance in batch]
            imgs = torch.stack(imgs, dim=0)
            returns['imgs'] = imgs
        
        return returns
    