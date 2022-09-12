import copy
import os
import torch
import time
import sklearn.metrics as metrics
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from param import args
from utils import processors
from model import MSAModel
from dataset import MSADataset
from transformers.optimization import AdamW


if args.dataset in ['t2015', 't2017', 'masad']:
    print('[#] Aspect-level')
else:
    print('[#] Sentence-level')


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # load datasets and create dataloaders
        self.train_set = MSADataset(args, processors[args.dataset], mode='train', max_seq_length=64 if args.dataset=='masad' else 128)
        self.train_loader = DataLoader(self.train_set, collate_fn=self.train_set.collate_fn, batch_size=args.batch_size, shuffle=True, drop_last=False)
        self.dev_set = MSADataset(args, processors[args.dataset], mode='dev', max_seq_length=64 if args.dataset=='masad' else 128)
        self.dev_loader = DataLoader(self.dev_set, collate_fn=self.dev_set.collate_fn, batch_size=args.batch_size * 4)
        self.test_set = MSADataset(args, processors[args.dataset], mode='test', max_seq_length=64 if args.dataset=='masad' else 128)
        self.test_loader = DataLoader(self.test_set, collate_fn=self.test_set.collate_fn, batch_size=args.batch_size * 4)
        
        # create model
        self.model = MSAModel(args, self.test_set.label_id_list)
        self.model.to(self.device)

        # create ckpt
        self.best_ckpt = {
            'test_size': len(self.test_set),
            'args': self.args
        }

        self.save_dir = self.get_save_dir()
        os.makedirs(self.save_dir, exist_ok=True)

    def get_save_dir(self):
        if '18' in self.args.few_shot_file:
            sample = '[18]'
        elif '36' in self.args.few_shot_file:
            sample = '[36]'
        elif '72' in self.args.few_shot_file:
            sample = '[72]'
        elif '108' in self.args.few_shot_file:
            sample = '[108]'
        elif '144' in self.args.few_shot_file:
            sample = '[144]'
        elif '1' in self.args.few_shot_file:
            sample = '[s1]'
        elif '2' in self.args.few_shot_file:
            sample = '[s2]'
        else:
            sample = '[s0]'
        template_name = '{}[t{}]'.format(sample, self.args.template)
        if self.args.template == 3:
            template_name += '[{}]'.format(self.args.prompt_shape)
        if not self.args.no_img:
            template_name += "[{}-{}]".format(self.args.visual_model_name, self.args.img_token_len)
        
        if self.args.lr_visual_encoder and self.args.lr_visual_encoder > 0:
            return os.path.join(self.args.out_dir, self.args.dataset, template_name, 'lrv_{}'.format(self.args.lr_visual_encoder), str(self.args.lr_lm_model))
        else:
            return os.path.join(self.args.out_dir, self.args.dataset, template_name, str(self.args.lr_lm_model))

    def save(self):
        '''save predictions but
        do not save model parameters
        '''
        ckpt_name = self.best_ckpt['ckpt_name']
        test_y = self.best_ckpt['test_y']
        test_y_ = self.best_ckpt['test_y_']
        with open(os.path.join(self.save_dir, ckpt_name) + '.txt', 'w', encoding='utf-8') as f:
            f.write('#True\t#Pred\n')
            for y, y_ in zip(test_y, test_y_):
                token_y = self.test_set.tokenizer.convert_ids_to_tokens(y)
                token_y_ = self.test_set.tokenizer.convert_ids_to_tokens(y_)
                f.write(f'{token_y:10}\t{token_y_:10}\n')
        print("[#] Checkpoint {} saved.".format(ckpt_name))
        return

    def load(self):
        """load ckpt
        """
        if self.args.ckpt_name == None:
            print("[#] Loading nothing, using BERT params")
            return
        ckpt_dict = torch.load(self.args.ckpt_name, map_location='cpu')
        print(f'[#] Loading {ckpt_dict["ckpt_name"]}')
        print(f'[#] dev_acc {ckpt_dict["dev_acc"]}')
        print(f'[#] test_acc {ckpt_dict["test_acc"]}')
        print(f'[#] test_mac_f1 {ckpt_dict["test_mac_f1"]}')
        self.model.load_state_dict(ckpt_dict["embedding"])

    def _evaluate(self, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
        else:
            loader = self.dev_loader
        with torch.no_grad():
            self.model.eval()
            loss = []
            y_ = []
            y = []
            
            pbar = tqdm(loader, unit="batch", desc=f'*{evaluate_type} pbar')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
                
                _loss, _y_, _y = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)

            loss = sum(loss) / len(loader)
            acc = metrics.accuracy_score(y, y_)
            f1_macro = metrics.f1_score(y, y_, average='macro')
            print(f"[{evaluate_type:5}] Loss: {loss:0.4f} Acc: {acc:0.4f}, Macro F1: {f1_macro:0.4f}")
        return loss, acc, f1_macro, y, y_

    def update_best_ckpt(self, epoch_idx=None, dev_acc=None, test_acc=None, test_f1=None, test_y=None, test_y_=None):
        if test_acc == None:  # during training
            model_params = copy.deepcopy(self.model.state_dict())
            self.best_ckpt['time'] = datetime.now()
            self.best_ckpt['embedding'] = model_params
            self.best_ckpt['epoch'] = epoch_idx
            self.best_ckpt['dev_acc'] = dev_acc
        else:  # after testing
            ckpt_name = time.strftime("%y%m%d_%H:%M:%S", time.localtime())
            epoch = epoch_idx if epoch_idx else self.best_ckpt['epoch']
            ckpt_name += "[Ep{}][Test{}-{}][Dev{}].ckpt".format(epoch, round(test_acc * 100, 2), round(test_f1 * 100, 2), round(self.best_ckpt['dev_acc'] * 100, 2))
            self.best_ckpt['ckpt_name'] = ckpt_name
            # self.best_ckpt['dev_acc'] = dev_acc
            self.best_ckpt['test_acc'] = test_acc
            self.best_ckpt['test_mac_f1'] = test_f1
            self.best_ckpt['test_y'] = test_y
            self.best_ckpt['test_y_'] = test_y_
        
    def train(self):
        best_dev_acc, early_stop = 0, 0
        
        params = []
        params.append({'params': self.model.lm_model.parameters(), 'lr': self.args.lr_lm_model})
        if self.args.template == 3:
            params.append({'params': self.model.prompt_encoder.parameters(), 'lr': self.args.lr_lm_model})
        if not self.args.no_img:
            params.append({'params': self.model.visual_encoder.backbone.parameters(), 'lr': self.args.lr_visual_encoder})
            params.append({'params': self.model.visual_encoder.visual_mlp.parameters(), 'lr': self.args.lr_lm_model})
        
        optimizer = AdamW(params=params, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(100):
            print(f'\n[#] Epoch {epoch_idx}')
            loss = []
            y_ = []
            y = []

            pbar = tqdm(self.train_loader, unit="batch")
            for batch in pbar:
                self.model.train()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                imgs = batch['imgs'].to(self.device) if not self.args.no_img else None
            
                _loss, _y_, _y = self.model(input_ids, attention_mask, labels, imgs)
                loss.append(_loss.item())
                y_.extend(_y_)
                y.extend(_y)
                pbar.set_description(f'*Train batch loss: {_loss.item():0.4f}')

                _loss.backward()
                # torch.cuda.empty_cache()
                optimizer.step()
                # torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()
            loss = sum(loss) / len(self.train_loader)
            acc = metrics.accuracy_score(y, y_)
            print(f"[Train] Loss: {loss:0.4f} Hit@1: {acc:0.4f}")

            dev_loss, dev_acc, dev_f1, _, _ = self._evaluate('Dev')
            if dev_acc >= best_dev_acc:
                print(f'[#] Best dev acc: {dev_acc:0.4f}')
                self.update_best_ckpt(epoch_idx, dev_acc)
                early_stop = 0
                best_dev_acc = dev_acc
            else:
                early_stop += 1
                if early_stop >= self.args.early_stop:
                    print("[*] Early stopping at epoch {}.".format(epoch_idx))
                    return
        print('[*] Ending Training')

    def evaluate_on_test(self):
        print('[#] Begin to evaluate on test set')
        self.model.load_state_dict(self.best_ckpt["embedding"])
        _, test_acc, test_f1_mac, test_y, test_y_ = self._evaluate('Test')
        self.update_best_ckpt(None, None, test_acc, test_f1_mac, test_y, test_y_)
        self.save()


def main():
    trainer = Trainer(args)
    if args.ckpt_name:
        trainer.load()
    trainer.train()
    trainer.evaluate_on_test()


if __name__ == '__main__':
    main()
