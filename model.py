import timm
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM


class PromptEncoder(nn.Module):
    '''learnable token generator modified from P-tuning
    https://github.com/THUDM/P-tuning
    '''
    def __init__(self, prompt_token_len, hidden_size, device, lstm_dropout):
        super().__init__()
        print("[#] Init prompt encoder...")
        # Input [0,1,2,3,4,5,6,7,8]
        self.seq_indices = torch.LongTensor(list(range(prompt_token_len))).to(device)
        # Embedding
        self.embedding = nn.Embedding(prompt_token_len, hidden_size)
        # LSTM
        self.lstm_head = nn.LSTM(input_size=hidden_size,
                                       hidden_size=hidden_size // 2,
                                       num_layers=2,
                                       dropout=lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size))

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class VisualEncoder(nn.Module):
    def __init__(self, model_name, img_token_len, embedding_dim):
        super().__init__()
        self.is_resnet = False
        self.img_token_len = img_token_len
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(model_name, pretrained=True)
        if "resnet" in model_name:
            self.is_resnet = True
            if model_name == "resnet50":
                self.global_pool = self.backbone.global_pool
            else:
                self.global_pool = self.backbone.head.global_pool
            self.visual_mlp = nn.Linear(2048, img_token_len * embedding_dim)  # 2048 -> n * 768
        elif "vit" in model_name:
            self.visual_mlp = nn.Linear(768, img_token_len * embedding_dim)  # 768 -> n * 768
        
    def forward(self, imgs_tensor):
        bs = imgs_tensor.shape[0]
        visual_embeds = self.backbone.forward_features(imgs_tensor)
        if self.is_resnet:
            visual_embeds = self.global_pool(visual_embeds).reshape(bs, 2048)
        visual_embeds = self.visual_mlp(visual_embeds)
        visual_embeds = visual_embeds.reshape(bs, self.img_token_len, self.embedding_dim)

        return visual_embeds


class MSAModel(torch.nn.Module):
    '''main model
    '''
    def __init__(self, args, label_id_list):
        super().__init__()
        self.args = args
        self.label_id_list = label_id_list

        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        self.lm_model = BertForMaskedLM.from_pretrained(args.model_name)

        self.embeddings = self.lm_model.bert.get_input_embeddings()
        self.embedding_dim = self.embeddings.embedding_dim  # 768

        if not args.no_img:
            self.img_token_id = self.tokenizer.get_vocab()[args.img_token]
            self.img_token_len = args.img_token_len
            self.visual_encoder = VisualEncoder(args.visual_model_name, self.img_token_len, self.embedding_dim)

        if args.template == 3:
            self.prompt_token_id = self.tokenizer.get_vocab()[args.prompt_token]
            self.prompt_token_len = sum([int(i) for i in args.prompt_shape.split('-')[0]]) + int(args.prompt_shape[-1])
            self.prompt_encoder = PromptEncoder(self.prompt_token_len, self.embedding_dim, args.device, args.lstm_dropout)

    def embed_input(self, input_ids, imgs=None):
        bs = input_ids.shape[0]
        embeds = self.embeddings(input_ids)

        if self.args.template == 3:
            prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
            prompt_embeds = self.prompt_encoder()
            for bidx in range(bs):
                for i in range(self.prompt_token_len):
                    embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[i, :]
        
        if not self.args.no_img:
            visual_embeds = self.visual_encoder(imgs)
            img_token_position = torch.nonzero(input_ids == self.img_token_id).reshape((bs, self.img_token_len, 2))[:, :, 1]
            for bidx in range(bs):
                for i in range(self.img_token_len):
                    embeds[bidx, img_token_position[bidx, i], :] = visual_embeds[bidx, i, :]
        
        return embeds
    
    def forward(self, input_ids, attention_mask, labels, imgs=None):
        inputs_embeds = self.embed_input(input_ids, imgs)
        output = self.lm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss, logits = output.loss, output.logits

        pred = logits[labels != -100]
        probs = pred[:, self.label_id_list]
        pred_labels_idx = torch.argmax(probs, dim=-1).tolist()
        y_ = [self.label_id_list[i] for i in pred_labels_idx]

        y = labels[labels != -100]

        return loss, y_, y.tolist()
