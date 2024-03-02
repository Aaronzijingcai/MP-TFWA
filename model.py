import math

import torch
import torch.nn.functional as F
from torch import nn


class MP_TFWA(nn.Module):
    def __init__(self, mrc_model, co_model, pl_model, num_classes, max_lengths, query_lengths, prompt_lengths):
        super().__init__()
        self.mrc_model = mrc_model
        self.co_model = co_model
        self.pl_model = pl_model
        self.num_classes = num_classes
        self.max_lengths = max_lengths
        self.query_lengths = query_lengths + 1
        self.prompt_lengths = prompt_lengths + 1

        for param in mrc_model.parameters():
            param.requires_grad = True
        for param in co_model.parameters():
            param.requires_grad =True
        for param in pl_model.parameters():
            param.requires_grad = True


        # MRC-IE
        self.t_mrc_key = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_mrc_query = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_mrc_value = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_mrc_norm = 1 / math.sqrt(self.co_model.config.hidden_size)

        self.f_mrc_key = nn.Linear(self.max_lengths + self.query_lengths,
                                   self.max_lengths + self.query_lengths)
        self.f_mrc_query = nn.Linear(self.max_lengths + self.query_lengths,
                                     self.max_lengths + self.query_lengths)
        self.f_mrc_value = nn.Linear(self.max_lengths + self.query_lengths,
                                     self.max_lengths + self.query_lengths)
        self.f_mrc_norm = 1 / math.sqrt(self.max_lengths + self.query_lengths)

        # Context-IE
        self.t_co_key = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_co_query = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_co_value = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_co_norm = 1 / math.sqrt(self.co_model.config.hidden_size)

        self.f_co_key = nn.Linear(self.max_lengths, self.max_lengths)
        self.f_co_query = nn.Linear(self.max_lengths, self.max_lengths)
        self.f_co_value = nn.Linear(self.max_lengths, self.max_lengths)
        self.f_co_norm = 1 / math.sqrt(self.max_lengths)

        # PL-IE
        self.t_pl_key = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_pl_query = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_pl_value = nn.Linear(self.co_model.config.hidden_size, self.co_model.config.hidden_size)
        self.t_pl_norm = 1 / math.sqrt(self.co_model.config.hidden_size)

        self.f_pl_key = nn.Linear(self.max_lengths + self.prompt_lengths,
                                  self.max_lengths + self.prompt_lengths)
        self.f_pl_query = nn.Linear(self.max_lengths + self.prompt_lengths,
                                    self.max_lengths + self.prompt_lengths)
        self.f_pl_value = nn.Linear(self.max_lengths + self.prompt_lengths,
                                    self.max_lengths + self.prompt_lengths)
        self.f_pl_norm = 1 / math.sqrt(self.max_lengths + self.prompt_lengths)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear((1000 + self.co_model.config.hidden_size) * 3, self.co_model.config.hidden_size),
            nn.Linear(self.co_model.config.hidden_size, num_classes)
        )

        self.mrcTW = nn.Linear(self.co_model.config.hidden_size, 100)
        self.mrcFW = nn.Linear(self.co_model.config.hidden_size, 100)
        self.coTW = nn.Linear(self.co_model.config.hidden_size, 100)
        self.coFW = nn.Linear(self.co_model.config.hidden_size, 100)
        self.plTW = nn.Linear(self.co_model.config.hidden_size, 100)
        self.plFW = nn.Linear(self.co_model.config.hidden_size, 100)

        self.mrcWoven = nn.Sequential(
            nn.Linear(10000, 1000)
        )
        self.coWoven= nn.Sequential(
            nn.Linear(10000, 1000)
        )
        self.plWoven = nn.Sequential(
            nn.Linear(10000, 1000)
        )

    def forward(self, mrc_inputs, text_inputs, mask_inputs, mask_index):
        mrc_tokens = self.mrc_model(**mrc_inputs).last_hidden_state
        co_tokens = self.co_model(**text_inputs).last_hidden_state
        pl_tokens = self.pl_model(**mask_inputs).last_hidden_state

        mrc_CLS = mrc_tokens[:, 0, :]
        co_CLS = co_tokens[:, 0, :]
        MASK = pl_tokens[0, mask_index[0, 1], :].reshape((1,self.pl_model.config.hidden_size))
        for i in range(1, mask_index.shape[0]):
            MASK = torch.cat((MASK, pl_tokens[i, mask_index[i, 1], :].reshape((1, self.pl_model.config.hidden_size))), 0)

        mrc_padding = F.pad(mrc_tokens[:, 1:, :].permute(0, 2, 1),
                               (0, self.max_lengths + self.query_lengths - mrc_tokens[:, 1:, :].shape[1]),
                               mode='constant',
                               value=0).permute(0, 2, 1)
        co_padding = F.pad(co_tokens[:, 1:, :].permute(0, 2, 1),
                            (0, self.max_lengths - co_tokens[:, 1:, :].shape[1]),
                            mode='constant',
                            value=0).permute(0, 2, 1)
        pl_padding = F.pad(pl_tokens[:, 1:, :].permute(0, 2, 1),
                               (0, self.max_lengths + self.prompt_lengths - pl_tokens[:, 1:, :].shape[1]),
                               mode='constant',
                               value=0).permute(0, 2, 1)
        # MRC-IE
        t_mrc_K = self.t_mrc_key(mrc_padding)
        t_mrc_Q = self.t_mrc_query(mrc_padding)
        t_mrc_V = self.t_mrc_value(mrc_padding)
        t_mrc_att = nn.Softmax(dim=-1)((torch.bmm(t_mrc_Q, t_mrc_K.permute(0, 2, 1))) * self.t_mrc_norm)
        mrc_TVSA = torch.bmm(t_mrc_att, t_mrc_V)

        f_mrc_K = self.f_mrc_key(mrc_padding.permute(0, 2, 1))
        f_mrc_Q = self.f_mrc_query(mrc_padding.permute(0, 2, 1))
        f_mrc_V = self.f_mrc_value(mrc_padding.permute(0, 2, 1))
        f_mrc_att = nn.Softmax(dim=-1)((torch.bmm(f_mrc_Q, f_mrc_K.permute(0, 2, 1))) * self.f_mrc_norm)
        mrc_FVSA = torch.bmm(f_mrc_att, f_mrc_V).permute(0, 2, 1)

        mrc_TVSA_W = self.mrcTW(mrc_TVSA)
        mrc_FVSA_W = self.mrcFW(mrc_FVSA)
        mrc_TFW = torch.bmm(mrc_TVSA_W.permute(0, 2, 1), mrc_FVSA_W)
        mrc_TFWA = self.mrcWoven(torch.reshape(mrc_TFW, [mrc_TFW.shape[0], 10000]))

        # Context-IE
        t_co_K = self.t_co_key(co_padding)
        t_co_Q = self.t_co_query(co_padding)
        t_co_V = self.t_co_value(co_padding)
        t_co_att = nn.Softmax(dim=-1)((torch.bmm(t_co_Q, t_co_K.permute(0, 2, 1))) * self.t_co_norm)
        co_TVSA = torch.bmm(t_co_att, t_co_V)

        f_co_K = self.f_co_key(co_padding.permute(0, 2, 1))
        f_co_Q = self.f_co_query(co_padding.permute(0, 2, 1))
        f_co_V = self.f_co_value(co_padding.permute(0, 2, 1))
        f_co_att = nn.Softmax(dim=-1)((torch.bmm(f_co_Q, f_co_K.permute(0, 2, 1))) * self.f_co_norm)
        co_FVSA = torch.bmm(f_co_att, f_co_V).permute(0, 2, 1)

        co_TVSA_W = self.coTW(co_TVSA)
        co_FVSA_W = self.coFW(co_FVSA)
        co_TFW = torch.bmm(co_TVSA_W.permute(0, 2, 1), co_FVSA_W)
        co_TFWA = self.coWoven(torch.reshape(co_TFW, [co_TFW.shape[0], 10000]))

        # PL-IE
        t_pl_K = self.t_pl_key(pl_padding)
        t_pl_Q = self.t_pl_query(pl_padding)
        t_pl_V = self.t_pl_value(pl_padding)
        t_pl_att = nn.Softmax(dim=-1)((torch.bmm(t_pl_Q, t_pl_K.permute(0, 2, 1))) * self.t_pl_norm)
        pl_TVSA = torch.bmm(t_pl_att, t_pl_V)

        f_pl_K = self.f_pl_key(pl_padding.permute(0, 2, 1))
        f_pl_Q = self.f_pl_query(pl_padding.permute(0, 2, 1))
        f_pl_V = self.f_pl_value(pl_padding.permute(0, 2, 1))
        f_pl_att = nn.Softmax(dim=-1)((torch.bmm(f_pl_Q, f_pl_K.permute(0, 2, 1))) * self.f_pl_norm)
        pl_FVSA = torch.bmm(f_pl_att, f_pl_V).permute(0, 2, 1)

        pl_TVSA_W = self.plTW(pl_TVSA)
        pl_FVSA_W = self.plFW(pl_FVSA)
        pl_TFW = torch.bmm(pl_TVSA_W.permute(0, 2, 1), pl_FVSA_W)
        pl_TFWA = self.plWoven(torch.reshape(pl_TFW, [pl_TFW.shape[0], 10000]))


        outputs = torch.cat((mrc_CLS, co_CLS, MASK, mrc_TFWA, co_TFWA, pl_TFWA), 1)
        predicts = self.fc(outputs)

        return predicts
