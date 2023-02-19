# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

# UPDATE:
# @Time   : 2020/8/24 14:58, 2020/9/16, 2020/9/21, 2020/10/9
# @Author : Yujie Lu, Xingyu Pan, Zhichao Feng, Hui Wang
# @Email  : yujielu1998@gmail.com, panxy@ruc.edu.cn, fzcbupt@gmail.com, hui.wang@ruc.edu.cn

"""
recbole.model.layers
#############################
Common Layers in recommender system
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn

class FilterMixerLayer(nn.Module):
    def __init__(self, hidden_size, i, config):
        super(FilterMixerLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.config = config
        self.filter_mixer = config['filter_mixer']
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']

        self.complex_weight = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        if self.filter_mixer == 'G':
            self.complex_weight_G = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        elif self.filter_mixer == 'L':
            self.complex_weight_L = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        elif self.filter_mixer == 'M':
            self.complex_weight_G = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
            self.complex_weight_L = nn.Parameter(torch.randn(1, self.max_item_list_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)

        self.out_dropout = nn.Dropout(config['attn_dropout_prob'])
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.n_layers = config['n_layers']

        self.dynamic_ratio = config['dynamic_ratio']
        self.slide_step = ((self.max_item_list_length // 2 + 1) * (1 - self.dynamic_ratio)) // (self.n_layers - 1)

        self.static_ratio = 1 / self.n_layers
        self.filter_size = self.static_ratio * (self.max_item_list_length // 2 + 1)

        self.slide_mode = config['slide_mode']
        if self.slide_mode == 'one':
            G_i = i
            L_i = self.n_layers - 1 - i
        elif self.slide_mode == 'two':
            G_i = self.n_layers - 1 - i
            L_i = i
        elif self.slide_mode == 'three':
            G_i = self.n_layers - 1 - i
            L_i = self.n_layers - 1 - i
        elif self.slide_mode == 'four':
            G_i = i
            L_i = i
        # print("slide_mode:", self.slide_mode, len(self.slide_mode), type(self.slide_mode))


        if self.filter_mixer == 'G' or self.filter_mixer == 'M':
            self.w = self.dynamic_ratio
            self.s = self.slide_step
            if self.filter_mixer == 'M':
                self.G_left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
                self.G_right = int((self.max_item_list_length // 2 + 1) - G_i * self.s)
            self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
            self.right = int((self.max_item_list_length // 2 + 1) - G_i * self.s)


        if self.filter_mixer == 'L' or self.filter_mixer == 'M':
            self.w = self.static_ratio
            self.s = self.filter_size
            if self.filter_mixer == 'M':
                self.L_left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (L_i * self.s))
                self.L_right = int((self.max_item_list_length // 2 + 1) - L_i * self.s)

            self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (L_i * self.s))
            self.right = int((self.max_item_list_length // 2 + 1) - L_i * self.s)
            print("====================================================================================G_left, right",
                  self.G_left, self.G_right, self.G_right - self.G_left)
            print("====================================================================================L_left, Light",
                  self.L_left, self.L_right, self.L_right - self.L_left)

    def forward(self, input_tensor):
        # print("input_tensor", input_tensor.shape)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        if self.filter_mixer == 'M':
            weight_g = torch.view_as_complex(self.complex_weight_G)
            weight_l = torch.view_as_complex(self.complex_weight_L)
            G_x = x
            L_x = x.clone()
            G_x[:, :self.G_left, :] = 0
            G_x[:, self.G_right:, :] = 0
            output = G_x * weight_g

            L_x[:, :self.L_left, :] = 0
            L_x[:, self.L_right:, :] = 0
            output += L_x * weight_l


        else:
            weight = torch.view_as_complex(self.complex_weight)
            x[:, :self.left, :] = 0
            x[:, self.right:, :] = 0
            output = x * weight

        sequence_emb_fft = torch.fft.irfft(output, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)

        if self.config['residual']:
            origianl_out = self.LayerNorm(hidden_states + input_tensor)
        else:
            origianl_out = self.LayerNorm(hidden_states)

        return origianl_out


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)  # ori


    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor, ori_x):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.config['dense']:
            hidden_states = self.LayerNorm(hidden_states + input_tensor + ori_x)
        elif self.config['residual']:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

class FMBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_dropout_prob,
                 hidden_act,
                 layer_norm_eps,
                 i,
                 config,
                 ) -> None:
        super().__init__()
        self.intermediate = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps, config)
        self.filter_mixer_layer = FilterMixerLayer(hidden_size, i, config)


    def forward(self, x):
        out = self.filter_mixer_layer(x)
        out = self.intermediate(out, x)
        return out


class Encoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
            self,
            n_layers=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            inner_skip_type='straight',
            outer_skip_type='straight',
            simgcl_lambda=0,
            inner_wide=False,
            outer_wide=False,
            add_detach=False,
            fine_grained=26,
            learnable=False,
            config=None,
    ):

        super(Encoder, self).__init__()

        self.outer_skip_type = outer_skip_type
        self.simgcl_lambda = simgcl_lambda

        self.n_layers = config['n_layers']

        self.layer = nn.ModuleList()
        for n in range(self.n_layers):
            self.fmblock = FMBlock(
                hidden_size,
                inner_size,
                hidden_dropout_prob,
                hidden_act,
                layer_norm_eps,
                n,
                config)
            self.layer.append(self.fmblock)



    def forward(self, hidden_states, output_all_encoded_layers):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """

        all_encoder_layers = []
        for layer_module in self.layer:
            if self.training:
                # simgcl
                # print("hidden_states:!!!!!!!!!!!!!!!!!!!!", layer_module)
                random_noise = torch.FloatTensor(hidden_states.shape).uniform_(0, 1).to('cuda')
                hidden_states += self.simgcl_lambda * torch.multiply(torch.sign(hidden_states),
                                                                     torch.nn.functional.normalize(random_noise,
                                                                                                   p=2, dim=1))
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers