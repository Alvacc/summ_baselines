""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

visual = False


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if(self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, parsing_info=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                                   layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        parsing_info = None  # todo: switch to None for baseline; otherwise, comment it

        # deal with parsing info
        if parsing_info:
            matrix_list = []
            args_batch_size = len(parsing_info)
            n_blocks = int(batch_size / args_batch_size)
            for batch_item in parsing_info:
                for block_i in range(n_blocks):
                    sent_parsing_matrix = np.zeros((key_len, key_len))
                    if block_i < len(batch_item):
                        block_item = batch_item[block_i]
                        start_cor = 0
                        for sent in block_item:
                            if len(sent) == 0:
                                continue
                            for dep in sent['dependencies']:
                                if start_cor+dep[1] > key_len or start_cor+dep[2] > key_len:
                                    continue

                                if dep[1] != 0:
                                    sent_parsing_matrix[start_cor + dep[1] - 1][start_cor + dep[2] - 1] = 1
                                    sent_parsing_matrix[start_cor + dep[2] - 1][start_cor + dep[1] - 1] = 1
                                else:
                                    sent_parsing_matrix[start_cor + dep[2] - 1][start_cor + dep[2] - 1] = 1
                            start_cor += len(sent['dependencies'])
                    matrix_list.append(torch.from_numpy(np.repeat(sent_parsing_matrix[np.newaxis, :, :], repeats=head_count, axis=0)))

            parsing_matrix = torch.stack(matrix_list).float().to(key.device)
            # scores += parsing_matrix
            # scores = parsing_matrix * scores + scores

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        # for visualization
        if visual:
            # attention_map = attn[0][0]
            # attention_map = parsing_matrix[0][0]
            attention_map = (parsing_matrix * attn * 1 + attn)[0][0]
            # attention_map = ((1-parsing_matrix * attn) * (1-parsing_matrix * attn) / 0.25 + attn)[0][0]
            # attention_map = (parsing_matrix * 0.25 + attn)[0][0]
            heatmap_visual(parsing_info[0][0][0]['tokens'], attention_map)

        if parsing_info:
            attn = parsing_matrix * attn * 2 + attn
            # attn = parsing_matrix * parsing_matrix * attn + attn
            # attn = (1-parsing_matrix * attn) * (1-parsing_matrix * attn) / 0.25 + attn
            # attn = parsing_matrix * 0.25 + attn
            # attn = (1 - parsing_matrix * attn) * (1 - parsing_matrix * attn) / 8 + attn
            drop_attn = attn
        else:
            drop_attn = self.dropout(attn)

        if(self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn




class MultiHeadedPooling(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedPooling, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if (use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)
        self.use_final_linear = use_final_linear

    def forward(self, key, value, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x, dim=dim_per_head):
            """  projection """
            return x.view(batch_size, -1, head_count, dim) \
                .transpose(1, 2)

        def unshape(x, dim=dim_per_head):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim)

        scores = self.linear_keys(key)
        value = self.linear_values(value)

        scores = shape(scores, 1).squeeze(-1)
        value = shape(value)
        # key_len = key.size(2)
        # query_len = query.size(2)
        #
        # scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.sum((drop_attn.unsqueeze(-1) * value), -2)
        if (self.use_final_linear):
            context = unshape(context).squeeze(1)
            output = self.final_linear(context)
            return output
        else:
            return context


def heatmap_visual(vis_sent, attention_map):
    sent_len = len(vis_sent)
    vis_map = attention_map[:sent_len, :sent_len].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(vis_map, interpolation='nearest')
    fig.colorbar(cax)

    xaxis = np.arange(len(vis_sent))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(vis_sent, rotation=90)
    ax.set_yticklabels(vis_sent)

    plt.show()
