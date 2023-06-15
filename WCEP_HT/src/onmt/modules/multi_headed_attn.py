""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

# from onmt.utils.misc import aeq

import numpy as np
# from onmt.modules.adaptive_span import AdaptiveSpan


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

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        # self.adapt_span_enabled = adapt_span_enabled

        # if self.adapt_span_enabled:
        #     self.adaptive_span = AdaptiveSpan(attn_span=1024, nb_heads=head_count, adapt_span_loss=2e-06,
        #                                       adapt_span_ramp=32, adapt_span_init=0, adapt_span_cache=True)

        self.unique_parsing_tags = []
        self.unique_parsing_dependencies = []

        self.tags_list = ['VB', '', 'MD', 'RBS', 'WP$', 'JJS', 'SYM', '-RRB-', '.', 'NNP', 'PDT', '$', '-LRB-', ':',
                          'DT', 'AFX', 'LS', 'WDT', 'NN', 'VBG', 'VBZ', 'EX', 'CD', 'XX', 'NFP', 'FW', 'NNS', 'HYPH',
                          'POS', 'PRP', 'TO', 'ADD', 'CC', 'UH', 'VBP', 'JJ', ',', 'RB', '``', 'PRP$', 'NNPS', 'RBR',
                          "''", 'RP', 'VBD', 'WP', 'IN', 'VBN', 'WRB', 'JJR']
        self.dependencies_list = ['nsubjpass', 'nsubj', 'acomp', 'parataxis', 'cop', 'number', 'prep', 'preconj',
                                  'conj', 'npadvmod', 'partmod', 'neg', 'xcomp', 'predet', 'csubj', 'mark', 'det',
                                  'amod', 'dobj', 'advcl', 'appos', 'nn', 'ccomp', 'possessive', 'dep', 'punct', 'cc',
                                  'root', 'pcomp', 'pobj', 'iobj', 'quantmod', 'tmod', 'advmod', 'poss', 'expl',
                                  'auxpass', 'infmod', 'rcmod', 'prt', 'aux', 'mwe', 'num', 'discourse', 'csubjpass']


        # self.dependencies_list = ['root', 'nsubjpass', 'nsubj', 'acomp', 'parataxis', 'cop', 'number', 'prep', 'preconj',
        #                           'conj', 'npadvmod', 'partmod', 'neg', 'xcomp', 'predet', 'csubj', 'mark', 'det',
        #                           'amod', 'dobj', 'advcl', 'appos', 'nn', 'ccomp', 'possessive', 'dep', 'punct', 'cc',
        #                            'pcomp', 'pobj', 'iobj', 'quantmod', 'tmod', 'advmod', 'poss', 'expl',
        #                           'auxpass', 'infmod', 'rcmod', 'prt', 'aux', 'mwe', 'num', 'discourse', 'csubjpass']

        self.dep_linear_l1 = nn.Linear(len(self.dependencies_list), 16)
        self.dep_linear_relu = nn.LeakyReLU()
        self.dep_linear_l2 = nn.Linear(16, 1)

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

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)


        # parsing_info = None  # todo: switch to None for baseline; otherwise, comment it

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

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # # deal with parsing info
        # if parsing_info:
        #     matrix_list = []
        #     for batch_item in parsing_info:
        #         sent_parsing_matrix = np.zeros((key_len, key_len))
        #         start_cor = 0
        #         for sent in batch_item:
        #             # if len(sent) == 0:
        #             #     continue
        #
        #             # get the number of tags
        #             # self.unique_parsing_dependencies = list(set(self.unique_parsing_dependencies + sent['predicted_dependencies']))
        #             # print('len(unique_parsing_dependencies):', len(self.unique_parsing_dependencies))
        #
        #             for dep in sent['dependencies']:
        #                 if start_cor + dep[1] > key_len or start_cor + dep[2] > key_len:
        #                     continue
        #
        #                 if dep[1] != 0:
        #                     # 1/n variant
        #                     n = 1
        #                     n = self.dependencies_list.index(dep[0]) + 1
        #                     # n = 45/n
        #                     sent_parsing_matrix[start_cor + dep[1] - 1][start_cor + dep[2] - 1] = 1 / n
        #                     sent_parsing_matrix[start_cor + dep[2] - 1][start_cor + dep[1] - 1] = 1 / n
        #                 else:
        #                     sent_parsing_matrix[start_cor + dep[2] - 1][start_cor + dep[2] - 1] = 1
        #             start_cor += len(sent['dependencies'])
        #         matrix_list.append(
        #             torch.from_numpy(np.repeat(sent_parsing_matrix[np.newaxis, :, :], repeats=head_count, axis=0)))
        #
        #     parsing_matrix = torch.stack(matrix_list).float().to(key.device)
        #     # scores += parsing_matrix
        #     # scores = parsing_matrix * scores + scores

        # deal with parsing info with two layers
        if parsing_info:
            onehot_matrices = []
            for batch_item in parsing_info:
                sent_dep_onehot = torch.zeros((key_len, key_len, len(self.dependencies_list)))
                start_cor = 0
                for sent in batch_item:
                    for dep in sent['dependencies']:
                        if start_cor + dep[1] > key_len or start_cor + dep[2] > key_len:
                            continue

                        dep_idx = self.dependencies_list.index(dep[0])
                        if dep[1] != 0:
                            sent_dep_onehot[start_cor + dep[1] - 1][start_cor + dep[2] - 1][dep_idx] = 1
                            sent_dep_onehot[start_cor + dep[2] - 1][start_cor + dep[1] - 1][dep_idx] = 1
                        else:
                            sent_dep_onehot[start_cor + dep[2] - 1][start_cor + dep[2] - 1][dep_idx] = 1
                    start_cor += len(sent['dependencies'])
                # linear multiplication to transfer parsing_matrix from [8,8,500,500,45] to [8,8,500,500]
                parsing_sent = self.dep_linear_l1(sent_dep_onehot.to(key.device))
                parsing_sent = self.dep_linear_relu(parsing_sent)
                parsing_sent = self.dep_linear_l2(parsing_sent).squeeze()
                onehot_matrices.append(parsing_sent.repeat(head_count, 1, 1))
            parsing_matrix = torch.stack(onehot_matrices).to(key.device)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)

        # todo: add relative position embedding here
        # if self.adapt_span_enabled:
        #     # trim attention lengths according to the learned span
        #     attn = self.adaptive_span(attn)


        if parsing_info:
            attn = parsing_matrix * attn * 1 + attn
            # attn = parsing_matrix * parsing_matrix * attn + attn
            # attn = (1-parsing_matrix * attn) * (1-parsing_matrix * attn) / 0.25 + attn
            # attn = parsing_matrix * 0.25 + attn
            # attn = (1 - parsing_matrix * attn # # deal with parsing info
            drop_attn = attn
            # add parsing_infor in value
            # value = torch.matmul(parsing_matrix, value) + value
        # elif self.adapt_span_enabled:
        #     drop_attn = attn
        else:
            drop_attn = self.dropout(attn)

        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn
