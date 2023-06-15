""" Embeddings module """
import math

import torch
import torch.nn as nn
import math

from onmt.modules.util_class import Elementwise

doc_emb_weight = 1


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)  # self.pe = pe
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        # self.div_term = div_term

        # self.complex_order = ComplexNN(n_token=50004, d_model=dim)

    def forward(self, emb, src_org, step=None):
        doc_emb = torch.zeros(src_org.shape)
        for i in range(emb.shape[1]):
            # split_idxes_tmp = (src_org[:, i, :] == 49).nonzero(as_tuple=True)[0]
            split_idxes = (src_org[:, i, :] == 49).nonzero()[:, 0]
            last_split_idx = 0
            doc_counter = 0
            for split_idx in split_idxes:
                doc_emb[last_split_idx: split_idx-1, i, :] = doc_counter
                last_split_idx = split_idx
                doc_counter += 1
            doc_emb[last_split_idx:, i, :] = doc_counter

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        # _, emb = self.complex_order.forward(emb, src_org)

        # doc positional encoding
        # emb = emb + doc_emb_weight * torch.sin(doc_emb).cuda()
        # emb = emb + doc_emb_weight * torch.sin(0.1 * (doc_emb % 2 == 0).float() * doc_emb).cuda()
        # emb = emb + doc_emb_weight * (torch.cos(0.1 * (doc_emb % 2 == 1).float() * doc_emb) - (doc_emb % 2 == 0).float()).cuda()
        # emb[:, :, -1] = emb[:, :, -1] + doc_emb_weight * torch.sin((doc_emb % 2 == 0).float() * doc_emb)[:, :, -1].cuda()
        # emb[:, :, -1] = emb[:, :, -1] + doc_emb_weight * (torch.cos((doc_emb % 2 == 1).float() * doc_emb) - (doc_emb % 2 == 0).float())[:, :, -1].cuda()

        # emb = emb + doc_emb_weight * torch.sin((doc_emb % 2 == 0).float() * doc_emb).cuda()
        # emb = emb + doc_emb_weight * torch.cos((doc_emb % 2 == 1).float() * doc_emb).cuda()

        emb = self.dropout(emb)
        return emb


class ComplexNN(nn.Module):
    def __init__(self, n_token, d_model):
        super(ComplexNN, self).__init__()
        # self.word_emb = torch.nn.Embedding(n_token, d_model)
        self.frequency_emb = torch.nn.Embedding(n_token, d_model)
        self.initial_phase_emb = torch.nn.Embedding(n_token, d_model)

    def forward(self, emb, x):
        """
        Args:
            emb: token embeddings after dict
            x: original tokens
        Returns:
        """

        amplitude = emb
        # amplitude = self.word_emb(x)
        frequency = self.frequency_emb(x.squeeze(dim=2))
        self.initial_phase_emb.weight = torch.nn.Parameter(self.initial_phase_emb.weight % (2* math.pi))

        sent_len = x.size(0)
        pos_seq = torch.arange(1, sent_len + 1, 1.0, device=amplitude.device)

        dimension_bais = self.initial_phase_emb(x.squeeze(dim=2))

        # reshape runnable
        # pos_seq = pos_seq.unsqueeze(0).unsqueeze(-1)
        # pos_seq = pos_seq.repeat([x.size(1), 1, 1])

        # pos_seq_flatten = pos_seq.view(pos_seq.shape[0] * pos_seq.shape[1], -1)
        # org_shape = frequency.shape
        # frequency_flatten = frequency.view(frequency.shape[0] * frequency.shape[1], -1)
        # enc_output_phase = torch.mul(pos_seq_flatten, frequency_flatten).view(org_shape) + dimension_bais

        # loop
        batchsize = x.size(1)
        pos_seq = pos_seq.unsqueeze(-1)
        phase_list = []
        for i in range(batchsize):
            phase_list.append(torch.mul(pos_seq, frequency[:, i, :]) + dimension_bais[:, i, :])
        enc_output_phase = torch.stack(phase_list, dim=1)

        # enc_output_phase = torch.mul(pos_seq, frequency) + dimension_bais
        enc_output_real = amplitude * torch.cos(enc_output_phase)
        enc_output_image = amplitude * torch.sin(enc_output_phase)

        return enc_output_real, enc_output_image

class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.

        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`

        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7, feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False):

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.position_encoding = position_encoding

        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        """ word look-up table """
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """ embedding look-up table """
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, source, step=None):
        """
        Computes the embeddings for words and features.

        Args:
            source (`LongTensor`): index tensor `[len x batch x nfeat]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        src_org = source  # original source
        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, src_org, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        return source
