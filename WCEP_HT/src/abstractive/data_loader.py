import gc
import glob
import random
import torch
from others.logging import logger

import json

# dataset_processed = []
# last_corpus_type = None

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def clean_str(s):
    s = s.replace('<unk>', '')
    s = s.replace('`', '')
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace(';', '')
    s = s.replace('\'', '')
    s = s.replace('\"', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace('-', ' ')
    s = s.replace('<p>', '')
    s = s.replace('</p>', '')
    s = s.replace('<t>', '')
    s = s.replace('</t>', '')
    s = s.replace('[!@#$]', '')
    s = s.replace('–', '')
    s = s.replace('—', '')
    s = s.replace('―', '')
    s = s.replace('"', '')
    return s


class AbstractiveBatch(object):
    def _pad(self, data, height, width, pad_id):
        """ ? """
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_length = [len(d) for d in data]
        rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))

        return rtn_data, rtn_length

    def __init__(self, data=None, hier=False, pad_id=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            src = [x[0] for x in data]
            tgt = [x[1] for x in data]
            parsing_info = [x[3] for x in data]

            if hier:
                max_nblock = max([len(e) for e in src])
                max_ntoken = max([max([len(p) for p in e]) for e in src])
                _src = [self._pad(e, max_nblock, max_ntoken, pad_id) for e in src]
                src = torch.stack([torch.tensor(e[0]) for e in _src])

            else:
                _src = self._pad(src, width=max([len(d) for d in src]), height=len(src), pad_id=pad_id)
                src = torch.tensor(_src[0])  # batch_size, src_len

            setattr(self, 'src', src.to(device))

            _tgt = self._pad(tgt, width=max([len(d) for d in tgt]), height=len(tgt), pad_id=pad_id)
            tgt = torch.tensor(_tgt[0]).transpose(0, 1)
            setattr(self, 'tgt', tgt.to(device))

            setattr(self, 'parsing_info', parsing_info)

            if is_test:
                tgt_str = [x[2] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle, spm):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.
    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]
    # corpus_type = "train"

    # _lazy_dataset_loader_data_preprocess, split multinews dataset into pts
    def _lazy_dataset_loader_bak(json_file, corpus_type):

        json_file = '../../data/json_multinews/' + corpus_type + '.label.jsonl'
        dataset = readJson(json_file)
        # dataset = dataset[:3800]
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, json_file, len(dataset)))

        # process dataset into the format that hiersumm can handle
        dataset_processed = []
        for item in dataset:
            data_point = {}

            data_point['src'] = []
            for text_i, text in enumerate(item['text']):
                cur_sent = ''
                data_point['src'].append([])
                for sentence in text:
                    sentence = clean_str(sentence)
                    if cur_sent == '':
                        cur_sent = sentence
                    elif len((cur_sent + ' <Q> ' + sentence).split()) <= 50:
                        cur_sent = cur_sent + ' <Q> ' + sentence
                    else:
                        data_point['src'][text_i].append(cur_sent)
                        cur_sent = sentence

                data_point['src'][text_i].append(cur_sent)

            summary = ' <Q> '.join(item['summary'])
            summary = clean_str(summary)
            data_point['tgt'] = summary
            dataset_processed.append(data_point)

        # chunk the data into 3800 frames
        n = 3800
        cks = list(chunks(dataset_processed, n))
        [torch.save(item, 'MULTINEWS.' + corpus_type + '.' + str(i) + '.pt') for i, item in enumerate(cks)]
        print('done')
        print('done')

    def _lazy_dataset_loader_multinews(pt_file, corpus_type):

        dataset = torch.load(pt_file)
        # dataset = torch.load('../../data/multinews/MULTINEWS.train.3.pt')
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))

        # load dep parsing pt
        parsing_pt_file = pt_file.replace('multinews/', 'multinews/dep_parsing/') \
            .replace('MULTINEWS', 'MULTINEWS_parsing')
        dep_parsing_info = torch.load(parsing_pt_file)
        # dep_parsing_info = torch.load('../../data/multinews/dep_parsing/MULTINEWS_parsing.train.3.pt')

        # # reorganized dataset
        # reorganized_dataset = []
        # for item in dataset:
        #     reorganized_item = {}
        #     reorganized_item['src'] = []
        #     reorganized_item['tgt'] = item['tgt']
        #     reorganized_item['src'] = []
        #     for doc_set in item['src']:
        #         reorganized_doc = []
        #         for doc in doc_set:
        #             sent_set = doc.split('<Q>')
        #             for org_sent in sent_set:
        #                 sent = {}
        #                 sent['org_doc'] = org_sent
        #                 reorganized_doc.append(sent)
        #         reorganized_item['src'].append(reorganized_doc)
        #     reorganized_dataset.append(reorganized_item)
        #
        # # reorganized parsing info
        # for item in dep_parsing_info:
        #     for doc_set in item:
        #         for doc in doc_set:
        #             for parsing_info in doc:
        #                 sent['parsing_info']= parsing_info


        # process dataset into the format that hiersumm can handle
        dataset_processed = []
        for i, item in enumerate(dataset):
            data_point = {}
            data_point['src'] = []
            for doc_set in item['src']:
                for doc in doc_set:
                    data_point['src'].append(spm.encode(doc.lower().replace(' </t> <t>', '<Q>'))[:100])

            # todo: deal with special tokens
            summary = item['tgt']
            tgt = spm.encode(summary.lower().replace(' </t> <t>', '<Q>'))
            tgt.insert(0, 4)
            tgt.append(5)
            data_point['tgt'] = tgt

            data_point['tgt_str'] = summary
            data_point['cosine'] = None
            data_point['ner_graph'] = None
            parsing_list = []
            for l in dep_parsing_info[i]:
                parsing_list += l
            data_point['parsing_info'] = parsing_list
            dataset_processed.append(data_point)

        return dataset_processed

    def _lazy_dataset_loader(pt_file, corpus_type):

        dataset = torch.load(pt_file)
        # dataset = torch.load('../../data/multinews/MULTINEWS.train.3.pt')
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))

        # load dep parsing pt
        parsing_pt_file = pt_file.replace('multinews/', 'multinews/dep_parsing/') \
            .replace('MULTINEWS', 'MULTINEWS_parsing')
        dep_parsing_info = torch.load(parsing_pt_file)
        # dep_parsing_info = torch.load('../../data/multinews/dep_parsing/MULTINEWS_parsing.train.3.pt')

        # # reorganized dataset
        # reorganized_dataset = []
        # for item in dataset:
        #     reorganized_item = {}
        #     reorganized_item['src'] = []
        #     reorganized_item['tgt'] = item['tgt']
        #     reorganized_item['src'] = []
        #     for doc_set in item['src']:
        #         reorganized_doc = []
        #         for doc in doc_set:
        #             sent_set = doc.split('<Q>')
        #             for org_sent in sent_set:
        #                 sent = {}
        #                 sent['org_doc'] = org_sent
        #                 reorganized_doc.append(sent)
        #         reorganized_item['src'].append(reorganized_doc)
        #     reorganized_dataset.append(reorganized_item)
        #
        # # reorganized parsing info
        # for item in dep_parsing_info:
        #     for doc_set in item:
        #         for doc in doc_set:
        #             for parsing_info in doc:
        #                 sent['parsing_info']= parsing_info


        # process dataset into the format that hiersumm can handle
        dataset_processed = []
        for i, item in enumerate(dataset):
            data_point = {}
            data_point['src'] = []
            doc = ' '.join(item.src)
            data_point['src'].append(spm.encode(doc.lower()))

            # todo: deal with special tokens
            summary = ' '.join(item.tgt)
            tgt = spm.encode(summary.lower())
            tgt.insert(0, 4)
            tgt.append(5)
            data_point['tgt'] = tgt

            data_point['tgt_str'] = summary
            data_point['cosine'] = None
            data_point['ner_graph'] = None
            parsing_list = []
            # for l in dep_parsing_info[i]:
            #     parsing_list += l
            data_point['parsing_info'] = None
            dataset_processed.append(data_point)

        return dataset_processed

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            if pt == '../../data/multinews/MULTINEWS.train.3.pt':
                continue
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


class AbstractiveDataloader(object):
    def __init__(self, args, datasets, symbols, batch_size, device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return AbstracticeIterator(args = self.args,
            dataset=self.cur_dataset, symbols=self.symbols, batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class AbstracticeIterator(object):
    def __init__(self, args, dataset, symbols, batch_size, device=None, is_test=False, shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        # self.secondary_sort_key = lambda x: len(x[0])
        # self.secondary_sort_key = lambda x: sum([len(xi) for xi in x[0]])
        # self.prime_sort_key = lambda x: len(x[1])
        self.secondary_sort_key = lambda x: sum([len(xi) for xi in x[0]])
        self.prime_sort_key = lambda x: len(x[1])
        self._iterations_this_epoch = 0


        self.symbols = symbols

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex):

        bos_id = self.symbols['BOS']
        eos_id = self.symbols['EOS']
        eot_id = self.symbols['EOT']
        eop_id = self.symbols['EOP']
        eoq_id = self.symbols['EOQ']
        src, tgt, tgt_str, parsing_info = ex['src'], ex['tgt'], ex['tgt_str'], ex['parsing_info']
        if (not self.args.hier):
            src = sum([p + [eop_id] for p in src], [])[:-1][:self.args.trunc_src_ntoken] + [eos_id]
            return src, tgt, tgt_str, parsing_info

        return src[:self.args.trunc_src_nblock], tgt, tgt_str, parsing_info

    def simple_batch_size_fn(self, new, count):
        src, tgt = new[0], new[1]

        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
        if (self.args.hier):
            max_src_in_batch = max(max_src_in_batch, sum([len(p) for p in src]))
        else:
            max_src_in_batch = max(max_src_in_batch, len(src))
        src_elements = count * max_src_in_batch
        return src_elements

    def get_batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            ex = self.preprocess(ex)
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 100):
            if (self.args.mode != 'train'):
                p_batch = self.get_batch(
                    sorted(sorted(buffer, key=self.prime_sort_key), key=self.secondary_sort_key),
                    self.batch_size)
            else:
                p_batch = self.get_batch(
                    sorted(sorted(buffer, key=self.secondary_sort_key), key=self.prime_sort_key),
                    self.batch_size)

            p_batch = list(p_batch)

            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):

        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = AbstractiveBatch(minibatch, self.args.hier, self.symbols['PAD'], self.device, self.is_test)

                yield batch
            return

def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
