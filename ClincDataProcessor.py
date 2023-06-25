# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all Conditional Generation tasks.
"""
import collections

import numpy as np
import pandas as pd
from openprompt.data_utils.utils import InputExample
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor


class ClincProcessor(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python

        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = "datasets/CondGen"

        dataset_name = "webnlg_2017"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 18025
        assert len(valid_dataset) == 18025
        assert len(test_dataset) == 4928
        assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
        assert test_dataset[0].text_b == ""
        assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."
    """

    def __init__(self, is_id_only=True):
        super().__init__()
        self.labels = None
        self.is_id_only = is_id_only

    def get_examples(self, data_dir: str, split: str, frac: int = 0) -> List[InputExample]:
        examples = []
        if split == 'train' and frac:
            path = os.path.join(data_dir, "{}_{}.csv".format(split, frac))
        else:
            path = os.path.join(data_dir, "{}.csv".format(split))
        data = pd.read_csv(path)
        intents = list(data["intent"])
        domian_index = list(data['domain_index'])
        indexs = list(data["index"])
        utts = list(data["utt"])
        id_utt = [utts[i] for i in range(len(utts)) if indexs[i] != -1]
        # id_intent = [intents[i] for i in range(len(utts)) if indexs[i] != -1]
        ood_utt = [utts[i] for i in range(len(utts)) if indexs[i] == -1]
        if self.is_id_only:
            utts = id_utt
            is_oods = [0 for _ in range(len(utts))]
            new_indexs = [indexs[i] for i in range(len(indexs)) if indexs[i] != -1]
            new_domian_index = [domian_index[i] for i in range(len(domian_index)) if domian_index[i] != -1]
        else:
            utts = id_utt + ood_utt
            is_oods = [0 for _ in range(len(id_utt))] + [1 for _ in range(len(ood_utt))]
            new_indexs = [indexs[i] for i in range(len(indexs)) if indexs[i] != -1] + [-1 for _ in range(len(ood_utt))]
            new_domian_index = [domian_index[i] for i in range(len(domian_index)) if domian_index[i] != -1] + [-1 for _
                                                                                                               in range(
                    len(ood_utt))]
            assert len(new_indexs) == len(utts)
            assert len(new_domian_index) == len(utts)
        for i, (tgt, is_ood, intent) in enumerate(zip(utts, is_oods, new_indexs)):
            example = InputExample(guid=str(i), text_a="", tgt_text=tgt, meta={"is_ood": is_ood},
                                   label=intent if is_ood == 0 else -1)  # label=intent  tgt_text="the intent is"
            examples.append(example)

        return examples

    def get_src_tgt_len_ratio(self, ):
        pass

    def get_label_words(self, data_dir):
        path = os.path.join(data_dir, "train.csv")
        data = pd.read_csv(path)
        intents = list(data["intent"])
        domian_index = list(data['index'])
        result = collections.defaultdict(str)
        for intent, index in zip(intents, domian_index):
            result[index] = intent
        result = sorted(result.items(), key=lambda x: x[0])
        result = [[b] for a, b in result]
        return result


class IMDBProcessor(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python

        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = "datasets/CondGen"

        dataset_name = "webnlg_2017"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 18025
        assert len(valid_dataset) == 18025
        assert len(test_dataset) == 4928
        assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
        assert test_dataset[0].text_b == ""
        assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."
    """

    def __init__(self, is_id_only=True, mode=None):
        super().__init__()
        self.labels = None
        self.is_id_only = is_id_only
        self.monitor_mode = 'label'

    def get_examples(self, data_dir: str, split: str, frac: int = 0) -> List[InputExample]:
        examples = []
        if split == 'train' and frac:
            path = os.path.join(data_dir, "{}_{}.csv".format(split, frac))
        else:
            path = os.path.join(data_dir, "{}.csv".format(split))
        data = pd.read_csv(path)
        intents = list(data["intent"])
        # domian_index = list(data['domain_index'])
        indexs = list(data["index"])
        utts = list(data["utt"])
        id_utt = [utts[i] for i in range(len(utts)) if indexs[i] != -1]
        # id_intent = [intents[i] for i in range(len(utts)) if indexs[i] != -1]
        ood_utt = [utts[i] for i in range(len(utts)) if indexs[i] == -1]
        if self.is_id_only:
            utts = id_utt
            is_oods = [0 for _ in range(len(utts))]
            new_indexs = [indexs[i] for i in range(len(indexs)) if indexs[i] != -1]
            # new_domian_index = [domian_index[i] for i in range(len(domian_index)) if domian_index[i] != -1]
        else:
            utts = id_utt + ood_utt
            is_oods = [0 for _ in range(len(id_utt))] + [1 for _ in range(len(ood_utt))]
            new_indexs = [indexs[i] for i in range(len(indexs)) if indexs[i] != -1] + [-1 for _ in range(len(ood_utt))]
            # new_domian_index = [domian_index[i] for i in range(len(domian_index)) if domian_index[i] != -1] + [-1 for _ in range(len(ood_utt))]
            assert len(new_indexs) == len(utts)
            # assert len(new_domian_index) == len(utts)
        if self.monitor_mode == 'label':
            monitor = new_indexs
        # elif self.monitor_mode == 'domain':
        #     monitor = new_domian_index
        for i, (tgt, is_ood, intent) in enumerate(zip(utts, is_oods, new_indexs)):
            example = InputExample(guid=str(i), text_a="", tgt_text=tgt, meta={"is_ood": is_ood},
                                   label=intent if is_ood == 0 else -1)  # label=intent  tgt_text="the intent is"
            examples.append(example)
        return examples

    def get_src_tgt_len_ratio(self, ):
        pass

    def get_label_words(self, data_dir):
        path = os.path.join(data_dir, "train.csv")
        data = pd.read_csv(path)
        intents = list(data["intent"])
        domian_index = list(data['index'])
        result = collections.defaultdict(str)
        for intent, index in zip(intents, domian_index):
            result[index] = intent
        result = sorted(result.items(), key=lambda x: x[0])
        result = [[b] for a, b in result]
        return result


class ClincProcessorWithContent(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python

        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = "datasets/CondGen"

        dataset_name = "webnlg_2017"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 18025
        assert len(valid_dataset) == 18025
        assert len(test_dataset) == 4928
        assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
        assert test_dataset[0].text_b == ""
        assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."
    """

    def __init__(self, is_id_only=True):
        super().__init__()
        self.labels = None
        self.is_id_only = is_id_only

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.csv".format(split))
        data = pd.read_csv(path)
        intents = list(data["intent"])
        indexs = list(data["index"])
        utts = list(data["utt"])
        id_utt = [utts[i] for i in range(len(utts)) if indexs[i] != -1]
        id_intent = [intents[i] for i in range(len(utts)) if indexs[i] != -1]
        ood_utt = [utts[i] for i in range(len(utts)) if indexs[i] == -1]
        if self.is_id_only:
            utts = id_utt
            is_oods = [0 for _ in range(len(utts))]
        else:
            utts = id_utt + ood_utt
            is_oods = [0 for _ in range(len(id_utt))] + [1 for _ in range(len(ood_utt))]

        for i, (tgt, is_ood) in enumerate(zip(utts, is_oods)):
            example = InputExample(guid=str(i), text_a="this intent is:", tgt_text=tgt, meta={"is_ood": is_ood})
            examples.append(example)

        return examples

    def get_src_tgt_len_ratio(self, ):
        pass

    def get_label_words(self, data_dir):
        path = os.path.join(data_dir, "train.csv")
        data = pd.read_csv(path)
        intents = list(data["intent"])
        domian_index = list(data['index'])
        result = collections.defaultdict(str)
        for intent, index in zip(intents, domian_index):
            result[index] = intent
        result = sorted(result.items(), key=lambda x: x[0])
        result = [[b] for a, b in result]
        return result


class ClincProcessorMINI(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python

        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = "datasets/CondGen"

        dataset_name = "webnlg_2017"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 18025
        assert len(valid_dataset) == 18025
        assert len(test_dataset) == 4928
        assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
        assert test_dataset[0].text_b == ""
        assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."
    """

    def __init__(self, is_id_only=True):
        super().__init__()
        self.labels = None
        self.is_id_only = is_id_only

    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.csv".format(split))
        data = pd.read_csv(path)
        intents = list(data["intent"])
        indexs = list(data["index"])
        utts = list(data["utt"])
        id_utt = [utts[i] for i in range(len(utts)) if indexs[i] != -1]
        id_intent = [intents[i] for i in range(len(utts)) if indexs[i] != -1]
        ood_utt = [utts[i] for i in range(len(utts)) if indexs[i] == -1]
        if self.is_id_only:
            utts = id_utt
            is_oods = [0 for _ in range(len(utts))]
        else:
            utts = id_utt[:100] + ood_utt[:100]
            is_oods = [0 for _ in range(100)] + [1 for _ in range(100)]

        for i, (tgt, is_ood) in enumerate(zip(utts, is_oods)):
            example = InputExample(guid=str(i), text_a="", tgt_text=tgt, meta={"is_ood": is_ood})
            examples.append(example)

        return examples

    def get_src_tgt_len_ratio(self, ):
        pass
