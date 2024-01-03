import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str
from lib.utils.string_utils import clean_string

############
# current 00000492.png of test_015_Sord_video_Q01_done is damaged and replaced by a copy of 00000491.png
############


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.tnl2k_path, 'test')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_name = sequence_name.split('/')[-1]
        # class_name = seq_name
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        # target_class = class_name
        if self.dir_type == 'one-level':
            return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4))
        elif self.dir_type == 'two-level':
            return Sequence(seq_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        subset_list = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))]
        
        # one-level directory
        if len(subset_list) > 9:
            self.dir_type = 'one-level'
            return sorted(subset_list) 
        
        # two-level directory
        self.dir_type = 'two-level'
        for x in subset_list:
            sub_sequence_list_path = os.path.join(self.base_path, x)
            for seq in os.listdir(sub_sequence_list_path):
                sequence_list.append(os.path.join(x, seq))
        sequence_list = sorted(sequence_list)

        return sequence_list


class TNL2k_LangDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.tnl2k_path, 'test')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_name = sequence_name.split('/')[-1]
        class_name = seq_name
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)
        text_dsp = clean_string(text_dsp)
        
        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        target_class = class_name
        if self.dir_type == 'one-level':
            return Sequence(sequence_name, frames_list, 'tnl2k_lang', ground_truth_rect.reshape(-1, 4),
                            text_description=text_dsp, object_class=target_class)
        elif self.dir_type == 'two-level':
            return Sequence(seq_name, frames_list, 'tnl2k_lang', ground_truth_rect.reshape(-1, 4),
                        text_description=text_dsp, object_class=target_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        subset_list = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))]
        
        # one-level directory
        if len(subset_list) > 9:
            self.dir_type = 'one-level'
            return sorted(subset_list) 
        
        # two-level directory
        self.dir_type = 'two-level'
        for x in subset_list:
            sub_sequence_list_path = os.path.join(self.base_path, x)
            for seq in os.listdir(sub_sequence_list_path):
                sequence_list.append(os.path.join(x, seq))
        sequence_list = sorted(sequence_list)

        return sequence_list
