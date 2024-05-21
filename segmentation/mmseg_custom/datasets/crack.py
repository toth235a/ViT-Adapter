# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class CrackDataset(CustomDataset):
    """Crack dataset.

    
    ``reduce_zero_label`` should be set to True. 
    """
    CLASSES = ('background', 'crack')

    PALETTE = [[0, 0, 0], [244, 35, 232]]

    def __init__(self, **kwargs):
        super(PotsdamDataset, self).__init__(
            reduce_zero_label=True,
            **kwargs)
