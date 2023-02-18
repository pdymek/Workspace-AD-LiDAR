####################################################################################
# HLD BUILDING BLOCK: DATALOADER                                                   #
####################################################################################
# TODO
# Data loading and data transformations.
# Assignment of seed according to parameter
# Assignment of torch.device according to parameter
####################################################################################

from torch.utils.data import DataLoader

class DataLoader_(DataLoader):
    #TODO
    # print("Dataloader executed!")
    def __init__(self, **kwargs):
        super(DataLoader_, self).__init__(**kwargs)