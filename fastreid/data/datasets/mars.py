from __future__ import division, print_function, absolute_import
import glob
from operator import le
import os.path as osp
from re import T
from sys import set_asyncgen_hooks
import os


from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.utils.file_io import PathManager
from .bases import ImageDataset


@DATASET_REGISTRY.register()
class MARS(ImageDataset):
    '''"
    "'''
    dataset_dir = 'mars'
    dataset_name = "mars"
    def __init__(self, root='datasets', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        required_files = [self.dataset_dir]
        self.check_before_run(required_files)
        train, query, gallery = self.get_data_list()

        super(MARS, self).__init__(train, query, gallery, **kwargs)

    def get_data_list(self):
        train, query, gallery = [], [], []
        pid_container = set()
        pid_container_test = set()
        for bbox in os.listdir(self.dataset_dir):
            #print(bbox)
            if bbox == 'bbox_train':
                bbox = osp.join(self.dataset_dir,bbox)
                for pid in os.listdir(bbox):
                    id = int(pid)
                    pid_container.add(id)
                    pid2label = {
                        pid : label for label,pid in enumerate(pid_container)
                    }
                for pid in os.listdir(bbox):
                    id = int (pid)
                    pid = osp.join(bbox,pid)
                    s = 0
                    for img in os.listdir(pid):
                        if s%100 == 0:
                            img_path = osp.abspath(osp.join(pid, img))
                            camid = int(img[5])-1
                            pid_ = self.dataset_name + '_' + str(pid2label[id])
                            camid = self.dataset_name + '_' + str(camid)
                            #print(img_path)
                            train.append((img_path, pid_ ,camid))
                        s = s+1
            else:
                bbox = osp.join(self.dataset_dir,bbox)
                for pid in os.listdir(bbox):
                    id = int(pid) 
                    pid = osp.join(bbox,pid)
                    s = 0
                    for img in os.listdir(pid):
                        if s%100 ==0:
                            img_path = osp.abspath(osp.join(pid, img))
                            camid = int(img[5])-1
                            if camid <3:
                                query.append((img_path,id,camid))
                            else:
                                gallery.append((img_path,id,camid))
                        s = s+1
        return train, query, gallery
