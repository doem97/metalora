import yaml

from .lt_data import LT_Dataset


class FUSRSv17(LT_Dataset):
    classnames_txt = "./datasets/FUSRSv17/classnames.txt"
    train_txt = "./datasets/FUSRSv17/train.txt"
    test_txt = "./datasets/FUSRSv17/test.txt"
    split_config = "./datasets/FUSRSv17/lt_split_config.yaml"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        # 加载类别划分配置
        with open(self.split_config) as f:
            split_cfg = yaml.safe_load(f)
            self.many_classes = split_cfg.get("many_classes", []) or []
            self.med_classes = split_cfg.get("med_classes", []) or []
            self.few_classes = split_cfg.get("few_classes", []) or []

        self.classnames = self.read_classnames()

        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(line.split()[1])])

        self.many_idxs = [self.classnames.index(c) for c in self.many_classes]
        self.med_idxs = [self.classnames.index(c) for c in self.med_classes]
        self.few_idxs = [self.classnames.index(c) for c in self.few_classes]

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                classnames.append(line)
        return classnames


class FUSRSv25(FUSRSv17):
    classnames_txt = "./datasets/FUSRSv25/classnames.txt"
    train_txt = "./datasets/FUSRSv25/train.txt"
    test_txt = "./datasets/FUSRSv25/test.txt"
    split_config = "./datasets/FUSRSv25/lt_split_config.yaml"
