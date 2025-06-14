from .cifar100 import CIFAR100, CIFAR100_IR10, CIFAR100_IR50, CIFAR100_IR100
from .dota import DOTA
from .imagenet_lt import ImageNet_LT
from .inat2018 import (
    iNaturalist2018,
    iNaturalist2018_Class,
    iNaturalist2018_Family,
    iNaturalist2018_Genus,
    iNaturalist2018_Kingdom,
    iNaturalist2018_Order,
    iNaturalist2018_Phylum,
    iNaturalist2018_Species,
)
from .places_lt import Places_LT
from .fusrs import FUSRSv17, FUSRSv25

__all__ = [
    "CIFAR100",
    "CIFAR100_IR10",
    "CIFAR100_IR50",
    "CIFAR100_IR100",
    "Places_LT",
    "ImageNet_LT",
    "iNaturalist2018",
    "iNaturalist2018_Kingdom",
    "iNaturalist2018_Phylum",
    "iNaturalist2018_Class",
    "iNaturalist2018_Order",
    "iNaturalist2018_Family",
    "iNaturalist2018_Genus",
    "iNaturalist2018_Species",
    "DOTA",
    "FUSRSv17",
    "FUSRSv25",
]
