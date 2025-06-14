#!/bin/bash
# Author: Doem1997
# Note: requires `pip install hf_transfer`

################################################################################
# Uplo
################################################################################
# split large file into 20GB parts
split -b 21474836480 inat2018.tar.gz "inat2018.tar.gz.part"
# # record the MD5 checksum of the original file
# md5sum inat2018.tar.gz >original_md5.txt
# # verify the MD5 checksum after restoration
# md5sum -c original_md5.txt

# Run with huggingface-cli version 0.25.0
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli upload doem1997/rs_lt ./inat2018.tar.gz.part00 --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./inat2018.tar.gz.part01 --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./inat2018.tar.gz.part02 --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./inat2018.tar.gz.part03 --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./inat2018.tar.gz.part04 --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./inat2018.tar.gz.part05 --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./places_lt.tar.gz --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./cifar-100-python.tar.gz --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./dota.tar --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./fmow_pretrain.pth --repo-type dataset
huggingface-cli upload doem1997/rs_lt ./fusrs.tar.gz --repo-type dataset

################################################################################
# Download and Extract
################################################################################
# Create directories
mkdir -p ./{inat2018,places_lt,cifar-100-python,dota}

# Download commands
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download doem1997/rs_lt ./inat2018.tar.gz.part00 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./inat2018.tar.gz.part01 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./inat2018.tar.gz.part02 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./inat2018.tar.gz.part03 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./inat2018.tar.gz.part04 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./inat2018.tar.gz.part05 --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./places_lt.tar.gz --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./cifar-100-python.tar.gz --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./dota.tar --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./fmow_pretrain.pth --repo-type dataset --local-dir ./
huggingface-cli download doem1997/rs_lt ./fusrs.tar.gz --repo-type dataset --local-dir ./

# Combine split files
cat inat2018.tar.gz.part* >inat2018.tar.gz

# For each archive, first check structure then move
tar -xzf inat2018.tar.gz -C ./inat2018
tar -xzf places_lt.tar.gz -C ./places_lt
tar -xzf cifar-100-python.tar.gz -C ./cifar-100-python
tar -xf dota.tar -C ./dota

mv ./cifar-100-python/cifar-100-python/* ./cifar-100-python/
mv ./inat2018/img/* ./inat2018/
mv ./places_lt/images/* ./places_lt/

rm -rf ./cifar-100-python/cifar-100-python
rm -rf ./inat2018/img
rm -rf ./places_lt/images
