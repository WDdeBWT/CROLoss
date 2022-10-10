# CROLoss

Model code for paper *CROLoss: Towards a Customizable Loss for Retrieval Models in Recommender Systems*. You can find the paper in <https://arxiv.org/abs/2208.02971>.

Accepted to CIKM 2022!

# Core code

The core_code.py file provides a simple code template to show how to deploy CROLoss in your retrieval model (to replace the softmax cross-entropy loss). We hope this can help you better understand our method.

# Full code

## Prerequisites

- Python 3
- TensorFlow-GPU >= 1.8 (< 2.0)
- Faiss-GPU 

## Getting started

### Dataset

Two preprocessed datasets can be downloaded through:

Taobao: https://cloud.tsinghua.edu.cn/f/e5c4211255bc40cba828/?dl=1

Amazon book: https://www.dropbox.com/s/m41kahhhx0a5z0u/data.tar.gz?dl=1

You can also download the original datasets and preprocess them by yourself. Follow the tutorial in the [link](https://github.com/THUDM/ComiRec/blob/master/README.md#dataset).

### Training

You can use `python src/train.py --dataset {dataset_name} --kernel_type {kernel_type} --weight_type {weight_type}` to train a specific model on a dataset. Other hyperparameters can be found in the code.

For example, you can use `python src/train.py --dataset book --kernel_type sigmoid --weight_type even` to train CROLoss with sigmoid kernel and even weight on Book dataset.

# Acknowledgement

The structure of our code is based on [ComiRec](https://github.com/THUDM/ComiRec).

# Cite

To be updated.
