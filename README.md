# Contextual Contrastive Loss (CCL)

**Paper: [Contrastive Loss based on Contextual Similarity for Image Classification](ccl.lucasvalem.com)**
**Authors:** [Lucas Pascotti Valem](http://www.lucasvalem.com), [Daniel Carlos Guimarães Pedronette](http://www.ic.unicamp.br/~dcarlos/), [Mohand Said Allili](http://w3.uqo.ca/allimo01/)
**In 19th International Symposium on Visual Computing ([ISVC 2024](isvc.net)), Lake Tahoe, NV, USA**

----------------------
* [Overview](#overview)
* [Running Instructions](#running-instructions)
* [Datasets](#datasets)
* [Contributing](#contributing)
* [Reference](#reference)
* [Contact](#contact)
* [Acknowledgments](#acknowledgments)
* [License](#license)

## Overview

This project is a fork of [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast). It modifies the original implementation by incorporating the [Contextual Contrastive Loss (CCL)](ccl.lucasvalem.com) as introduced in the paper *'Contrastive Loss based on Contextual Similarity for Image Classification'*. This repository currently includes the CCL loss function and main files used in the paper. It is intended to be continuously maintained and improved over time.

- **CCL Loss Function:** The CCL loss function is implemented in the `losses_ccl.py` file and is available for use under the BSD-2-Clause license. If you use it or incorporate it into your project, please [cite our work](#reference).

## Running Instructions

Below, we provide a concise step-by-step guide for executing the proposed approach.

### 0) Prepare Environment

To prepare the environment with all the dependencies, please follow the same steps of [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast).

### 1) Metric Learning Pre-training with SupCon

As an initial step, as described in the paper, it is required to run a pre-training of 10 epochs with the original SupCon loss to obtain the initial weights and other structures requires for the next step. The code to generate these output structures is not yet available in this repository. To facilitate the reproducibility of the results, all those used in the paper can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1SZWqGE1zDtS1U1SBOzX08uB__uYOgs75?usp=sharing). Below there is a explain of each:
* **splits**: These files specify the division of each split for each dataset. They contain the image path of each image in the split. For convenience, they also include the labels of each image in string format.
* **pretrained weights**: These are the weights obtained as the output of the pre-training stage.
* **ranked lists**: These refer to the neighborhood sets. They contain the top-k closest neighbors of each image in the training set.
* **labels**: These contain the label id (integer format) for each of the images in the training set.
* **features**:  These contain the feature arrays for each of the images in the training set. These features are the output of the pretrained model.

All files are `.npy`, except the **pretrained weights** that are `.pth` files.
Please note that all files refer only to training data, except the **splits** that have a split for training and testing sets.
Note that three random splits were considered: `split 0`, `split 1`, and `split 2`.
Also, for each split, four different training and testing divisions were used:
* `tr20ts80` (train = 20% of dataset; test = 80% of dataset)
* `tr40ts60` (train = 40% of dataset; test = 60% of dataset)
* `tr60ts40` (train = 60% of dataset; test = 40% of dataset)
* `tr80ts20` (train = 80% of dataset; test = 20% of dataset)

### 2) Metric Learning Training with CCL

To execute the metric learning with CCL and considering the files available in [Google Drive](https://drive.google.com/drive/folders/1SZWqGE1zDtS1U1SBOzX08uB__uYOgs75?usp=sharing), you can run it as follows:
```scala
1   python main_ccl.py \ 
2           --model "resnet18" \
3           --epochs 100 \
4           --batch_size 128 \
5           --temp 0.1 \
6           --k_loss 70 \
7           --cosine \
8           --warm \
9           --dataset "food101" \
10          --npy_file "splits/food101_tr60ts40_train_split0.npy" \
11          --load_ckpt_filename "pretrained_ckpts/pretrain_food101_tr60ts40_split0_epoch_10.pth" \
12          --rks_file "ranked_lists/food101_tr60ts40_split0_rks.npy"
13          --labels_file "labels/food101_tr60ts40_split0_labels.npy" \
14          --features_file "features/food101_tr60ts40_split0_features.npy" \
15          --ckpt_filename "output_weights_ccl"
```

In this example, we use the Food101 dataset with the first split (`split 0`), allocating 60% of the dataset for training and 40% for testing (denoted as `tr60ts40`). All parameters are consistent with those described in the original paper. Lines 10 to 14 specify the files, which are available on [Google Drive](https://drive.google.com/drive/folders/1SZWqGE1zDtS1U1SBOzX08uB__uYOgs75?usp=sharing). The output weights are saved in the `output_weights_ccl.pth` file.

### 3) Downstream Classifier Training

Following the previous example, you can train the downstream classifier and obtain the accuracy results by running the following steps:
```scala
1   python main_linear_ccl.py \
2           --model resnet18 \
3           --epochs 20 \
4           --batch_size 128 \
5           --learning_rate 5 \
6           --cosine \
7           --warm \
8           --dataset "food101" \
9           --npy_file_train "splits/food101_tr60ts40_train_split0.npy" \
10          --npy_file_test "splits/food101_tr60ts40_test_split0.npy" \
11          --ckpt "output_weights_ccl.pth"
```
Note that both the training and testing splits are specified, as the testing data is used for evaluation. The downstream classifier is trained based on the output of the last steps, provided by the file `output_weights_ccl.pth`.

## Datasets

In the paper, the following datasets were considered: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), and [MiniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet).
We do not redistribute the images from the datasets, they must be downloaded from the original source.

- **Other datasets:** Custom dataset loaders have been implemented for CIFAR-100, Food101, and MiniImageNet. If you wish to experiment with different datasets, you can modify the loader calls to incorporate your own dataset. Ensure that your custom dataset follows the same file format as our provided `.npy` files for an easier integration.

## Contributing
We appreciate suggestions, ideas and contributions. If you want to contribute, feel free to [contact us.](#contact) To report small bugs, you can use the [issue tracker](https://github.com/lucasPV/CCL/issues) provided by GitHub. This repository is intended to be continuously maintained and improved over time.

## Reference
If you use this code, please cite
 ```latex
    @inproceedings{Valem2024CCL,
      author    = {Lucas Pascotti Valem and Daniel Carlos Guimarães Pedronette and Mohand Said Allili},
      title     = {Contrastive Loss based on Contextual Similarity for Image Classification},
      booktitle = {19th International Symposium on Visual Computing (ISVC)},
      year      = {2024},
      address   = {Lake Tahoe, NV, USA},
    }
```

## Contact
**Lucas Pascotti Valem**: `lucaspascottivalem@gmail.com`
**Daniel Carlos Guimarães Pedronette**: `daniel.pedronette@unesp.br`
**Mohand Said Allili**: `mohandsaid.allili@uqo.ca`

## Acknowledgments
The authors are grateful to the São Paulo Research Foundation - FAPESP (grant \#2018/15597-6), the Brazilian National Council for Scientific and Technological Development - CNPq (grants \#313193/2023-1 and \#422667/2021-8), and Petrobras (grant \#2023/00095-3) for their financial support.

## License
This project is licensed under the BSD-2-Clause license.
