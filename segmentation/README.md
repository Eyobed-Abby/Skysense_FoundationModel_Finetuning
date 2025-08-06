# SkySense for Semantic Segmentation

## Usage

### Setup Environment
Please install the [mmsegmentation 0.X](https://github.com/open-mmlab/mmsegmentation/tree/0.x) first.

You can either install it as a Python package, please refer to [Get Started](https://mmsegmentation.readthedocs.io/en/0.x/get_started.html) to install MMsegmentation 0.X by other means.

### Data Preparation

We use the iSAID dataset as an example.

Then the dataset structure should be, you can also prepare any datasets following the same directory structure:
```
data_root/
├── img_dir/
│   ├── train
│   |   ├── xxx.jpg
│   |   ├── xxx.jpg
│   ├── val
│   |   ├── xxx.jpg
│   |   ├── xxx.jpg
├── ann_dir/
│   ├── train
│   |   ├── xxx.png
│   |   ├── xxx.png
│   ├── val
│   |   ├── xxx.png
│   |   ├── xxx.png
...
```
More details about the dataset structure can be found at [instruction](https://mmsegmentation.readthedocs.io/en/0.x/dataset_prepare.html).

### Set up the path of pretrained weights 

Please refer to *SkySense/README.md* first to complete the weight conversion process in order to extract the appropriate weight components.

Edit the #6 line of the config file: *configs/swin_transformer_v2/upernet_swinv2_huge_patch4_window8_896x896_80k_isaid.py.py* to set up the path to the pre-trained weights.

### Training commands

Set the appropriate number of GPUs in *run_train.sh* according to the device conditions.

```bash
sh run_train.sh
```

### Results

|    Dataset     |  Metric  | Performance (%) | Config/Code |
|:---------------------------|:-------:|:-------:|:------:|
|  iSAID                     |   mIoU  |   70.91 |  [Config](SkySense/segmentation/configs/swin_transformer_v2/upernet_swinv2_huge_patch4_window8_896x896_80k_isaid.py)      |

Note: All results were obtained using NVIDIA A100 GPUs (80GB).
