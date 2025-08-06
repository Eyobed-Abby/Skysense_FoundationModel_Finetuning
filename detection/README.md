# SkySense for Horizontal Object Detection

## Usage

### Setup Environment
Please install the [mmdet 2.28.2](https://mmdetection.readthedocs.io/en/v2.28.2/get_started.html) first.

You can either install it as a Python package, please refer to [Get Started](https://mmdetection.readthedocs.io/en/v2.28.2/get_started.html#installation) to install MMdetection 2.28.2 by other means.

### Data Preparation

We use the DIOR dataset as an example.

Then the dataset structure should be, you can also prepare any datasets following the same directory structure:
```
data_root/
├── trainval.json
│── test.json
|
├── JPEGImages-trainval/
│   ├── xxx.jpg
│   ├── xxx.jpg
├── JPEGImages-test/
│   ├── xxx.jpg
│   ├── xxx.jpg
...
```

### Set up the path of pretrained weights 

Please refer to *SkySense/README.md* first to complete the weight conversion process in order to extract the appropriate weight components.

Edit the #8 line of the config file: *configs/swin_transformer_v2/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py* to set up the path to the pre-trained weights.

### Training commands

Set the appropriate number of GPUs in *run_train.sh* according to the device conditions.

```bash
sh run_train.sh
```

### Results

|    Dataset     |  Metric  | Performance (%) | Config/Code |
|:---------------------------|:-------:|:-------:|:------:|
|  DIOR                      |   mAP50  |   78.73 | [Config](SkySense/detection/configs/swin_transformer_v2/faster_rcnn_swinv2_huge_patch4_window8_fpn-1x_dior.py)       |

Note: All results were obtained using NVIDIA A100 GPUs (80GB).