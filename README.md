# Large Kernel Modulation Network for Efficient Image Super-Resolution

## Environment in our experiments
[python 3.8]

[Ubuntu 20.04]

[BasicSR 1.4.2](https://github.com/XPixelGroup/BasicSR)

[PyTorch 1.13.0, Torchvision 0.14.0, Cuda 11.7](https://pytorch.org/get-started/previous-versions/)

### Installation
```
git clone https://github.com/Supereeeee/LKMN.git
pip install -r requirements.txt
python setup.py develop
```

## How To Test
· Refer to ./options/test for the configuration file of the model to be tested and prepare the testing data.  

· The pre-trained models have been palced in ./experiments/pretrained_models/  

· Then run the follwing codes for testing:  

```
python basicsr/test.py -opt options/test/test_LKMN_x2.yml
python basicsr/test.py -opt options/test/test_LKMN_x3.yml
python basicsr/test.py -opt options/test/test_LKMN_x4.yml
```
The testing results will be saved in the ./results folder.

## How To Train
· Refer to ./options/train for the configuration file of the model to train.  

· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  

· Note that the default training dataset is based on lmdb, refer to [docs in BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to learn how to generate the training datasets.  

· The training command is like following:
```
python basicsr/train.py -opt options/train/train_LKMN_x2.yml
python basicsr/train.py -opt options/train/train_LKMN_x3.yml
python basicsr/train.py -opt options/train/train_LKMN_x4.yml
```
For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  


## Inference and latency
· You can run ./inference/main_inference.py to obtain SR results with your own figures (LR only).

· You can run ./inference/main_time.py on your decive to test the inference time.


## Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

## Contact
If you have any question, please email quanwei1277@163.com.
