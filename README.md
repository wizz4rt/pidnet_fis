# PIDNet for Facial Image Segmentation
This project is based on the work in [this repository](https://github.com/XuJiacong/PIDNet).
Please check it out for in depth information on the architecture.
##  How to setup?

#### Load Repository, Dependendencies and Model
-  clone repository: [https://github.com/wizz4rt/pidnet_fis](https://github.com/wizz4rt/pidnet_fis)
-   install conda or mamba
-  install dependencies: python: 3.8, pytorch: 1.13, cuda: 11.6
-   download pre-trained model (ImageNet): [https://github.com/XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet)
-   place in ./pretrained_models/imagenet/
#### Load Dataset into RAM
-   sudo mkdir /mnt/ramdisk
-   sudo mount -t tmpfs -o size=40G tmpfs /mnt/ramdisk
-   sudo cp -r /mnt/hls-nas/EESFM/Zuarbeit_Abschlussarbeiten/masterarbeit_jan-niclas_nutt/datasets/mydataset/ /mnt/ramdisk

#### Link RAM to Repository

-   ln -s /mnt/ramdisk/mydataset ./data/facial

##  How to start a training?

-   specify the desired hyperparameters in a config file (.yaml) in ./config/ (template available)
-   specify the path to your config file in ./start_training.sh
-   nohup ./start_training.sh > ./terminal_log/train_.txt 2>&1 &

##   How to monitor a training?

### In Terminal Log

-   terminal log can be seen in ./terminal_log/train_.txt
-   or in './output/facial/...'
### In Graphs
-   run tensorboard on vm: nohup tensorboard --logdir=log > /dev/null 2>&1 &
-   forward port to local machine (execute on local machine): ssh 6006:localhost:6006 @
-   visit localhost:6006 in your local browser
## How to evaluate a model?

-   copy used config file to ./evaluate/
-   change TEST_SET parameter to 'list/facial/test.lst'
-   specify path to new config file in ./eval_model.sh
-   specify path to trained model ('./output/facial/.../best.pt') in ./eval_model.sh
-   nohup ./eval_model.sh > ./terminal_log/eval_.txt 2>&1 &

## How to test the model on sample images?

-   place sample images in ./samples/
-   specify path to model in ./test_model.sh
-   ./test_model.sh
-   view label maps in ./samples/outputs/

## Important Notes

-   number of classes is important when generating label maps
-   color coding of input label maps need to be 0,1,2,...,n-1 (ignore label: 255)
