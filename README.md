# WeedNet-R

A impoved weed detection model based on RetinaNet.

## Installation

1) Clone this repo
2) Install environment

   ```shell
   cd code
   pip install -r requirements.txt
   ```

## Training

### dataset preparation

1. download sugarbeets2016 dataset from [dataset](https://pan.baidu.com/s/1MGoRRnL9kTcRMS4SWV76TQ "extraction code code：zr06")   extraction code：zr06
2. Unpack the dataset to your path
3. run `tocsv.py` under `./dataset` directory
4. Copy the generated train.csv, test.csv, and val.csv files to `./code/dataset/`

### start train with train.py script

```
python train.py --dataset csv --csv_train <path/to/train_annots.csv>  --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv>
```



## Pre-trained model

A pre-trained model is available at: [WeedNet-R pretrain model](https://pan.baidu.com/s/14dB-7mKGTkCu5TZkBGimdQ "psw：k3xf") psw：k3xf

## Validation

run the following script to validate:

`python csv_validation.py --csv_annotations_path ./dataset/test.csv --model_path path/to/model.pt --images_path path/to/images_dir --class_list_path path/to/class_list.csv   (optional) iou_threshold iou_thres (0<iou_thresh<1) `

## Visualization

This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:

```
python visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```

## Model

The retinanet model uses a resnet backbone ([download link](https://pan.baidu.com/s/1tXSp3MfIGGoXWmgQi5zZhQ) psw：v4o1) You can set the depth of the resnet model using the --depth argument. Depth must be one of 18, 34, 50, 101 or 152. Note that deeper models are more accurate but are slower and use more memory.

## Acknowledgements

The original weed dataset source form [SugarBeets2016](http://www.ipb.uni-bonn.de/data/sugarbeets2016/)

The base network retinanet from [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)

## Examples
