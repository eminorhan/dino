# DINO

This is my personal copy of the [DINO](https://github.com/facebookresearch/dino) repository from Facebook AI customized for my own purposes. The code here can be used to train and evaluate image-based SSL models.

## Usage examples

### Training 
To train a DINO model with a ViT-B/14 architecture from scratch on your data, use [`train_dino.py`](https://github.com/eminorhan/dino/blob/master/train_dino.py): 
```python
python -u train_dino.py \
	--arch "vit_base" \
	--patch_size 14 \
	--batch_size_per_gpu 116 \
	--num_workers 8 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--optimizer adamw \
	--saveckp_freq 5000 \
	--print_freq 5000 \
	--output_dir OUTPUT_DIR \
	--data_path DATA_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX
```
This version uses the [`webdataset`](https://github.com/webdataset/webdataset) interface to feed the data into the model. There's a separete training file that uses the standard `torch`-`torchvision` data loading interface instead, if you'd prefer that: [`train_dino_nowds.py`](https://github.com/eminorhan/dino/blob/master/train_dino_nowds.py). 

### Linear evaluation 
To evaluate a model with the linear probing approach, use [`eval_linear.py`](https://github.com/eminorhan/dino/blob/master/eval_linear.py):
```python
python -u eval_linear.py \
	--arch vitb14 \
	--patch_size 14 \
	--pretrained_weights MODEL_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX \
	--batch_size 1024 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0005 \
	--output_dir OUTPUT_DIR \
	--train_data_path TRAIN_DATA_PATH \
	--val_data_path VAL_DATA_PATH \
	--num_labels 1000
```

### Finetuning evaluation 
To evaluate a model with the finetuning approach, use [`eval_finetune.py`](https://github.com/eminorhan/dino/blob/master/eval_finetune.py):
```python
python -u eval_finetune.py \
	--arch "vit_base" \
	--patch_size 14 \
	--pretrained_weights MODEL_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX \
	--batch_size 256 \
	--epochs 25 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir OUTPUT_DIR \
	--train_data_path TRAIN_DATA_PATH \
	--val_data_path VAL_DATA_PATH \
	--frac_retained 0.01 \
	--num_labels 1000
```
Here `frac_retained` is the fraction of the training set used for finetuning and can be set to do few-shot finetuning evals (*e.g.* `--frac_retained 0.01` corresponds to finetuning with 1% of the training data, *i.e.* 12-13 examples per class in the case of ImageNet).

### Computing and saving embeddings for a set of images 
To compute and save the embeddings of a set of images with a given model, use [`eval_outputs.py`](https://github.com/eminorhan/dino/blob/master/eval_outputs.py):
```python
python -u eval_outputs.py \
	--arch vitb14 \
	--patch_size 14 \
	--pretrained_weights MODEL_PATH \
	--save_prefix INFORMATIVE_SAVE_PREFIX \
	--batch_size 256 \
	--num_workers 8 \
	--output_dir OUTPUT_DIR \
	--val_data_path IMAGES_PATH
```

### Visualizing class-conditional attention maps for ResNeXt models
To visualize class-conditional attention maps for the ResNeXt models, use [`visualize_resnext.py`](https://github.com/eminorhan/dino/blob/master/visualize_resnext.py):
```python
python -u visualize_resnext.py \
	--data_path IMAGES_PATH \
	--class_idx 2 \
	--n_out 26 \
	--batch_size 4 \
	--pretrained_backbone BACKBONE_PATH \
	--pretrained_fc FC_PATH
```
Here, `data_path` is the path to the images for which we compute the attention maps, `class_idx` is the index of the class with respect to which we compute the attention maps, `n_out` is the total number of classes in the pretrained model's output layer, `pretrained_backbone` is the path to the pretrained backbone (trained with SSL), and `pretrained_fc` is the path to the pretrained final `fc` layer of the model (trained separately as a linear probe).