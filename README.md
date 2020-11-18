# FasterRCNN

## Dependency
- python 3.7
- pytorch 1.7
- wandb 
    + for logging, set LOGGING= "" in train.py to disable logging

## File Structure 
- FasterRCNN
	+ src
	+ data
		- hw3_mycocodata_bboxes_comp_zlib.npy
		- ...
		- train_indices.npy
		- test_indices.npy
	+ pretrained
	    - checkpoint680.pth
	    - hold_out_images.npz

## Usage
- train
```bash
python train
```

- eval (visualization)
```bash
# this will save all visualized images preNMS/postNMS to ../results
python main_visualization
```

- eval (map)
```bash
python main_eval.py
```

## Google Drive
- https://drive.google.com/drive/folders/1CS7EVFLP1vCknKUhA8gAJOfFY_1SiMAB?usp=sharing

## IDE (Pycharm) Node:
- Please mark folder 'src' as source root