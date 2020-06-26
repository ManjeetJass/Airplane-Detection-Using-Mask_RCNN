# Airplane detection using Mask R-CNN

The purpose of this project is to demonstrate how to identify airplanes in satellite imagery using Mask R-CNN.

## Requirements
 1. Clone repo: git clone https://github.com/matterport/Mask_RCNN 
 2. Install requirements.txt dependencies of Mask RCNN
  `pip install -r requirements.txt`
 3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [release page.](https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes) 
 4. Download datasets of airplanes from [kaggle.](https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes)
 
## Test installed set-up

To check the set-up either run sample from Mask R-CNN repository or run test.py provided.
`python test.py`

![](images/sample_output.png)

## Train Mask R-CNN on airplane dataset

To identify airplanes in satellite images we need to train Mask R-CNN on planes dataset downloaded from the kaggle website.

Training of Mask R-CNN starts from the pre-trained COCO weights.

`python ./plane.py --model=coco`

To resume training from the last saved weights.

`python ./plane.py --model=last`
