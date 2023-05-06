from mmdet.apis import init_detector, inference_detector
import os

# Paths
work_path = '/mmdetection/inference'
checkpoint_path = f'{work_path}/nn_checkpoint'
img_path = f'{work_path}/image'
output_path = f'{work_path}/output'

# Get the nn_checkpoint loaded as volume 
nn_checkpoint_name = os.listdir(checkpoint_path)[0]
checkpoint_file = f'{checkpoint_path}/{nn_checkpoint_name}'
config_file = '/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file)

# Get the image loaded as volume
img_name = os.listdir(img_path)[0]
img_file = f'{img_path}/{img_name}'

# Test a single image and show the results
result = inference_detector(model, img_path)
model.show_result(img_file, result, out_file=f'{output_path}/output.png')