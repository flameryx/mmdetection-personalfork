from mmdet.apis import init_detector, inference_detector
import os

configs_path = '/mmdetection/configs'
nn_checkpoint_name = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

nn_checkpoint_name = os.path.splitext(nn_checkpoint_name).stem
print(nn_checkpoint_name)

for folder in os.listdir(configs_path):
    if os.path.isdir(f'{configs_path}/{folder}'):
      for file_name in folder:
         file_name_short = os.path.splitext(nn_checkpoint_name).stem
         if file_name_short in nn_checkpoint_name:
            print(f'{configs_path}/{folder}/{file_name}')