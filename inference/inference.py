from mmdet.apis import init_detector, inference_detector

#TODO: Take the NN model name as input when starting the container
#config_file and checkpoint_file paths will depend of this name

# Specify the path to model config and checkpoint file
config_file = '/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/mmdetection/inference/nn-checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file)

#TODO: Take the image file as input when starting the container
# Test a single image and show the results
img = '/mmdetection/inference/image/dogpark.jpg'
result = inference_detector(model, img)

model.show_result(img, result, out_file='/mmdetection/inference/output/output.png')