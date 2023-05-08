from mmdet.apis import init_detector, inference_detector
import os

def GetFileExtension(file_path):
   return os.path.splitext(file_path)[1]

def GetFileNameNoExtension(file_path):
   return os.path.splitext(file_path)[0].split('/')[0]

# Version 1
# Searches for all folders and files in /mmdetection/configs for a file that matches
# the name of the checkpoint file, ensuring that the adecuate configuration is used
# for the given NN.
# Works if the checkpoint file has the exact same name as the confi file
def GetConfigFile1(configs_path, nn_checkpoint_name):
  # Splits the file name in name and extension
  nn_checkpoint_name = GetFileNameNoExtension(nn_checkpoint_name)

  match_path = ""
  found_match = False
  # Iterates through all files in config folder
  for folder in os.listdir(configs_path):
      folder_path = f'{configs_path}/{folder}'
      # Continues if it is a folder that is not _user_
      if os.path.isdir(folder_path) and folder != '_user_':
        # Iterates through all the files in the folder
        for file_name in os.listdir(folder_path):
          # Splits the file name in name and extension
          file_name_short, file_name_ext = os.path.splitext(file_name)
          # Continues if it is a python file whose entire name is in the checkpoint's name
          if file_name_ext == ".py" and file_name_short == nn_checkpoint_name:
              match_path = f'{folder_path}/{file_name}'
              found_match = True
              break
        
        if found_match:
           break
  
  return match_path;

def BuildPartialName(splitted_name, last_index):
  builded_name = ""
  for idy in range(last_index + 1):
    builded_name += splitted_name[idy]

    if idy != last_index:
      builded_name += "_"
  
  return builded_name

# Version 2
# Searches for all folders and files in /mmdetection/configs for a file that matches
# the name of the checkpoint file, ensuring that the adecuate configuration is used
# for the given NN. It breaks down the name by the delimeter "_" and searches for matches up to
# a given part of the name.
# Works even when there is some identification data after the name of the NN in the checkpoint
# file name. Drawback is that it is not so precise, since it chooses the last match of the last 
# builded name. In some cases this may not be the correct configuration
def GetConfigFile2(configs_path, nn_checkpoint_name):
  # Splits the file name in name and extension
  nn_checkpoint_name = GetFileNameNoExtension(nn_checkpoint_name)

  splitted_checkpoint_name = nn_checkpoint_name.split('_')

  last_match_path = ""

  for idx, name_part in enumerate(splitted_checkpoint_name):
    found_match = False

    builded_name = BuildPartialName(splitted_checkpoint_name, idx)

    # Iterates through all files in config folder
    for folder in os.listdir(configs_path):
        folder_path = f'{configs_path}/{folder}'
        # Continues if it is a folder that is not _user_
        if os.path.isdir(folder_path) and folder != '_user_':
          # Iterates through all the files in the folder
          for file_name in os.listdir(folder_path):
            # Splits the file name in name and extension
            file_name_short, file_name_ext = os.path.splitext(file_name)
            # Continues if it is a python file whose entire name is in the checkpoint's name
            if file_name_ext == ".py" and builded_name in file_name_short:
                last_match_path = f'{folder_path}/{file_name}'
                found_match = True
                break
          
          if found_match:
             break
    
    if found_match:
       continue
    else:    
      break
  
  return(last_match_path)

def main():
  # Paths
  work_path = '/mmdetection/inference'
  configs_path = '/mmdetection/configs'
  checkpoint_path = f'{work_path}/nn-checkpoint'
  img_path = f'{work_path}/image'
  output_path = f'{work_path}/output'

  # Get the nn_checkpoint loaded as volume 
  nn_checkpoint_name = os.listdir(checkpoint_path)[0]
  checkpoint_file = f'{checkpoint_path}/{nn_checkpoint_name}'

  # Get the config file by extracting the name of the NN from the checkpoint filename
  config_file = GetConfigFile1(configs_path, nn_checkpoint_name)
  if not config_file:
    config_file = GetConfigFile2(configs_path, nn_checkpoint_name)
  
  print("Selected Config = " + config_file)

  # Build the model from a config file and a checkpoint file
  model = init_detector(config_file, checkpoint_file)

  # Get the image loaded as volume
  img_name = os.listdir(img_path)[0]
  img_file = f'{img_path}/{img_name}'

  # Test a single image and show the results
  result = inference_detector(model, img_file)

  output_file_name = f'{GetFileNameNoExtension(img_name)}_{GetFileNameNoExtension(config_file)}.png' 
  print(output_file_name)

  model.show_result(img_file, result, out_file=f'{output_path}/{output_file_name}')

main()