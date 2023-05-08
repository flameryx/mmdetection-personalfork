import os


# Version 1
# Searches for all folders and files in /mmdetection/configs for a file that matches
# the name of the checkpoint file, ensuring that the adecuate configuration is used
# for the given NN.
# Works if the checkpoint file has the exact same name as the confi file
def GetConfigFile1(configs_path, nn_checkpoint_name):
  # Splits the file name in name and extension
  nn_checkpoint_name = os.path.splitext(nn_checkpoint_name)[0]

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
  nn_checkpoint_name = os.path.splitext(nn_checkpoint_name)[0]

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


configs_path = '/mmdetection/configs'
nn_checkpoint_name = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
nn_checkpoint_name = 'faster_rcnn_r50_fpn_1x_coco.pth'

print("GetConfigFile1 : ")
print(GetConfigFile1(configs_path, nn_checkpoint_name))
print("GetConfigFile2 : ")
print(GetConfigFile2(configs_path, nn_checkpoint_name))