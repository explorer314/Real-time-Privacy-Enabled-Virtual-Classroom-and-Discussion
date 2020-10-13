## Real time Privacy Enabled Virtual Classroom


#Real time background blurring, removal and customization with Instance Segmentation 

# Project setup

Os support - Linux, MacOs, Windows  
Install OpenVINO - https://docs.openvinotoolkit.org/latest/index.html  
```
	sudo -E ./install_openvino_dependencies.sh
	sudo ./install.sh
  ```
Enable openvino with this command:
```
source <openvino_installed_dir>/bin/setupvars.sh
```

## Download pre-trained models - [openvino models with precision fp32, fp16 and int8](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/instance-segmentation-security-0083/)


## Clone this repository 
	
## Running

Run the application with the `-h` option to see the following usage message:


```
cd Real-time-Video-background-Blurring-Removal-and-Customization-with-Deep-Learning

python3 customize_video_background_with_person_segment.py
```


```
usage: customize_video_background_with_person_segment.py [-h] -f FUNCTION -m
                                                         "<path>" --labels
                                                         "<path>" -i "<path>"
                                                         [-d "<device>"]
                                                         [-l "<absolute_path>"]
                                                         [--delay "<num>"]
                                                         [--custom_image "<path>"]
                                                         [-pt "<num>"]
                                                         [--no_keep_aspect_ratio]
                                                         [--no_track]
                                                         [--show_scores]
                                                         [--show_boxes] [-pc]
                                                         [-r] [--no_show]

Options:
  -h, --help            Show this help message and exit.
  -f FUNCTION, --function FUNCTION
                        Required. enter  
                        0 for removing background (to get black backgground)  
                        1 for removing background (to get white background)    
                        2 for changing background  
                        3 for background blurring  
  -m "<path>", --model "<path>"
                        Required. Path to an .xml file with a trained model.
  --labels "<path>"     Required. Path to a text file with class labels.
  -i "<path>"           Required. Path to an image, video file or a numeric
                        camera ID.
  -d "<device>", --device "<device>"
                        Optional. Specify the target device to infer on: CPU,
                        GPU, FPGA, HDDL or MYRIAD. The demo will look for a
                        suitable plugin for device specified (by default, it
                        is CPU).
  -l "<absolute_path>", --cpu_extension "<absolute_path>"
                        Required for CPU custom layers. Absolute path to a
                        shared library with the kernels implementation.
  --delay "<num>"       Optional. Interval in milliseconds of waiting for a
                        key to be pressed.
  --custom_image "<path>"
                        Required. Path to an custom backgroung image file.
  -pt "<num>", --prob_threshold "<num>"
                        Optional. Probability threshold for detections
                        filtering.
  --no_keep_aspect_ratio
                        Optional. Force image resize not to keep aspect ratio.
  --no_track            Optional. Disable tracking.
  --show_scores         Optional. Show detection scores.
  --show_boxes          Optional. Show bounding boxes.
  -pc, --perf_counts    Optional. Report performance counters.
  -r, --raw_output_message
                        Optional. Output inference results raw values.
  --no_show             Optional. Don't show output

```

# [commands to execute Background blur, removal and customization](https://github.com/explorer314/Real-time-Privacy-Enabled-Virtual-Classroom-and-Discussion/blob/master/demo/features_execution_commands.md) 

# [Demo results](https://github.com/explorer314/Real-time-Privacy-Enabled-Virtual-Classroom-and-Discussion/tree/master/demo)
