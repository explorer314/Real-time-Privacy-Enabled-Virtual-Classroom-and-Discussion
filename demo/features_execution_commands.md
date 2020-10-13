
# Background Removal

```
python3 customize_video_background_with_person_segment.py \
-m <path to model>/instance-segmentation-security-0083.xml \
--label coco_labels.txt  \
--delay 1 \
-i input/input_video.mov \
--function 1
```


# Set custom Background 
```
python3 customize_video_background_with_person_segment.py \
-m <path to model>/instance-segmentation-security-0083.xml \
--label coco_labels.txt  \
--delay 1 \
-i input/input_video.mov \
--custom_image input/background_image.jpeg
--function 2
```
### Note: pass required custom image as an argument to set video background (--custom_image <path to img>)
# Background blur
```
python3 customize_video_background_with_person_segment.py \
-m <path to model>/instance-segmentation-security-0083.xml\
--label coco_labels.txt  \
--delay 1 \
-i input/input_video.mov \
--function 3
```
