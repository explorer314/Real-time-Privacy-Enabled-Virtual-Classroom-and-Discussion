"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 /********************************************************************
* Real-Time Video Background Removal, Background Blurring and setting Virual Video Background Application Created by explorer314.
*
********************************************************************/
"""

from __future__ import print_function

import cv2
import numpy as np


class Visualizer(object):
    color_palette = np.array([[0, 113, 188]], dtype=np.uint8)
    def __init__(self, class_labels, confidence_threshold=0.8, show_boxes=False,
                 show_masks=True, show_scores=False):
        super().__init__()
        self.class_labels = [class_labels]
        #self.class_labels = [[1]]
        self.confidence_threshold = confidence_threshold
        self.class_color_palette = np.asarray([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.instance_color_palette = self.color_palette
        self.show_masks = show_masks
        #print(show_masks)
        self.show_boxes = show_boxes
        self.show_scores = show_scores

    def __call__(self, function, c_img, image, boxes, classes, scores, segms=None, ids=None):
        result = image.copy()

        # Filter out detections with low confidence.
        filter_mask = scores > self.confidence_threshold
        scores = scores[filter_mask]
        classes = classes[filter_mask]
        #print(classes)
        boxes = boxes[filter_mask]
        #print(filter_mask)
        filter_mask = [True]
        
        if self.show_masks and segms is not None:
            segms = list(segm for segm, show in zip(segms, filter_mask) if show)
            #print(segms[0])
            result = self.overlay_masks(function, c_img, result, segms, classes, ids)

        if self.show_boxes:
            result = self.overlay_boxes(result, boxes, classes)

        result = self.overlay_class_names(result, boxes, classes, scores,
                                          show_score=self.show_scores)
        return result

    def compute_colors_for_labels(self, labels):
        print(labels)
        colors = labels[:, None] * self.class_color_palette
        colors = (colors % 255).astype(np.uint8)
        #print(colors)

        return colors

    def overlay_boxes(self, image, boxes, classes):
        colors = self.compute_colors_for_labels(classes).tolist()
        for box, color in zip(boxes, colors):
            box = box.astype(int)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            cv2.imshow("g",image)
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
        return image

    def overlay_masks(self, function, c_img, image, masks, classes, ids=None):
        
        bg = cv2.imread(c_img, cv2.IMREAD_UNCHANGED)
        colors = (  1, 127,  31)
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        for i, (mask, color) in enumerate(zip(masks, colors)):
            color_idx = i if ids is None else ids[i]
        
            mask_color = [255, 255, 255]
            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            
            cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_color, dtype=np.uint8),
                           dst=aggregated_colored_mask, mask=mask)
                    
            mask_inv= cv2.bitwise_not(aggregated_colored_mask)
            #cv2.imshow("mask", aggregated_colored_mask)

        if function == '1':
            image = cv2.bitwise_or(mask_inv, image)

        if function == '2':
            image = cv2.bitwise_or(mask_inv, image)
            bg = cv2.resize(bg, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)

            bg = cv2.bitwise_or(bg, aggregated_colored_mask)
            image = cv2.bitwise_and(image, bg)

        if function == '3':
            image = cv2.bitwise_or(mask_inv, image)
            segments_image = cv2.bitwise_or(segments_image, aggregated_colored_mask)
            segments_image = cv2.GaussianBlur(segments_image, (235,235), 0)
            
            image = cv2.bitwise_and(image, segments_image)
        
        if function == '0':
            image = cv2.bitwise_and(image, aggregated_colored_mask)
            #cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)
            
        image = image[0:image.shape[0]-70, 0:image.shape[1]]

        return image

    def overlay_class_names(self, image, boxes, classes, scores, show_score=True):
        labels = ['person']#[self.class_labels[i] for i in classes]
        template = '{}: {:.2f}' if show_score else '{}'
        white = (255, 255, 255)

        for box, score, label in zip(boxes, scores, labels):
            s = template.format(label, score)
            textsize = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(int)
            #cv2.putText(image, s, tuple(position), cv2.FONT_HERSHEY_SIMPLEX, .5, white, 1)
        return image
