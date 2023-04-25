import numpy as np

from detector_car import Detector
from detector_lpr import DetectorLPR
import cv2
import time
import sys
import datetime
from datetime import datetime
import torch
import math

from glob import glob
import os
import shutil

def read_license_plate_number(img, char_threshold):
    
    plate_reading = 'NULL'
    
    # detect plate
    bboxes = detector.detect(img)
    
    # filter 'plate only' detections
    plate_only = []
    for i in bboxes:
        if i[4] == 'plate':
            plate_only.append(i)
    
    if plate_only:
        max_tuple = max(plate_only, key=lambda x:x[5].item()) # find the most probable plate detection
        x1 = max_tuple[0]
        y1 = max_tuple[1]
        x2 = max_tuple[2]
        y2 = max_tuple[3]
        plate_crop = img[y1:y2, x1:x2]
        # print(plate_crop.shape)
        bboxes_lpr = detector_lpr.detect(plate_crop)
        
        if bboxes_lpr:
            characters_only = []
            coordinates_only = []
            midpoints_only = []
            xc_only = []
            yc_only = []
            norms_only = []
            for i in bboxes_lpr:
                if i[5].item() >= char_threshold:     # filter out the character predictions below the threshold
                    characters_only.append(i[4])
                    coordinates_only.append((i[0], i[1], i[2], i[3]))
                    xc = (float(i[0])+float(i[2]))/2
                    yc = (float(i[1])+float(i[3]))/2
                    xc_only.append(xc)
                    yc_only.append(yc)
                    midpoints_only.append( ( xc, yc ) )
                    norm = round(math.sqrt(xc**2 + yc**2), 3)
                    norms_only.append(norm)
            
            if xc_only == [] or yc_only == [] or characters_only == []:
                return plate_reading
            yc_only, xc_only, characters_only = zip(*sorted(zip(yc_only, xc_only, characters_only))) # sort characters according to y positions
            
            max_diff = 5
            char_matrix = []
            char_row = []
            xc_matrix = []
            xc_row = []
            yc_matrix = []
            yc_row = []
            
            anchor = yc_only[0]
            for i, o in enumerate(yc_only):     # segment characters into rows
                if abs(o-anchor) <= max_diff:
                    char_row.append(characters_only[i])
                    xc_row.append(xc_only[i])
                    yc_row.append(yc_only[i])
                    anchor = (anchor + yc_only[i])/2
                    if i == (len(yc_only) - 1):
                        char_matrix.append(char_row)
                        xc_matrix.append(xc_row)
                        yc_matrix.append(yc_row)
                else:
                    char_matrix.append(char_row)
                    xc_matrix.append(xc_row)
                    yc_matrix.append(yc_row)
                    char_row = []
                    xc_row = []
                    yc_row = []
                    char_row.append(characters_only[i])
                    xc_row.append(xc_only[i])
                    yc_row.append(yc_only[i])
                    anchor = yc_only[i]
            
            final_plate_str = ""
            for j, o in enumerate(char_matrix):
                char_row = char_matrix[j]
                xc_row = xc_matrix[j]
                yc_row = yc_matrix[j]
                xc_row, yc_row, char_row = zip(*sorted(zip(xc_row, yc_row, char_row))) # sort row characters according to x positions
                for c in char_row:
                    final_plate_str += c
            
            print('characters detected')
            # cv2.imwrite('alpr_results/'+final_plate_str+'.jpg', plate_crop)
            plate_reading = final_plate_str
            
        else:
            print('no characters detected')
            # cv2.imwrite('alpr_results/no_characters_detected.jpg', plate_crop)
            
    else:
        print('plate not detected')
        # cv2.imwrite('alpr_results/plate_not_detected.jpg', img)
        
    return plate_reading

if __name__ == '__main__':
    
    image_path = sys.argv[1]    # path of the test image passed as an argument in the command line
    
    # initialize yolov5; for plate localization and LPR
    detector = Detector()
    detector_lpr = DetectorLPR()

    # initialize LPR prediction threshold
    licence_plate_char_threshold = 0.70

    # perform license plate recognition
    img = cv2.imread(image_path)
    plate_reading = read_license_plate_number(img, licence_plate_char_threshold)
    print('license plate number: ', plate_reading)