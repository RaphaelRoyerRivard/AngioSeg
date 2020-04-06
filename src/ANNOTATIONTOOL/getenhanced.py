import numpy as np
import struct
from scipy import ndimage
import cv2
from os import walk
#
# filenamecour = 'G:\RECHERCHE\databases\KAGGLE\\train\\163_right.jpeg'
# input_img = ndimage.imread(filenamecour)
# input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)


def getimenhanced(input_img, ROI):
    # meantrue = np.mean(input_img[ROI], axis=0)
    # stdtrue = np.std(input_img[ROI], axis=0)

    # Median blur
    D = input_img.shape[0]
    kmedian = min(D // 20, 255)
    if kmedian % 2 == 0:
        kmedian += 1
    cvblur = cv2.medianBlur(input_img, int(kmedian))
    image = (input_img.astype(float) - cvblur.astype(float) )

    # Mean and std after median blur
    meannew = np.mean(image[ROI], axis=0)
    stdnew = np.std(image[ROI], axis=0)

    # Equalize
    enhanced = ((image - meannew) / stdnew) * 35.0 + 127.0

    enhanced = (np.clip(enhanced, 0, 255)).astype(np.uint8)
    cv2.imwrite('test.png', enhanced)
    return enhanced


def angio2fundus2enhanced(dirin, dirout):
    _, _, filenames = next(walk(dirin), (None, None, []))
    for j in range(len(filenames)):
        angio = cv2.imread(dirin + '/' + filenames[j])
        enhanced = getimenhanced(angio, np.ones((angio.shape[0], angio.shape[1]), dtype=np.bool))
        filenamesplit = filenames[j].split('.')
        cv2.imwrite(dirout + '/' + filenamesplit[len(filenamesplit)-2]+ '.jpg', enhanced)



if __name__ == '__main__':
  angio2fundus2enhanced('G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v00/train/img', 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v00/train/enhanced')