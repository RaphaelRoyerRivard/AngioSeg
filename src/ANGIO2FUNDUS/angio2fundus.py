'# code 2 simulate fundus from angio: first test to see CNN capability 2 genereralize on other vessels structures'
from os import walk
import numpy as np
from scipy import ndimage
from scipy import misc
import math

def getmeanrgb(rgb):
    return np.mean(np.mean(rgb, axis=0), axis=0)

def flipRedBlue(rgb):
    test = rgb[:,:,0].copy()
    rgb[:,:,0]=rgb[:,:,2]
    rgb[:, :, 2] = test

    return rgb

def getimgfromlut(image, lut):
    angionew = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    angio = image
    if len(image.shape) == 3:
        angio = np.mean(image, axis=2).astype(np.uint8)
    for x in range(angio.shape[0]):
        for y in range(angio.shape[1]):
            angionew[x, y, :] = lut[angio[x, y], 0:3]

    #angionew = flipRedBlue(angionew)

    return angionew

def getlut(dirfundus):
    lut = np.zeros((256, 4), dtype=np.float)
    _, _, filenames = next(walk(dirfundus), (None, None, []))
    for j in range(len(filenames)):
        fundus = ndimage.imread(dirfundus + '/' + filenames[j])
        fundusd = fundus.astype(np.float)
        for x in range(fundusd.shape[0]):
            for y in range(fundusd.shape[1]):
                gray = fundus[x, y, 1]
                if gray>10:
                    lut[gray, 0:3] += fundusd[x,y,:]
                    lut[gray, 3] += 1


    for i in range(256):
        if not lut[i, 3]==0:
            lut[i, 0:3]/=lut[i, 3]
    return lut

def angio2fundus(dirin, dirout , dirfundus):

    #get the LUT
    lut = getlut(dirfundus)

    _, _, filenames = next(walk(dirin), (None, None, []))
    for j in range(len(filenames)):
        angio = ndimage.imread(dirin + '/' + filenames[j])
        angionew= getimgfromlut(angio, lut)
        filenamesplit = filenames[j].split('.')
        misc.imsave(dirout + '/' + filenamesplit[len(filenamesplit)-2]+ '.jpg', angionew)



if __name__ == '__main__':
  angio2fundus('G:/RECHERCHE/Work_CORSTEM/data/TRAIN','G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v00/train', 'G:/RECHERCHE/databases/DRIVE/DRIVE/training/images/normal')