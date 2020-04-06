import numpy as np
import struct
from scipy import ndimage
import cv2
from os import walk
import sys
import glob



def getimenhancedmedian(input_img, ROI):
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
    #cv2.imwrite('test.png', enhanced)
    return enhanced


def getimenhancedtophat(input_img, ROI):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48))
    enhanced = cv2.morphologyEx(input_img, cv2.MORPH_BLACKHAT, kernel)
    maxperc = np.percentile(enhanced, 99)
    minperc = np.percentile(enhanced, 1)
    enhanced2 =255- np.clip((enhanced-minperc)*255/(maxperc - minperc),0,255).astype(np.uint8)

    return enhanced2

def getenhancedoneimage(angio, angioback,  mode):
    if mode == 'median':
        enhanced = getimenhancedmedian(angio, np.ones((angio.shape[0], angio.shape[1]), dtype=np.bool))
    elif mode == 'tophat':
        enhanced = getimenhancedtophat(angio, np.ones((angio.shape[0], angio.shape[1]), dtype=np.bool))
    elif mode == 'image':
        muimage = np.mean(angio[:, :, 0])
        stdimage = np.std(angio[:, :, 0])
        if muimage < 70 or muimage > 148:
            angio = (angio - muimage) / stdimage * 31 + 115
        enhanced = angio[:, :, 0]
    elif mode == 'imageback':
        muimage = np.mean(angio[:, :, 0])
        stdimage = np.std(angio[:, :, 0])
        if muimage < 70 or muimage > 148:
            angio = np.clip((angio - muimage) / stdimage * 31 + 115, 0, 255)
        muimage = np.mean(angioback)
        stdimage = np.std(angioback)
        if muimage < 70 or muimage > 148:
            angioback = np.clip((angioback - muimage) / stdimage * 31 + 115, 0, 255)
        enhanced = np.zeros((angioback.shape[0], angioback.shape[1], 3), dtype=np.uint8)
        enhanced[:, :, 0] = angioback
        enhanced[:, :, 1] = angio[:, :, 0]
        enhanced[:, :, 2] = angioback


    return enhanced

def angio2fundus2enhanced(dirin, dirout, mode, grayscale):
    filenamesall = glob.glob(dirin + '/*.tif', recursive=True)
    filenamesback = glob.glob(dirin + '/*back.tif', recursive=True)
    filenames = list(set(filenamesall) - set(filenamesback))
    for j in range(len(filenames)):
        print(str(j))
        angio = cv2.imread(filenames[j])
        filenamesplitinter = filenames[j].split('\\')
        filenamesplit = filenamesplitinter[len(filenamesplitinter) - 1].split('.')
        if mode=='median':
            enhanced = getimenhancedmedian(angio, np.ones((angio.shape[0], angio.shape[1]), dtype=np.bool))
        elif mode=='tophat':
            enhanced = getimenhancedtophat(angio, np.ones((angio.shape[0], angio.shape[1]), dtype=np.bool))
        elif mode=='image':
            muimage = np.mean(angio[:,:,0])
            stdimage = np.std(angio[:,:,0])
            if muimage<70 or muimage>148:
                angio = (angio - muimage)/stdimage * 31 + 115
            enhanced=angio[:,:,0]
        elif mode=='imageback':
            muimage = np.mean(angio[:,:,0])
            stdimage = np.std(angio[:,:,0])
            if muimage<70 or muimage>148:
                angio = np.clip((angio - muimage)/stdimage * 31 + 115,0,255)
            imageback = cv2.imread(dirin + '/' + filenamesplit[0] + '_back.tif', cv2.IMREAD_GRAYSCALE)
            muimage = np.mean(imageback)
            stdimage = np.std(imageback)
            if muimage<70 or muimage>148:
                imageback = np.clip((imageback - muimage)/stdimage * 31 + 115,0,255)
            enhanced = np.zeros((imageback.shape[0], imageback.shape[1], 3), dtype=np.uint8)
            enhanced[:, :, 0] = imageback
            enhanced[:, :, 1] = angio[:,:,0]
            enhanced[:, :, 2] = imageback
        if grayscale:
            cv2.imwrite(dirout + '/' + filenamesplit[len(filenamesplit)-2]+ '.jpg', enhanced[:,:,0])
        else:
            cv2.imwrite(dirout + '/' + filenamesplit[len(filenamesplit) - 2] + '.jpg', enhanced)



if __name__ == '__main__':
  angio2fundus2enhanced(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))