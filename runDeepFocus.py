from __future__ import division, print_function, absolute_import
import openslide
import numpy
import matplotlib.pyplot as plt
from skimage import filters
from skimage import color
import tflearn
import sys
import numpy as np
import classificationModel3
import hyperparameterModel
import platform
import time
import os
import glob
import csv


def writeXML(filename,data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def analyze(imgpath,model):

    imgname=os.path.basename(imgpath)
    kernelSize=64;
    kernelStepSize=1;
    bufferVal = 8# will load kernelSize x bufferVal
    stepsize=1
    starttime=time.time()

    slide = openslide.open_slide(imgpath)
    #tissue detection
    thumbnail=np.array (slide.get_thumbnail((slide.level_dimensions[0][0]/64,slide.level_dimensions[0][1]/64)))
    thumbnailGray = color.rgb2gray(thumbnail)
    val = filters.threshold_otsu(thumbnailGray)
    tissueMask = thumbnailGray < max(val,0.8)
    plt.imsave('tissue.png', tissueMask) #save the thumb of tissue mask

    buffersize=kernelSize*bufferVal;
    resultMask = thumbnail.astype(numpy.uint8) * 0;
    if stepsize>1:
        resultMask=np.resize(resultMask,( int(resultMask.shape[0]/stepsize) , int(resultMask.shape[1]/stepsize) ,resultMask.shape[2]))
    counter1=0
    counter2=0
    expectedStep = tissueMask.shape[0] / bufferVal
    outputsVec=[]
    for i in range( 0,tissueMask.shape[0]-bufferVal,bufferVal): #Height

        curMod= i %  (bufferVal*max(5,int(expectedStep/20)))
        if curMod==0:
            print('.', end='', flush=True)
        for j in range(0,tissueMask.shape[1]-bufferVal,bufferVal): #Width

            if np.mean(tissueMask[i:i+bufferVal,j:j+bufferVal])< (8/16):  #most of them are background
                continue
            bigTile=numpy.array(slide.read_region((j*kernelSize,i*kernelSize),0,[buffersize,buffersize]))
            bigTile=color.rgba2rgb(bigTile)
            sz = bigTile.itemsize
            h, w,cs = bigTile.shape
            bh, bw =kernelSize,kernelSize
            shape = (int(h / bh/stepsize), int(w / bw/stepsize), bh, bw, cs)
            strides = (stepsize*w * bh * sz * cs, stepsize * sz * cs*bw, w * sz * cs, sz * cs, sz)
            blocks = np.lib.stride_tricks.as_strided(bigTile, shape=shape, strides=strides)
            blocks= blocks.reshape(blocks.shape[0]*blocks.shape[1], blocks.shape[2], blocks.shape[3], blocks.shape[4])
            predictions = model.predict(blocks)
            outputsVec=outputsVec+predictions
            qwe = np.array(predictions)
            qwe = qwe.reshape(int(h / bh/stepsize), int(w / bw / stepsize),2)
            counter1= counter1 + sum(np.array(predictions)[:,1]>0.5)
            counter2= counter2 + len(predictions)- sum(np.array(predictions)[:,1]>0.5)
            resultMask[int(i/stepsize): int((i + bufferVal)/stepsize) , int(j/stepsize): int((j+ bufferVal)/stepsize),0]=255*qwe[:,:,1]
            resultMask[int(i/stepsize): int((bufferVal+i)/stepsize) , int(j/stepsize):int((bufferVal +j)/stepsize), 1] = 255*qwe[:, :, 0]
    endtime = time.time()
    elapsedtime=endtime-starttime
    print ('elapsed time ' +  str(elapsedtime))
    outputname= 'output/' + imgname +'-f'+ str(counter2)+'-o'+ str(counter1)+'.png'
    plt.imsave(outputname, resultMask, cmap=plt.cm.gray)
    outputname2 ='output/' + imgname + '-f' + str(counter2) + '-o' + str(counter1) + '.csv'
    writeXML(outputname2, outputsVec)
    return (counter2,counter1)

def main():
    params=hyperparameterModel.hyperparameterModel()
    print  (sys.argv)
    outputpath = ''
    outputFile = outputpath + 'ver5'
    # Model training

    tflearn.init_graph()
    g = classificationModel3.createModel(params)
    model = tflearn.DNN(g)
    model.load(outputFile)
    files = glob.glob("D:/DATA/testForLopez/*.svs")
    f = open('output/outputlogs.txt', 'w')
    for myfile in files:
        print (myfile)
        results=analyze(myfile,model)
        f.write((myfile+' ' +str(results[0])+' '+ str(results[1])+'\n'))

    f.close()

if __name__ == "__main__":
    main()
