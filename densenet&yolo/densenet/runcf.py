# data channel order: RGB(0~255) input = input / 255 crop: crop the central region of the image with an area containing 87.5%
# of the original image. resize: 224 * 224 (tf.image.resize_bilinear(image, [height, width], align_corners=False)) input = 2*(input - 0.5)

#1. data channel order: RGB(0~255)
#2. resize: short side reisze to 256 and keep the aspect ratio.
#3. center crop: 224 * 224
#4. input = input / 255
#5. input = 2*(input - 0.5) 

from ctypes import *
import cv2
import numpy as np
from dnndk import n2cube
import os
#import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
#import preprocess
#import queue
import sys

try:
    pyc_libdputils = cdll.LoadLibrary("libn2cube.so")
except Exception:
    print('Load libn2cube.so failed\nPlease install DNNDK first!')

top = 1 
resultname = "image.list.result"
threadPool = ThreadPoolExecutor(max_workers=2,)
#scale = 1
#shortsize = 256

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']  
listPredictions = []

#from IPython.display import display
#from PIL import Image

#path = os.path.join(image_folder, listimage[2])
#print("path = %s" % path)
#img = cv2.imread(path)
#display(Image.open(path))

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]

def BGR2RGB(image):
  # B, G, R = cv2.split(image)
  # image = cv2.merge([R, G, B])
  # image = image[:,:,::-1] 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def resize_shortest_edge(image, size):
  H, W = image.shape[:2]
  if H >= W:
    nW = size
    nH = int(float(H)/W * size)
  else:
    nH = size
    nW = int(float(W)/H * size)
  return cv2.resize(image,(nW,nH))

def central_crop(image, crop_height, crop_width):
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2
  return image[offset_height:offset_height + crop_height, offset_width:
               offset_width + crop_width, :]

def normalize(image):
  image = image.astype(np.float32)
  image=image/256.0
  image=image-0.5
  image=image*2.0
  return image

#def preprocess_fn(image_path):
#    '''
#    Image pre-processing.
#    Rearranges from BGR to RGB then normalizes to range 0:1
#    input arg: path of image file
#    return: numpy array
#    '''
#    image = cv2.imread(image_path)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = image/255.0
#    return image

def preprocess_fn(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
#    print(f"image.type = {type(image)}")
    image = image/255.0
    return image

#def preprocess_fn(image, crop_height, crop_width):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
##    cv2.imshow("test", image)
##    cv2.waitKey(0)  
#    image = resize_shortest_edge(image, 256)
##    print("shape in = {}".format(image.shape))  
#    image = central_crop(image, crop_height, crop_width)
##    print("shape cr = {}".format(image.shape))
#    image = normalize(image)   
#    return image  

#def dpuSetInputImageWithScale(task, nodeName, image, mean, scale, height, width, channel, shortsize, idx=0):
#    (imageHeight, imageWidth, imageChannel) = image.shape
#    if height == imageHeight and width == imageWidth:
#        newImage = image
#    else:
#        newImage = preprocess_fn(image, height, width)
#    return newImage

def parameter(task, nodeName, idx=0):
    #inputscale =  n2cube.dpuGetInputTensorScale(task, nodeName, idx)
    #print("inputscale = %f"%inputscale)
    channel = n2cube.dpuGetInputTensorChannel(task, nodeName, idx)
    output = (c_float * channel)()
    outputMean = POINTER(c_float)(output)
    pyc_libdputils.loadMean(task, outputMean, channel)
    height = n2cube.dpuGetInputTensorHeight(task, nodeName, idx)
    print("height = %d"%height)
    width = n2cube.dpuGetInputTensorWidth(task, nodeName, idx)
    print("width = %d"%width)    
    for i in range(channel):
        outputMean[i] = float(outputMean[i])
#        print("outputMean[%i] = %f"%(i,outputMean[i]))
    return height, width, channel, outputMean
                 
def predict_label(img, task, inputscale, mean, height, width, inputchannel, shortsize, KERNEL_CONV_INPUT):
#    imageRun = preprocess_fn(img, height, width)
    imageRun = preprocess_fn(img)
#    imageRun = dpuSetInputImageWithScale(task, KERNEL_CONV_INPUT, img, mean, inputscale, height, width, inputchannel, shortsize, idx=0)
#    n2cube.dpuGetInputTensor(task, KERNEL_CONV_INPUT)
    imageRun = imageRun.reshape((imageRun.shape[0]*imageRun.shape[1]*imageRun.shape[2]))
#    input_len = 150528
#    print("imageRUN = {}".format(imageRun))
#    input_len = len(imageRun)
#    print(f"imageRun = {imageRun.shape}")
#    print(f"input_len = {input_len}")    
    return imageRun

def TopK(softmax, imagename, fo, correct, wrong):
    for i in range(top):
         num = np.argmax(softmax)
#         print("softmax = %f" % softmax[num])    
#         argmax = np.argmax((out_q[i]))
         prediction = classes[num]  
#         print(prediction)
#         softmax[num] = 0
#         num -1, should notice
#         num = num -1
#         fo.write(imagename+" "+str(num)+"\n")  
         ground_truth, _ = imagename.split('_')
         fo.write(imagename+' p: '+prediction+' g: '+ground_truth+' : '+str(softmax[num])+'\n')
         if (ground_truth==prediction):
            correct += 1
#            print(f"correct = {correct}")
         else:
            wrong += 1
#            print(f"wrong = {wrong}")
    return correct, wrong   
#sem=threading.BoundedSemaphore(1)
def run_dpu_task(outsize, task, outputchannel, conf, outputscale, listimage, imageRun, KERNEL_CONV_INPUT, KERNEL_FC_OUTPUT): 
    input_len = len(imageRun)
#    print(f"input_len = {input_len}")
    n2cube.dpuSetInputTensorInHWCFP32(task,KERNEL_CONV_INPUT,imageRun,input_len)
    n2cube.dpuRunTask(task)
#    outputtensor = n2cube.dpuGetOutputTensorInHWCFP32(task, KERNEL_FC_OUTPUT, outsize)
#    print(outputtensor)
#    print(outputchannel)
#    print(outputscale)
    softmax = n2cube.dpuRunSoftmax(conf, outputchannel, outsize//outputchannel, outputscale)
#    print(f"softmax = {softmax}")
    return softmax, listimage

def run(image_folder, shortsize, KERNEL_CONV, KERNEL_CONV_INPUT, KERNEL_FC_OUTPUT, inputscale):

    start = time.time()
#    listimage = [i for i in os.listdir(image_folder) if i.endswith("JPEG")]
    listimage = [i for i in os.listdir(image_folder) if i.endswith("jpg")]
    listimage.sort()
#    wordstxt = os.path.join(image_folder, "words.txt")
#    with open(wordstxt, "r") as f:
#        lines = f.readlines()
    fo = open(resultname, "w")
    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)
    height, width, inputchannel, mean = parameter(task, KERNEL_CONV_INPUT)
#    print("mean = %f"%mean[0])
    outsize = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT)
#    print("size = %d"%size)
    outputchannel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_FC_OUTPUT)
#    print("outputchannel = %d"%outputchannel)
    conf = n2cube.dpuGetOutputTensorAddress(task, KERNEL_FC_OUTPUT)
#    print("conf = {}".format(conf))
#    print("inputscale = %f"%inputscale)
    inputscale = n2cube.dpuGetInputTensorScale(task,KERNEL_CONV_INPUT)
#    print("inputscalenow = %f"%inputscale)
    outputscale = n2cube.dpuGetOutputTensorScale(task, KERNEL_FC_OUTPUT)
#    print("outputscale = %f"%outputscale)  
    imagenumber = len(listimage) 
    print("\nimagenumber = %d\n"%imagenumber)
    softlist = []
#    imagenumber = 1000
    correct = 0
    wrong = 0
    for i in range(imagenumber):
        print(f"i = {i+1}") 
        print(listimage[i]) 
#        path = os.path.join(image_folder, listimage[i])
#        if i % 50 == 0:
#        print("\r", listimage[i], end = "") 
        path = image_folder + listimage[i]
        img = cv2.imread(path)
        imageRun = predict_label(img, task, inputscale, mean, height, width, inputchannel, shortsize, KERNEL_CONV_INPUT)
        input_len = len(imageRun)
#        print(f"input_len = {input_len}")     
#        soft = threadPool.submit(run_dpu_task, outsize, task, outputchannel, conf, outputscale, listimage[i], imageRun, KERNEL_CONV_INPUT, KERNEL_FC_OUTPUT)
#        softlist.append(soft)
#    for future in as_completed(softlist):
#        softmax, listimage = future.result()
        softmax, listimage[i] = run_dpu_task(outsize, task, outputchannel, conf, outputscale, listimage[i], imageRun, KERNEL_CONV_INPUT, KERNEL_FC_OUTPUT)
        correct, wrong = TopK(softmax, listimage[i], fo, correct, wrong)
        print("")

    fo.close()
    accuracy = correct/imagenumber
    print('Correct:',correct,' Wrong:',wrong,' Accuracy:', accuracy)    
    n2cube.dpuDestroyTask(task)
    n2cube.dpuDestroyKernel(kernel)
    n2cube.dpuClose()
    print("")

    end = time.time()
    total_time = end - start 
    print('\nAll processing time: {} seconds.'.format(total_time))
    print('\n{} ms per frame\n'.format(10000*total_time/imagenumber))
   
#threadPool.shutdown(wait=True)          
#criteria = (Accruacy-top1% -68.5)/15)*0.4+(10/latencyms)*0.6 
