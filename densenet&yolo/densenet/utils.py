import os
import time
import runcf
from pynq_dpu import DpuOverlay

#data channel order: BGR(0~255) resize: short side reisze to 320 and keep the aspect ratio. 
#center crop: 299 * 299 mean_value: 104, 117, 123 scale: 1.0
#data channel order: BGR(0~255) resize: short side reisze to 320 and keep the aspect ratio. 
#center crop: 299 * 299 mean_value: 104, 117, 123 scale: 0.0078125
BASE_DIR = "."
RESULT_DIR = BASE_DIR
RESULT_FILE = RESULT_DIR + '/image.list.result'
baseImagePath = "/home/xilinx/jupyter_notebooks/densenetcifar/images/"

overlays = "overlays_300M2304"
#overlays = "overlays_400M2304"  
#overlays = "overlays_300M1600"
#overlays = "overlays_400M1600"
print("\noverlays = %s" % overlays)

KERNEL_CONV = "tf_densenet"
#KERNEL_CONV = "inceptionv4_0"
print("Model: %s" % KERNEL_CONV)
KERNEL_CONV_INPUT = "conv2d_Conv2D"
KERNEL_FC_OUTPUT = "dense_MatMul"
#KERNEL_CONV_INPUT = "conv1_7x7_s2"
#KERNEL_FC_OUTPUT = "loss3_classifier"
scale = 0.0078125
shortsize = 320

class Processor:
    def __init__(self):
        pass

    # User should rewrite this function to run their data set and save output to result file.
    # Result file name should be "image.list.result" and be saved in the main directory
    def run(self):
        os.system("cp ./"+overlays+"/* /usr/local/lib/python3.6/dist-packages/pynq_dpu/overlays/")
        os.system("cp ./"+overlays+"/*.so /usr/lib/")
#        overlay = DpuOverlay("dpu.bit")
        os.system("dexplorer -w")
        #os.system("ls -l /usr/local/lib/python3.6/dist-packages/pynq_dpu/overlays/*")
        #print(" ")
        #os.system("ls -l /usr/lib/libdpu*.so")
        runcf.run(baseImagePath, shortsize, KERNEL_CONV, KERNEL_CONV_INPUT, KERNEL_FC_OUTPUT, scale)
        # Run all date set and write your outputs to result file.
        # Please see README and "classification_result.sample" to know the result file format.
        # time.sleep(10)
        # import os
        # os.system('python3 inception.py baseImagePath imagenumber') 
        return
   
