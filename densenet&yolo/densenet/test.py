
# coding: utf-8

import time

# Import your processing code

import utils

# Run all process

evaluate_file = "./evaluation.py"
golden_file = "./image.list.gt"
result_file = "./image.list.result"
total_time = 0

processor = utils.Processor()

# Start timer
start = time.time() 
# run processor and save output 
processor.run()
#time.sleep(10)

# timer stop after batch processing is complete
end = time.time()
total_time = end - start 

#('\nAll processing time: {} seconds.'.format(total_time))
#print('\nEvaluate accuracy:\n')

# run accuracy
#%run ./evaluation.py ./image.list.gt ./image.list.result
#get_ipython().magic('run ./evaluation.py $golden_file $result_file')
#import os
#os.system('python3 evaluation.py image.list.gt image.list.result')

