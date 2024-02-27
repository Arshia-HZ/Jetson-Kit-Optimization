import numpy as np
import imutils
import cv2
import time
from operator import itemgetter
from baseFewShotMatcher import BaseFewShotMatcher

class TMOpenCv(BaseFewShotMatcher):
     def predict(self, target, templates, tempbox, _):
        predictions = []
        
        tar = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        # scales = np.linspace(0.2, 5.0, 9)
        scales = [0.2, 0.25, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.3, 1.6, 2.0, 2.3, 2.6, 3.0, 4.0, 5.0]
        x1, y1, x2, y2 = -1, -1, -1, -1
        maxVal = -1
        for i, temp in enumerate(templates):
            w, h = temp.shape[::-1]
            nms_pred = []
            
            
            for scale in scales:
                start = time.time()
                if scale <= 1:
                    resized = imutils.resize(tar, width=int(tar.shape[1]*scale), height=int(tar.shape[0]*scale))
                    resized_t = temp
                else:
                    resized_t = imutils.resize(temp, width=int(temp.shape[1]/scale), height=int(temp.shape[0]/scale))
                    resized = tar
                    
                r = tar.shape[1] / float(resized.shape[1])
                w, h = resized_t.shape[::-1]
                
                # if the resized image is smaller than the template, then break from the loop
                if resized.shape[0] < h or resized.shape[1] < w:
                    end = time.time()
                    timePred = float("{:.3f}".format(end - start))
                    continue
                    
                if w*h < 300: # TODO: min desired scale
                    end = time.time()
                    timePred = float("{:.3f}".format(end - start))
                    continue
                    
                result = cv2.matchTemplate(resized, resized_t, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                end = time.time()
                timePred = float("{:.3f}".format(end - start))
                
                if scale > 1:
                    maxVal /= (scale ** 0.5)
                
                x1 = int(maxLoc[0] * r)
                y1 = int(maxLoc[1] * r)
                x2 = int((maxLoc[0]+w) * r)
                y2 = int((maxLoc[1]+h) * r)

                predictions.append([x1, y1, x2, y2,
                    float("{:.3f}".format(maxVal)), timePred])
    
            if len(predictions) == 0:
                predictions.append([x1, y1, x2, y2,
                -1, timePred])
                
        predictions.sort( reverse = True, key = itemgetter(4))
        
        # TODO : add a parameter for get max of predictions data e.g. in this case it is 10.
        return predictions[:10]
