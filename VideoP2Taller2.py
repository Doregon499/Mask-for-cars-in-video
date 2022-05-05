import numpy as np
import cv2
import time

from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

def video():
    folder = 'C:/Users/andre/Desktop/Python/Codigos Vs/Procesamiento/'
    inputFile = 'Video.mp4'

    
    
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(folder + inputFile)
    
    if not cap.isOpened():
        print('Error opening video stream or file')
        return 0
    
    FPS = cap.get(cv2.CAP_PROP_FPS)
    videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cap.get(cv2.CAP_PROP_FOURCC)

    print('/n')
    print('-----------------------')
    print('VIDEO INPUT INFORMATION')
    print('-----------------------')
    print('Frames per second : {:.2f}'.format(FPS))
    print('Video size: ' + str(videoWidth) + ' X ' + str(videoHeight))
    
    print('\n')
    print('------------------------')
    print('VIDEO PROCESSING STARTED')
    print('------------------------')
    
    # Frame 1
    cap.set(cv2.CAP_PROP_POS_FRAMES,1200)
    ret, frame = cap.read()
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
        
    while (True):
        prev_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frameGRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        v, u = optical_flow_ilk(prev_frame, frameGRAY, radius=15)
        
        norm = np.sqrt(u * 2 + v * 2)
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
        
        ax0.imshow(prev_frame, cmap='gray')
        ax0.set_title("Frame")

        nvec = 20  # Number of vectors to be displayed along each image dimension
        nl, nc = prev_frame.shape
        step = max(nl//nvec, nc//nvec)
        
        y, x = np.mgrid[:nl:step, :nc:step]
        u_ = u[::step, ::step]
        v_ = v[::step, ::step]
        
        ax1.imshow(norm)
        ax1.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
        ax1.set_title("movimiento flujo optico")
        ax1.set_axis_off()
        fig.tight_layout()
        
        plt.show()
        
        prev_frame = frameGRAY
        
        try:
            print('Processing speed: {:.2f} FPS'.format(1/(time.time()-prev_time)), end='\n')
        except:
            print('Processing speed: 0 FPS')
        
        cv2.imshow('Original', norm)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cap.release()
    
    cv2.destroyAllWindows()
    
    print('\n')
    print('----------------------')
    print('VIDEO PROCESSING ENDED')
    print('----------------------')
    
if __name__ == "__main__":
    video()
