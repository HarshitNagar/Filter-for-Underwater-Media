import cv2
import numpy as np
from scipy.ndimage.filters import median_filter

cap = cv2.VideoCapture('gopro_op.mp4')
while(cap.isOpened()):
    ret, raw = cap.read()
    #raw = cv2.imread("raw.jpg")
    B, G, R = cv2.split(raw)
    '''
    B = blue channel
    G = green channel
    R = red channel
    '''

    B = B.astype(float)
    G = G.astype(float)
    R = R.astype(float)

    '''
    rescale B,G,R channels
    '''
    resB = (B - B.min())/(B.max() - B.min())
    resG = (G - G.min())/(G.max() - G.min())
    resR = (R - R.min())/(R.max() - R.min())

    '''
    VARIABLES

    Gbar = mean value of resB
    Rbar = mean value of resR
    alpha = constant parameter
    proc = image after boosting red channel in raw
    '''

    '''
    Boost red channel
    '''
    Rbar = np.mean(resB)
    Gbar = np.mean(resG)
    alpha = 1

    boostR = resR + alpha*(Gbar - Rbar)*(1-resR)*resG
    proc = cv2.merge((resB,resG,boostR))

    '''
    applying Static Gray World algorithm on proc
    '''
    B, G, R = cv2.split(proc)

    B = B.astype(float)
    G = G.astype(float)
    R = R.astype(float)

    Bbar = np.mean(resB)
    Gbar = np.mean(resG)
    Rbar = np.mean(resR)

    #proc.size
    illumination = np.sum(B)/proc.size, np.sum(G)/proc.size, np.sum(R)/proc.size
    scale = np.sum(illumination)/3

    B = B*scale/illumination[0]
    G = G*scale/illumination[1]
    R = R*scale/illumination[2]

    proc2 = cv2.merge((B,G,R))

    '''
    Apply Gamma Correction proc2

    set correction constant accordingly
    '''
    correction = 3
    beta = 1.0
    proc2gamma = beta*cv2.pow(proc2, correction)
    #proc2med = cv2.medianBlur(proc2gamma,3)
    #proc2med = gaussian_filter(proc2gamma, sigma=0.6)
    proc2med = median_filter(proc2gamma, 2, mode = 'constant')#, footprint=2,2,2, output=None, mode='reflect', cval=0.0, origin=0)


    '''
    cv2.imshow("raw",raw)
    cv2.waitKey()
    cv2.imshow("proc",proc)
    cv2.waitKey()
    cv2.imshow("proc2", proc2)
    cv2.waitKey()
    cv2.imshow("proc2gamma", proc2gamma)
    cv2.waitKey()
    '''
    #cv2.waitKey()
    cv2.imshow("proc2med",proc2med)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
