import cv2


url = "http://192.168.1.97:8080/video"  


cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow("Phone Live (Adjustable)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Phone Live (Adjustable)", 800, 600)  


while True:
    ok, frame = cap.read()
    if not ok:
        continue

    
    frame = cv2.resize(frame, (800, 600))  

    cv2.imshow("Phone Live (Adjustable)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
