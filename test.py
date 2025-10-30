import cv2

# --------------------------
# Configuration
# --------------------------
phone_url = "http://192.168.1.97:8080/video"   # your phone IP Webcam stream
laptop_cam_index = 0                           # default webcam index

# --------------------------
# Initialize both cameras
# --------------------------
phone_cap = cv2.VideoCapture(phone_url)
laptop_cap = cv2.VideoCapture(laptop_cam_index)      #Initialize video capture objects for phone and laptop camera inputs

if not phone_cap.isOpened():
    print("❌ Could not open phone camera stream.")
if not laptop_cap.isOpened():
    print("❌ Could not open laptop camera.")

# Reduce buffering and tune FPS for phone
phone_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
phone_cap.set(cv2.CAP_PROP_FPS, 30)

# Optional: set resolution for laptop cam
laptop_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
laptop_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --------------------------
# Create resizable windows
# --------------------------
cv2.namedWindow("Laptop Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Phone Camera", cv2.WINDOW_NORMAL)

# Initial window sizes (can drag to resize freely)
cv2.resizeWindow("Laptop Camera", 640, 480)
cv2.resizeWindow("Phone Camera", 800, 600)

# --------------------------
# Main loop
# --------------------------
while True:
    ok_lap, laptop_frame = laptop_cap.read()
    ok_phone, phone_frame = phone_cap.read()

    # Resize each frame programmatically (optional)
    if ok_lap:
        laptop_frame = cv2.resize(laptop_frame, (640, 480))
        cv2.imshow("Laptop Camera", laptop_frame)

    if ok_phone:
        phone_frame = cv2.resize(phone_frame, (800, 600))
        cv2.imshow("Phone Camera", phone_frame)

    # Quit both feeds with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------
# Cleanup
# --------------------------
laptop_cap.release()
phone_cap.release()
cv2.destroyAllWindows()
