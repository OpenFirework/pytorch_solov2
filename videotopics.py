import cv2
vid = cv2.VideoCapture('cam0.avi')
target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(num_frames):
   if 1:
     frame=vid.read()
     img=frame[1]
     filename=str(i)+".jpg"
     cv2.imwrite(filename, img)

