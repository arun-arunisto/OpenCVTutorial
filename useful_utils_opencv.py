import cv2
import matplotlib.pyplot as plt
### displaying single frame from video
"""cap = cv2.VideoCapture(video_path)
while True:
    success, frame = cap.read()
    cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break"""
"""ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)"""
"""plt.imshow(frame_rgb)
plt.show()"""

### saving the frame as image
"""
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite("parking_lot.jpg", frame)
"""

##cropping image
#crop_region = frame_rgb[121:615, 269:500]
"""plt.imshow(crop_region)
plt.show()"""

###saving cropped image
"""crop_region = cv2.cvtColor(crop_region, cv2.COLOR_RGB2BGR)
cv2.imwrite("cropped_image.jpg", crop_region)"""

### cropping a video
"""while True:
    success, frame = cap.read()
    #cropping video
    crop = frame[121:615, 269:500]
    #cv2.imshow("video", frame)
    ### saving video
    print(crop.shape)
    cv2.imshow("cropped", crop)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break"""

## writing video
"""fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_put = cv2.VideoWriter("cropped_video.mp4", fourcc, 10, (231, 494))
while True:
    success, frame = cap.read()
    crop = frame[121:615, 269:500]

    if not success:
        break
    out_put.write(crop)

cap.release()
out_put.release()"""

###masking image
"""parking_img = cv2.imread("cropped_img_for_mask.jpg")
parking_rgb = cv2.cvtColor(parking_img, cv2.COLOR_BGR2RGB)
plt.imshow(parking_rgb)
plt.show()
#mask for original image
parking_gray = cv2.cvtColor(parking_rgb, cv2.COLOR_RGB2GRAY)
retval, parking_mask = cv2.threshold(parking_gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(parking_mask, cmap="gray")
plt.show()
cv2.imwrite("cropped_img_mask.jpg", parking_mask)"""

### reversing a video
"""def reverse_video(video_file, outputfile):
    cap = cv2.VideoCapture(video_file)
    #getting the total number of frames in the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #creating a list to store the frames
    frames = []
    #reversing the total frames
    print("Starting....")
    count = 0
    for i in range(int(total_frames)-1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count+=1
        print("Frame Count:",count)
    print("2nd stage...")
    #creating a new video file to write the reverse order
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputfile, fourcc, 10, (231, 494))
    print("Starting....")
    count = 0
    for frame in frames:
        out.write(frame)
        count += 1
        print("Frame Count:", count)
    print("Success...!")
    cap.release()
    out.release()

if __name__ == "__main__":
    reverse_video("cropped_video.mp4", "crop_reverse.mp4")"""