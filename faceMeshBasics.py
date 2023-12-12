import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

#implementing model for face mesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
#module for drawing landmark
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles
#drawing spec
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


pTime = 0
while True:
    success, img = cap.read()
    #to improve the performance, optionally mark the image as not writeable to pass by reference
    img.flags.writeable = False
    #converting image to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)
    #drawing the face mesh annotations on the image
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #drawing face landmarks
            mpDraw.draw_landmarks(image=img,
                                  landmark_list=faceLms,
                                  connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawStyles.get_default_face_mesh_tesselation_style())
            #borders like face, lips, eybrow
            """mpDraw.draw_landmarks(image=img,
                                  landmark_list=faceLms,
                                  connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawStyles.get_default_face_mesh_contours_style())"""
            #getting landmarks
            for id, lm in enumerate(faceLms.landmark):
                #print(lm)
                #getting height, width, channel for getting pixel values
                h, w, c = img.shape
                x, y = int(lm.x*w), int(lm.y*h)
                print(id, x, y)
    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break