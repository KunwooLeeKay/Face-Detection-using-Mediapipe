import cv2
import mediapipe as mp

# 얼굴을 찾고, 찾은 얼굴에 표시를 해주기 위한 변수 정의
mp_face_detection = mp.solutions.face_detection # 얼굴 검출을 위한 face_detection 모듈을 사용
mp_drawing = mp.solutions.drawing_utils # 얼굴의 특징을 그리기 위한 drawing_utils 모듈을 사용

# 동영상 파일 열기
cap = cv2.VideoCapture('face_video.mp4')

# with 를 쓴 것은 자동으로 자원 해제를 하기 위해서임. 
# model_selection 에서 0 은 2m 내, 1은 멀리 있는 얼굴. mind_detection_confidence 는 쓰레스홀드같은 거임. 높으면 확실
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    
    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            break
            
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 미디어파이프는 RGB 기준이라서 바꿔준것
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        if results.detections:
            # 6개의 특징 반환: 오른쪽 눈, 왼쪽 눈, 코 끝부분, 입 중심, 오른쪽 귀, 왼쪽 귀(귀구슬점)
            for detection in results.detections:
                #mp_drawing.draw_detection(image, detection)
                #print(detection)
                
                # 특정 위치 가져오기
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                nose = keypoints[2]
                
                h, w, _ = image.shape # 이미지로부터 세로, 가로 크기 가져옴
                right_eye = (int(right_eye.x * w), int(right_eye.y * h)) # 이미지 내에서 실제 좌표 (x, y)
                left_eye = (int(left_eye.x * w), int(left_eye.y * h))
                nose = (int(nose.x * w), int(nose.y * h))
                
                # 양 눈에 동그라미 그리기
                cv2.circle(image, right_eye, 50, (255, 0, 0), 10, cv2.LINE_AA)
                cv2.circle(image, left_eye, 50, (0, 255, 0), 10, cv2.LINE_AA)
                cv2.circle(image, nose, 50, (0, 0, 255), 10, cv2.LINE_AA)
                

        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None,fx = 0.5, fy = 0.5))
        
        if cv2.waitKey(1) == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
