import cv2
import mediapipe as mp
from imgbeddings import imgbeddings
import chromadb
from PIL import Image

# Mediapipe Face Detection 초기화
mp_face_detection = mp.solutions.face_detection

# ChromaDB PersistentClient 초기화
path = './faces'  # 데이터베이스 저장 경로
client = chromadb.PersistentClient(path)
db = client.get_or_create_collection(
    name='facedb',
    metadata={
        "hnsw:space": 'cosine',
    },
)

# imgbeddings 초기화
ibed = imgbeddings()

# 유사도 임계값 설정
similarity_threshold = 0.10

# VideoCapture 초기화
cap = cv2.VideoCapture(0)
frame_count = 0  # 프레임 카운트 초기화

# 이전에 탐지된 얼굴 바운딩 박스 및 라벨
previous_bboxes = []
previous_labels = []

def generate_new_label():
    """ChromaDB에 저장된 얼굴 벡터 수 기반으로 새로운 레이블 생성"""
    count = db.count()  # 현재 저장된 총 문서 수 확인
    new_label = f"a{count + 1:05d}"  # a00001, a00002 형태로 생성
    return new_label

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        frame_count += 1  # 프레임 카운트 증가

        # 얼굴 탐지
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        current_bboxes = []
        labels = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                current_bboxes.append((x1, y1, x2, y2))

            if len(current_bboxes) != len(previous_bboxes) or frame_count % 50 == 0:
                labels = []
                for bbox in current_bboxes:
                    x1, y1, x2, y2 = bbox
                    cropped_face = image[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        cropped_face_rgb = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                        embedding = ibed.to_embeddings(cropped_face_rgb)[0]

                        # ChromaDB에서 유사한 얼굴 검색
                        search_results = db.query(
                            query_embeddings=[embedding.tolist()],
                            n_results=1,
                            include=["distances", "metadatas"]
                        )
                        print('search_results:', search_results)
                        try:
                            if (search_results["distances"] and 
                                len(search_results["distances"][0]) > 0 and 
                                search_results["distances"][0][0] < similarity_threshold):
                                label = search_results["metadatas"][0][0]["filename"]
                            else:
                                label = "Unknown"
                        except (IndexError, KeyError):
                            label = "Unknown"

                        labels.append(label)
            else:
                labels = previous_labels

            # 바운딩 박스 및 라벨 표시
            for bbox, label in zip(current_bboxes, labels):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)

        # 이전 바운딩 박스 및 라벨 업데이트
        previous_bboxes = current_bboxes
        previous_labels = labels

        # 이미지 출력
        cv2.imshow('Real-time Face Detection', image)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 또는 Q 키
            print("Exiting...")
            break
        elif key == 13:  # ENTER 키
            for bbox, label in zip(current_bboxes, labels):
                if label == "Unknown":
                    x1, y1, x2, y2 = bbox
                    cropped_face = image[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        cropped_face_rgb = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                        embedding = ibed.to_embeddings(cropped_face_rgb)[0]
                        
                        # 새로운 레이블 생성
                        new_label = generate_new_label()
                        
                        # ChromaDB에 새로운 얼굴 저장
                        db.add(
                            embeddings=[embedding.tolist()],
                            metadatas=[{"filename": new_label}],
                            ids=[new_label]
                        )
                        print(f"New face embedding saved with label '{new_label}'")
                        
                        # 현재 프레임에 새로운 라벨을 즉시 반영
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, new_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cap.release()
cv2.destroyAllWindows()
