import cv2
import chromadb
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np

# MTCNN 및 InceptionResnetV1 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# ChromaDB PersistentClient 초기화
path = './faces'  # 데이터베이스 저장 경로
client = chromadb.PersistentClient(path)
db = client.get_or_create_collection(
    name='facedb',
    metadata={"hnsw:space": 'cosine'}
)

# 유사도 임계값 설정
similarity_threshold = 0.09

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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    frame_count += 1  # 프레임 카운트 증가

    # 얼굴 탐지 및 바운딩 박스 얻기
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(image_pil)

    current_bboxes = []
    labels = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            current_bboxes.append((x1, y1, x2, y2))

        if len(current_bboxes) != len(previous_bboxes) or frame_count % 50 == 0:
            labels = []
            for box in current_bboxes:
                x1, y1, x2, y2 = box
                cropped_face = image[y1:y2, x1:x2]
                if cropped_face.size > 0:
                    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                    aligned_face = mtcnn(cropped_face_pil)

                    if aligned_face is not None:
                        # aligned_face = aligned_face.unsqueeze(0).to(device)
                        embedding = resnet(aligned_face.to(device)).detach().cpu().numpy()[0]
                        
                        print('embedding.shape')
                        print(embedding.shape) 
                        
                        # ChromaDB에서 유사한 얼굴 검색
                        search_results = db.query(
                            query_embeddings=[embedding.tolist()],
                            n_results=1,
                            include=["distances", "metadatas"]
                        )
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
                    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                    aligned_face = mtcnn(cropped_face_pil)

                    if aligned_face is not None:
                        embedding = resnet(aligned_face.to(device)).detach().cpu().numpy()[0]
                        

                        # 새로운 레이블 생성
                        new_label = generate_new_label()

                        # ChromaDB에 새로운 얼굴 저장
                        db.add(
                            embeddings=[embedding.tolist()],
                            metadatas=[{"filename": new_label}],
                            ids=[new_label]
                        )
                        print(f"New face embedding saved with label '{new_label}'")

cap.release()
cv2.destroyAllWindows()
