# Team Medic(CV-16)

---

## Project Overview

- Project Period:
2022.04.14 ~ 2022.04.21
- Project Wrap Up Report:
    
    [CV 데이터제작_CV_팀 리포트(16조)](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0deed663-df50-492e-a4eb-3a30f6cfa1c0/CV_%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%A0%9C%EC%9E%91_CV_%ED%8C%80_%EB%A6%AC%ED%8F%AC%ED%8A%B8%2816%EC%A1%B0%29.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220513%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220513T073323Z&X-Amz-Expires=86400&X-Amz-Signature=f4bb6d9571751d8ffe8510afb6bd974dafdfa14a3442e4de4f77c8d368c04eef&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22CV%2520%25EB%258D%25B0%25EC%259D%25B4%25ED%2584%25B0%25EC%25A0%259C%25EC%259E%2591_CV_%25ED%258C%2580%2520%25EB%25A6%25AC%25ED%258F%25AC%25ED%258A%25B8%2816%25EC%25A1%25B0%29.pdf%22&x-id=GetObject)
      

## 🔎 **글자 검출 대회**

![Untitled](https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/blob/main/src/Competition%20Title%20Heading.png)  

### 😎 Members

| 권순호 | 서다빈 | 서예현 | 이상윤 | 전경민 |
| --- | --- | --- | --- | --- |
| [Github](https://github.com/tnsgh9603) | [Github](https://github.com/sodabeans) | [Github](https://github.com/yehyunsuh) | [Github](https://github.com/SSANGYOON?tab=repositories) | [Github](https://github.com/seoulsky-field) |   


### 🌏 Contribution

- 권순호: Conduct experiments on training Epoch, Batch size, albumentation
- 서다빈: Experiment with various learning rate schedulers, apply straug augmentation, and concat multiple datasets
- 서예현: boostcamp's data implementation, split train/val in ICDAR17 dataset, dealt with polygon bounding box issues
- 이상윤: ICDAR 2019 dataset,add validation process
- 전경민: Hyperparameter Tuning(with Auto-ML), Data Augmentation   

### **❓Problem Definition**

![Untitled](https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/blob/main/src/Problem%20Definition.png)

- 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.
- OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.
    1. 본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
    2. 예측 csv 파일 제출 (Evaluation) 방식이 아닌 **model checkpoint 와 inference.py 를 제출하여 채점**하는 방식입니다.
    3. 대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 상세 내용은 **하단 Rules를 참고**해주세요**.**
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format   

### 🚨 Competition Rules

본 대회는 데이터를 구성하고 활용하는 방법에 집중하는 것을 장려하는 취지에서, 제공되는 베이스 코드 중 모델과 관련한 부분을 변경하는 것이 금지되어 있습니다. 이에 대한 세부적인 규칙은 아래와 같습니다.

- 베이스라인 모델인 EAST 모델이 정의되어 있는 아래 파일들은 변경사항 없이 그대로 이용해야 합니다.
    - model.py
    - loss.py
    - east_dataset.py
    - detect.py
- 변경이 금지된 파일들의 내용을 이용하지 않고 모델 관련 내용을 새로 작성해서 이용하는 것도 대회 규정에 어긋나는 행위입니다.
- 이외의 다른 파일을 변경하거나 새로운 파일을 작성하는 것은 자유롭게 진행하셔도 됩니다.
    - [예시] dataset.py에서 pre-processing, data augmentation 부분을 변경
    - [예시] train.py에서 learning rate scheduling 부분을 변경  

### 💾 Datasets

- ICDAR17 & ICDAR19 Korean dataset
- Boostcamp’s self-annotated korean dataset
- Dataset Examples   
 <img src="https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/blob/main/src/img_4002.jpg" height="200"/> <img src="https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/blob/main/src/img_4351.jpg" height="200"/> <img src="https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/blob/main/src/img_1135.jpg" height="200"/>   

### 💻 **Development Environment**

- GPU: Tesla V100
- OS: Ubuntu 18.04.5LTS
- CPU: Intel Xeon
- Python : 3.8.5  

### 📁 Project Structure

```markdown
├─ level2-data-annotation_cv-level2-cv-16
│  ├─ download_ICDAR
│  ├─ nni
│  │   ├─ README.md
│  │   ├─ config.yml
│  │   └─ search_space.json
│  ├─ trained_models
│  │   └─ latest.pth
│  ├─ model.py
│  ├─ loss.py
│  ├─ train.py
│  ├─ train_with_valid.py
│  ├─ inference.py
│  ├─ dataset.py
│  ├─ detect.py
│  ├─ deteval.py
│  ├─ east_dataset.py
│  ├─ convert_mlt.py
│  └─ requirements.txt
└─ input
   └─ data
        ├─ ICDAR2017_Korean
        │   ├─ ufo
        │   │    ├─ train.json
        │   │    └─ val.json
        │   ├─ train
        │   └─ valid
        └─ ICDAR2017_MLT/raw
            ├─ ch8_training_images
            ├─ ch8_training_gt
            ├─ ch8_training_gt
            └─ ch8_validation_gt
```

Input directory is removed from the github repository due to memory. The user must create the directory and follow the steps below in order to use the data.

### 👨‍🏫 Evaluation Methods

평가방법은 DetEval 방식으로 계산되어 진행됩니다.

DetEval은, 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법 중 하나 입니다.

평가가 이루어지는 방법은 다음과 같습니다.

[Evaluation methods](https://www.notion.so/Evaluation-methods-700f3a9352574fed8663de74a8f2d5b3)  

### 💯 Final Score

![Untitled](https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/blob/main/src/Final%20Score.png)

- final result
    1. Hyperparameter: epoch=200, optimizer=AdamW, scheduler=CosineAnnealingLR, lr=0.001
    2. Datasets: ICDAR_2017 & ICDAR_2019 Korean dataset
    3. Augmentation: CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.6), InvertImg(p=0.4)  

## 👀 How to Start

- Downloading the github repository

```powershell
git clone https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16.git
cd level2-data-annotation_cv-level2-cv-16
```

- How to use ICDAR17 Korean Datasets for Train & Val

prerequisites: over 5GBs of memory left in the hard disk

```powershell
cd download_ICDAR
sh download.sh
sh unzip_train.sh
sh unzip_val.sh
```

FYI: follow the steps in [issue #17 comments](https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/issues/17#issuecomment-1098864585) and [issue #21](https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/issues/21)

- Installing the requirements for training(Note [issue #8](https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-16/issues/8))

```powershell
pip install -r requirements.txt

apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
```

- Training the model

```powershell
python train.py
```

- Training the model with validation set

```powershell
python train_with_valid.py
```
  
### 📄 Experiments & Submission Report

[Notion](https://www.notion.so/W13-14-Data-Annotation-Project-Team-Medic-e18cd7ceb89a4923a4d471c327cdbc21)
