# 20252R0136COSE36203

2025-2 가을학기 기계학습 팀 프로젝트 입니다.

---
## 소개
온라인 패션 시장의 급성장으로 선택지는 늘어났지만, TPO와 감성까지 반영한 의상을 찾는 데에는 여전히 정보 과부하와 높은 탐색 비용이 존재한다. 본 프로젝트는 LLM을 활용해 사용자의 모호한 자연어 질의를 의미 기반 검색 질의로 구조화하고, Retriever와 결합한 상황 인식형 패션 추천 시스템을 통해 이러한 한계를 극복하고자 한다. 나아가 AI 기반 가상 피팅 기술을 결합해 추천된 의류의 실제 착용감을 시각적으로 검증함으로써, 발견부터 구매 결정까지 이어지는 통합 패션 AI 플랫폼을 구축하는 것을 목표로 한다.

## 방법론

![image](https://github.com/user-attachments/assets/1655ab6f-0724-40f5-b183-7452347dd931)

### 데이터셋
Multimodal-Deepfashion과 VITON-HD 데이터셋을 활용하여, 옷 추천과 가상 피팅을 진행함.

### 모델
Cross-Attention을 활용한 Mapping Layer와 query-document cosine similarity를 통한 상황-옷 추천 모듈을 개발함.

[StableVITON](https://github.com/rlawjdghek/StableVITON)을 활용한 Virtual Try-on까지 가능한 pipeline 구성

## 환경 설정
NVIDIA TITAN RTX x 8


## 사용 방법
1. 데이터셋을 준비해주세요. 데이터셋은 다음 drive에서 다운로드 가능합니다. [Google Drive](https://drive.google.com/drive/folders/1NP4U68feUHDyWosfTOGfcI5VFBBX29ny?hl=ko)
  ```text
  project-root/
  ├── data/
  │   ├── cloth/
  │   │      ├── img/
  │   │      ├── segm/
  │   │      └── captions.json
  │   └── human/
  │         └──VITON-HD/
  ├── fashion_recommendation/
  │   ├── model.py
  │   └── train.py
  ├── StableVITON/
  │   └── ckpt/
  │         ├── VITONHD_VAE_finetuning.ckpt
  │         ├── VITONHD_PBE_pose.ckpt
  │         └── VITONHD.ckpt
  ```

2. 데이터셋 전처리를 해주세요. (google drive에 전처리가 완료된 코드까지 넣어두었습니다.)
  ```
  sh llm_generation.sh
  ```
3. 데이터셋을 바탕으로 추천 모델을 학습시켜주세요.
 ```
 sh train.sh
 ```
4. inference_gt.sh를 실행하시면, virtual try-on까지 한번에 추론 가능합니다.
 ```
 sh inference_gt.sh
 ```
5. 추천 시스템 모델의 성능을 test.sh로 실행 가능합니다.
 ```
 sh test.sh
 ```

## 팀원
- [김민영](https://github.com/EuroMinyoung186/)
- 장충준
- 진혜성
