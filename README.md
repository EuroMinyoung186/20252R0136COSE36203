# 20252R0136COSE36203

2025-2 가을학기 기계학습 팀 프로젝트 입니다.

---
## 소개
온라인 패션 시장의 급성장으로 선택지는 늘어났지만, TPO와 감성까지 반영한 의상을 찾는 데에는 여전히 정보 과부하와 높은 탐색 비용이 존재한다. 본 프로젝트는 LLM을 활용해 사용자의 모호한 자연어 질의를 의미 기반 검색 질의로 구조화하고, Retriever와 결합한 상황 인식형 패션 추천 시스템을 통해 이러한 한계를 극복하고자 한다. 나아가 AI 기반 가상 피팅 기술을 결합해 추천된 의류의 실제 착용감을 시각적으로 검증함으로써, 발견부터 구매 결정까지 이어지는 통합 패션 AI 플랫폼을 구축하는 것을 목표로 한다.

## 방법론

![image]

### 데이터셋
Tag가 붙어있는 인터넷 짤들을 크롤링하여 수집. (2runzzal.com, Jjalbang.today, jjalbang.net 사이트에서 데이터 수집)

### 모델
![image](https://github.com/AIKU-Official/Magical_SOLARgodong/assets/80198264/cb47d752-54ef-4227-adb0-e6966fdab7ea)

Sbert를 사용하여 입력 query와 이미지 태그들 사이의 유사도를 도출하고 상위 K개를 추출

LLM(OPEN-SOLAR-KO-10.7B)를 사용하여 이미지 태그들이 감정이나 상황 등의 정보를 담을 수 있도록 가다듬는 역할과 Sbert를 사용하여 추출한 특정 query에 맞는 상위 K개의 이미지 중 하나의 이미지를 최종 output으로 결정하는 역할을 담당

BLIP-2를 사용하여 이미지 캡셔닝을 진행. 기존 크롤링으로 수집한 태그들로는 부족한 이미지에 대한 설명을 보충하는 역할을 담당.

## 환경 설정
Google Colab Pro A100

## TODO List
- [x] Modulization
- [ ] Reveal our Meme Dataset
- [ ] Update Summarization Model
- [ ] Making DB Systems
- [ ] Training Retriever

## 사용 방법
1. 데이터셋을 준비해주세요.
2. 데이터셋 전처리를 해주세요. (공개할 데이터셋은 2번까지 처리되어 있을 예정입니다.)
  ```
  cd data
  python data_preprocessing.py
  ```
3. 데이터셋을 바탕으로 사용자 query를 입력해주세요.
 ```
 python main.py --user_input 꼭 이번에는 성공할거야
                  --filename input.csv
 ```

## 예시 결과
![image](https://github.com/AIKU-Official/Magical_SOLARgodong/assets/80198264/9a78913d-a947-4b8d-b981-44a2ecdb1d19)

![image](https://github.com/AIKU-Official/Magical_SOLARgodong/assets/80198264/661a9f15-b312-450f-b3d6-8029eeb4b2e0)

![image](https://github.com/AIKU-Official/Magical_SOLARgodong/assets/80198264/17212892-7f20-4a2e-8be1-54ca8a60e073)

## 팀원
- [김민영](https://github.com/EuroMinyoung186/)
- [박서현](https://github.com/emiliebell)
- [박정규](https://github.com/juk1329)
