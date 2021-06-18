# pstage_01_image_classification

  
### Dependencies
- torch==1.6.0
- torchvision==0.7.0
- timm
- albumentation
- pytorch_pretrained_vit
- efficientnet_pytorch                                                              

### Install Requirements
- `pip install -r requirements.txt`



### py files

- Datasets.py
  - 데이터 로더(train,test), augmentation 를 구현한 코드입니다.
  
- Models.py
  - 실험에 사용한 모델들을 class로 구현한 파일 입니다.  
  
- Trainer.py
  - 전반적인 train 실행하는 코드, Optimizer, Loss, gradupdate 등을 수행 합니다.  
  
- Train.py
  - model, augmentation method, optimizer hyperparameter를 dictionary 형태로 입력하고 학습을 진행하는 main 코드 입니다.
   
Train
```
python Train.py
```


- Inference.py
  - 학습된 모델을 가지고 결과를 제출하는 소스코드 입니다. 실행은 inference.ipynb 로 구현하였습니다.


### ipynb files

- EDA.ipynb
  - 데이터 EDA 관련 파일 입니다.
  
- inference.ipynb
  - 제출을 위한 추론을 진행하는 파일 입니다. Inference.py를 불러와서  앙상블, TTA를 수행할 수 있게 구현하였습니다.
  
- submission_report.ipynb
  - 예측된 결과들의 label분포도를 시각화한 파일입니다.

#### csv file

- model Chart.csv
  - 실험 결과들을 기록한 파일입니다.




