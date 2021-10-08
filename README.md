# MOAI 2021 Body Morphometry AI Segmentation Online Challenge : Kidney and Tumor
Final Ranking : **6th** / 27 teams



### Abstract

----

**MOAI 2021 Body Morphometry AI Segmentation Challenge**는 원광대학교병원과 서울아산병원의 공동 연구 중인 Body Morphometry와 관련된 주제로, CT 영상에서 신장과 신장암에 대하여 인공지능 기술을 기반으로 자동 Semantic Segmentation 기술 개발 및 정량적 평가 수행을 통해 의료 인공지능 연구 개발 활성화를 목표로 진행



### Introduction

---------

신장암은 전 세계적으로 발생률이 꾸준히 증가하고 있으며, 매년 400,000건 이상의 새로운 신장암 환자가 발생하고 있습니다. 신장암은 종양 발생 후 상당 기간 증상이 없는 경우가 많습니다. 특히, 암이 작을 때에는 증상이 거의 없으며, 암이 진행되어 신장 내부조직의 상당 부분을 침범하거나 주위 장기를 밀어내거나 다른 장기로 전이되었을 때 증상이 나타나는 경우가 많습니다. 흔하게 전이되는 곳은 폐, 림프절, 간, 뼈, 부신, 반대편 신장, 뇌등이 있습니다.

 신장암의 진단을 위해 복부 초음파, 컴퓨터단층촬영(CT), 자기공명영상(MRI), 양전자 단층촬영(PET) 등을 시행하며, 양성과 악성 구분이 모호하다면 조직 검사를 시행하는 경우도 있습니다. 복부CT는 신장암의 진단과 병기 결정을 위하여 가장 중요한 검사로 활용되고 있으며, 신장 주변에 위치한 장기로의 침범이나 전이 유무를 파악하여 암의 진행 정도를 결정하게 됩니다.

 신장암은 대부분의 경우 수술을 통해 치료하게 되며, 최근에는 신장의 기능을 보존할 수 있는 부분 신절제술이 표준적 치료로 자리 잡고 있습니다. 신장과 종양 형태는 매우 다양하기 때문에 신장과 종양의 형태를 고려한 수술 계획 기술과 수술 결과와의 연관성, 그리고 이러한 기술들의 발전에 대한 관심이 많습니다.

 신장과 신장암의 Semantic Segmentation 기술은 이러한 노력과 발전을 위한 유망한 도구로써 사용되어 집니다. 하지만, 신장과 종양의 형태학적 이질성(Morphological heterogeneity)은 이러한 Semantic Segmentation 기술 개발에 어려움을 야기합니다. 이 대회의 목표는 신뢰할 수 있는 신장과 신장암의 인공지능 기반 Semantic Segmentation 방법론과 기술 개발의 가속화입니다.



### Method

---

- 2D segmentation - baseline.py
  - 적은 Data의 갯수에 대해 Overfitting을 완화하고자, CT Data의 특징을 이용하여 **Random Windowing** 방식을 적용하였습니다.
  - 환자에 대한 CT Data Case는 총 100 Case이므로, 환자마다 다른 장기의 크기, 양상은 모두 다를 수 있습니다. 이에 대한 Variation을 증가시키고자 Augmentation step에서 **Spatial Transformation**을 다양하게 적용하여 학습을 진행하고자 하였습니다.
  - 유사한 대회인 KiTS-19 Challenge를 참고하여 **z-normalization** value를 선정하였습니다.
  - Class Imbalance : Data에 대해 분석할 때, Kidney에 비해 tumor는 상대적으로 적은 Region으로 존재하였습니다. 이에 대한 **Class Imbalance를 Loss Function으로 해결**하고자 하였습니다.
  - Model : **U-Net architecture with ResNet-50**
  - Loss Function : **DiceCELoss(0.5*DiceLoss + 0.5*WeightedCrossEntropyLoss[0.3, 0.3, 0.4])**
  - Optimization : **Adam**



다음의 방법들은 Baseline을 토대로 접근방법을 수정하여 성능을 개선하고자 하였습니다.

- 2D segmentation - based_patch.py
  - Tumor ROI에 대해 부족한 성능을 개선하고자 Kidney와 Tumor가 포함된(Positive) Patch와Background(Negative) Patch를 **9 : 1** 비율로 사용하여 접근하였습니다.
- 2D segmentation - fastai.py
  - fast.ai 라이브러리는 적은 CODE Line으로도 효과적으로 최적화를 진행하고 **Self-Attention, Mish Activation Function**등 최신 학습 Scheme을 적용할 수 있는 강력한 라이브러리입니다. baseline CODE와 비교하여 Architecture, Optimization 등의 차이가 얼마나 존재하는지를 확인할 수 있었습니다.

- 3D segmentation.py
  - Medical Data는 기본적으로 3D Data이기에 제공된 DICOM file을 3D Array로 구성하여 3D Segmentation을 진행하였습니다.
  - 3D로 접근할 경우, 활용할 수 있는 Case는 Validation Case를 제외하면 80 Case 이기에 해당 Challenge에서 효과적으로 이용하기 어려운 문제점이 존재했습니다.



### Post-processing

---

![](https://github.com/hwanseung2/kidney-tumor-segmentation/blob/main/img/img1.png)

- **Remove Small Objects**

  - Segmentation 시, 작은 Region의 False Positive들이 상당 수 존재하였는데, 8-Connected Component들 중 100-pixel이 넘지 않는 Region들은 제거하였습니다.

    

- **Remove False Positive by considering z-axis**

  - 3D Data를 2D Segmentation 방식으로 접근했을 경우, z-axis는 고려하지 못하게 됩니다. 이를 개선하고자 Prediction이 존재할 때, 위&아래 2 Slices까지 고려하여 Prediction이 존재하지 않을 경우, 사이에 존재하는 Prediction을 제거하였습니다.



![](https://github.com/hwanseung2/kidney-tumor-segmentation/blob/main/img/img2.png)

- Interpolate False Negatives

  - 위의 방식과 유사한 방법으로, 위&아래 2 Slices가 Prediction이 존재함에도 불구하고 중간의 Slice가 Prediction이 존재하지 않을 경우, 3D Data의 연속성을 고려하여 Region interpolation을 진행하였습니다.

    

- **Remove Tumor Predictions**

  - Tumor가 Kidney를 벗어나 존재하는 Case들도 충분히 존재하였지만 예측값들을 분석했을 때, True Positive보다 False Positive가 대다수였습니다. 이 문제점을 개선하고자 Tumor Prediction의 경우, 해당 Slice에 Kidney의 유무를 고려하였습니다.
