# det-lab
## requirements
```pytorch > 1.8.1```  
```albumentations```  
```PyYaml```

## Config Train Parameters

기본 설정값은 ./configs/default_settings.yaml에 정의됨.  
train 시 사용하는 yaml 파일에 정의되지 않은 파라미터는 default_settings 에서 가져옴.

[default_settings.yaml](./configs/default_settings.yaml)


    // ./configs/*.yaml 파일 수정
    // ex) cls_frostnet -> default_settings 파라미터를 업데이트 해서 사용
    model : 'FrostNet' # 모델 선택자 -> utils/module_select.py / get_model 에서 dictionary 로 정의
    dataset_name : tiny-imagenet 
    classes : 200
    epochs: 1000
    data_path : '/host_dir/fssv1/tiny_imagenet-200/'
    save_dir : './saved'
    workers: 16



## train classifier
    python train_classifier.py --cfg configs/cls_frostnet.yaml


## train detector
    TBD