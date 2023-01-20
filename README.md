# Image Captioning Baseline
본 소스 코드는 '국립국어원 인공 지능 언어 능력 평가' 시범 운영 과제 중 '표 기반 문장 생성' 과제 베이스라인 모델 및 학습과 평가를 위한 코드입니다.

학습 및 추론, 평가는 아래의 실행 방법(How to Run)에서 확인하실 수 있습니다.  

|Model|ROUGE-1|BLUE|
|:---:|---|---|
|ViT + KoGPT2|0.5071|0.7419|

## 디렉토리 구조(Directory Structure)
```
# 학습에 필요한 리소스들이 들어있습니다.
resource
├── data
└── tokenizer

# 실행 가능한 python 스크립트가 들어있습니다.
run
├── infernece.py
└── train.py

# 학습에 사용될 커스텀 함수들이 구현되어 있습니다.
src
├── data.py     # torch dataloader
├── module.py   # pytorch-lightning module
└── utils.py
```

## 데이터(Data)
### 제공 데이터
```
{
    "id": "nikluge-2022-image-dev-000001",
    "input": "K0A0001",
    "output": [
        "얼룩덜룩한 털빛의 고양이 한 마리가 길가에 앉은 채 고개를 들고 위를 보고 있다.",
        "고양이가 보도블록 위에 앉아서 무언가를 올려다보고 있다.",
        "보도블록 한가운데 흑갈색 얼룩 고양이가 위쪽을 응시한 채 앉아 있다.",
        "삼색 고양이가 보도블록에서 위를 바라보고 있다.",
        "길고양이가 도보 위에 앉아서 하늘을 쳐다보고 있다."
    ]
}
{
    "id": "nikluge-2022-image-dev-000002",
    "input": "K0A0018",
    "output": [
        "한 어린 남자아이가 동물원에서 호랑이 조형물 위에 올라타 있다.",
        "니트를 입은 어린아이가 작은 호랑이 모형에 올라타 있다.",
        "베이지 색 카디건을 입은 남자 아이가 공원에서 새끼 호랑이 동상 등에 올라타 앉아 있다.",
        "아기 호랑이 조형물 위에 남자아이가 올라탔다.",
        "사람이 커다란 호랑이 모형 앞의 아기호랑이 모형 위에 타고 있다."
    ]
}
...
```
`input`은 이미지 파일명입니다.

## 설치(Installation)
Execute it, if mecab is not installed
```
./install_mecab.sh
```

Install python dependency
```
pip install -r requirements.txt
```

## 실행 방법(How to Run)
### 학습(Train)
```
python -m run train \
    --output-dir outputs/ttt \
    --tokenizer "resource/tokenizer/kobart-base-v2(ttt)" \
    --seed 42 --epoch 10 --gpus 4 --warmup-rate 0.1 \
    --max-learning-rate 2e-4 --min-learning-rate 1e-5 \
    --batch-size=32 --valid-batch-size=64 \
    --logging-interval 100 --evaluate-interval 1 \
    --wandb-project <wandb-project-name>
```
- 기본 모델은 `google/vit-base-patch16-224-in21k`와 `skt/kogpt2-base-v2`를 이용합니다.
- 학습 로그 및 모델은 지정한 `output-dir`에 저장됩니다.

### 추론(Inference)
```
python -m run inference \
    --model-ckpt-path outputs/ttt/<your-model-ckpt-path> \
    --output-path test_output.jsonl \
    --batch-size=64 \
    --output-max-seq-len 512 \
    --num-beams 5 \
    --device cuda:2
```
- `transformers` 모델을 불러와 inference를 진행합니다.
- Inference 시 출력 데이터는 jsonl format으로 저장되며, "output"의 경우 입력 데이터와 다르게 `list`가 아닌 `string`이 됩니다.

### 채점(scoring)
```
python -m run scoring \
    --candidate-path <your-candidate-file-path>
```
- Inference output을 이용해 채점을 진행합니다.
- 기본적으로 Rouge-1과 BLEU를 제공합니다.

## Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
SKT-AI/KoGPT2 (https://github.com/SKT-AI/KoGPT2)  
국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
