# recipe-recommendation

소프트웨어응용 추천 시스템 설계 프로젝트

## GraphSAGE 2단계 추천 엔진 구조

#### 텍스트 입력 인코딩

`BGETextEncoder`: Recipe와 Ingredient 텍스트에 적용해 768차원 정규화 임베딩 생성

#### 1단계

`RecipeEncoder`: `Recipe —SIMILAR— Recipe` 연결의 평가 점수를 `label / 4` 가중치로 집계해 레시피 임베딩과 레시피 쌍의 유사도 점수 학습

#### 2단계

`IngredientLinkEncoder`: 1단계 레시피 임베딩을 고정하고 `Recipe —CONTRIBUTION— Ingredient` 연결의 0~1 기여도를 학습.\
기존 연결의 기여도는 양방향 이웃 집계에서 가중치로 사용

```python
from src.engine import BGETextEncoder
from torch_geometric.data import HeteroData

encoder = BGETextEncoder()
recipe_texts = ["Title: tomato pasta", "Steps: 1. Boil the pasta 2. Heat the olive oil in a pan 3. Add pasta and tomato sauce into the pan and toss for a minute"]
ingredient_texts = ["tomato sauce", "pasta", "olive oil"]

data = HeteroData()
data["Recipe"].x = encoder.encode_texts(recipe_texts)
data["Ingredient"].x = encoder.encode_texts(ingredient_texts)
```

### 실행 방법

1. Python에서 `torch`, `torch-geometric`, `sentence-transformers` 설치
    - 최소 버전: `requirements.min.txt`
    - Colab 사용 시: `requirements.colab.txt`

```bash
pip install -r [파일명]
```

2. 프로젝트 루트에서 아래 명령어 순서대로 실행

```bash
python -m training.stage1 prepare
python -m training.stage1 train
python -m training.stage2 --data [파일명]
python -m training.evaluate --data [파일명]
python -m unittest discover -s tests
```

모델 저장 경로

- 1단계: `checkpoints/stage1.pt`
- 2단계: `checkpoints/stage2.pt`

사용 가능한 옵션

- `--epochs`
- `--lr`
- `--output`(체크포인트 저장할 경로 지정)
- `--stage1-checkpoint`(2단계 입력으로 들어갈 체크포인트 경로 지정)

### 데이터 형식

\* `examples/toy_graph.json` 참고

`recipe_features`, `ingredient_features`: 순서가 노드 ID인 2차원 숫자 배열

`similar_edges`: `[recipe_a, recipe_b, weight]` (`weight`는 0(전혀 유사하지 않음)~1(완전히 유사함) 사이 실수)

`contribution_edges`: `[recipe_id, ingredient_id, contribution]` (`contribution`은 0~1 사이 실수)

- 각 edge는 두 방향 중 하나만 기록하면 실제 그래프에서는 양방향으로 연결 생성

`toy_graph.json`에서 `stage1_train/test` 항목은 `[recipe_a, recipe_b, label]`, `stage2_train/test` 항목은 `[recipe_id, ingredient_id, contribution]` 형식. 동일한 레시피–재료 쌍은 각 목록에 한 번만 기록

- 1단계 유사도 라벨: 0~1 사이 실수 (예: 0.75)
- 2단계 기여도 라벨: 0~1 사이 실수 (예: 주재료 0.9, 보조 재료 0.4, 검증된 부적합 재료 0.0)

#### 학습 데이터 제작 시 유의 사항

- 재료 특징(이름, 카테고리 등)과 레시피–재료 쌍별 기여도 정답 라벨(<-정빈님 마스킹 결과에 포함됨) 필요.

- 연결되지 않은 쌍의 예상 연결 점수를 계산하기 위한 목적이므로 연결이 없다고 해서 무조건적으로 negative로 라벨링하지 말 것

- 학습할 때 연결을 비운 입력도 함께 사용하여 직접 연결되지 않은 Ingredient도 자신의 feature를 통해 임베딩 생성 가능

- 학습 및 평가 대상으로 사용할 정답 라벨은 메시지 전달용 edge 목록에서 제외할 것 (+ data.py의 로더에서 검사함)

### 추론

1. 1단계(SIMILAR)와 2단계(CONTRIBUTION) 체크포인트로 모델 복원

2. `IngredientLinkEncoder.edge_weights(data, recipe_embeddings, pairs)` 호출
    - `pairs`:

        `[2, 후보 수]` 모양의 `torch.long` 텐서. 각 후보 연결의 예상 기여도(0~1)를 반환

3. `IngredientLinkEncoder.recommend_ingredients(data, recipe_embeddings, recipe_id, top_k, save_predicted_edges=False)` 호출

    이미 연결된 재료를 제외하고 `(ingredient_ids, weights)`를 예상 기여도가 높은 순으로 반환
    - `save_predicted_edges`:

        `True`: 반환할 추천 연결을 점수 계산 후 `data`에 추가, edge 속성 `edge_weight`에 가중치 기록, `is_predicted`에 예측 여부 기록

        `False` (기본값): 그래프를 변경하지 않음

- 예측한 연결 점수는 다음 계산의 이웃 집계에서 제외해 자기 자신을 강화하는 문제 방지

- 학습, 추론 시 전체 그래프를 메모리에 올려 실행

---

## 1단계: CSV 파일로 레시피 유사도 학습

`data/recipes_masked.csv`의 전체 고유 레시피를 노드로 만들고, `data/labeled_pairs.csv`의 0~4점 평가를 0~1 라벨로 변환한다. 레시피 ID가 중복되면 첫 번째 행을 사용한다. BGE 입력 텍스트는 레시피 이름과 전체 조리 단계(`steps`)로 구성한다.

```bash
# 명령어 예시
pip install -r requirements.min.txt
python -m training.stage1 prepare
python -m training.stage1 train
python -m training.stage1 evaluate
python -m training.stage1 predict --a-id 172520 --b-id 306863
python -m training.stage1 top-similar --recipe-id 172520 --top-k 10
```

- `prepare`: BGE 모델을 처음 사용할 때 다운로드, 인코딩 결과와 점수별 비율을 유지하면서 나눈 train/val/test 쌍을 `data/stage1.pt`에 저장
    - 옵션: `--recipes`, `--pairs`, `--output`, `--seed`, `--batch-size`

- `train`: 학습 모델 생성 (결과 파일: `checkpoints/stage1.pt`)
    - 옵션: `--data`, `--output`, `--hidden`, `--epochs`, `--folds`, `--lr`, `--seed`

- `evaluate`: 생성한 모델 검증/테스트, 유사도 2점 이상을 positive로 간주하여 정확도 계산. 검증·테스트 쌍의 `a_id,b_id,similarity,predicted_similarity`(정답과 예측 모두 0~4점)를 `data/stage1_predictions.csv`에 저장 (`--output`으로 경로 변경 가능)
    - 옵션: `--data`, `--checkpoint`, `--output`

- `predict`: 입력 그래프에 직접 연결되지 않은 두 레시피도 0~1 유사도로 평가
    - 옵션: `--data`, `--checkpoint`, `--a-id`(required), `--b-id`(required)

- `top-similar`: 지정한 레시피(`--recipe-id`)를 제외하고 전체 레시피 중 유사도가 높은 상위 `--top-k`개를 원본 ID, 이름, 0~1 예측 유사도와 함께 출력. 이름은 `--recipes` CSV에서 읽어옴
    - 옵션: `--data`, `--checkpoint`, `--recipes`, `--recipe-id`(required), `--top-k`(default=10)

CSV 1단계 학습 손실과 학습 로그의 오차는 정규화된 0~1 유사도 기준의 RMSE로 계산한다. `evaluate`의 검증·테스트 오차도 같은 기준의 RMSE로 출력한다.

`similarity >= 1`인 **학습 쌍**만 `SIMILAR` 메시지 전달용 연결로 사용하며, 양방향 엣지의 가중치는 `similarity / 4`로 설정한다. 두 그래프 집계층 모두 이웃 메시지의 가중합을 이웃 수로 나누어 집계하므로 이웃이 하나여도 유사도에 따라 메시지 크기가 달라진다. 0점 쌍도 낮은 유사도를 학습하는 정답으로 사용한다.

학습 중에는 5개 폴드를 번갈아 예측하며, 현재 예측하는 폴드의 연결을 그래프에서 제외한다. 검증·테스트 쌍은 연결로 사용하지 않는다.

---

## 2단계: 레시피–재료 기여도

`IngredientLinkEncoder`는 1단계 레시피 임베딩을 고정하고 `Recipe —CONTRIBUTION— Ingredient` 관계의 0~1 기여도를 예측한다. 현재 `training.train`, `training.evaluate`, `training.data`와 `examples/toy_graph.json`은 2단계 엔진 검증을 위한 임시 JSON 예제다. 실제 2단계 데이터 형식은 추후 정할 예정이며, 이 JSON은 1단계 학습 입력으로 사용하지 않는다.

`src.engine.BGETextEncoder`는 레시피·재료 텍스트를 BGE 임베딩으로 변환한다. 2단계에 사용할 재료 특징과 기여도 라벨의 생성 방식은 실제 데이터 형식을 정할 때 연결한다.

---

## 테스트

```bash
python -m unittest discover -s tests
```
