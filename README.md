# recipe-recommendation

소프트웨어응용 추천 시스템 설계 프로젝트

## GraphSAGE 2단계 추천 엔진 구조

#### 텍스트 입력 인코딩

`GTETextEncoder` (`thenlper/gte-base`): Recipe와 Ingredient 텍스트를 768차원 L2 정규화 임베딩으로 변환

- 최대 입력 길이(512토큰)를 넘는 텍스트는 잘라내지 않고 연속된 토큰 창으로 나눠 각각 인코딩 (`ordered_token_windows_v1`)
    - 창 하나는 특수 토큰 포함 512토큰 이하, 모든 문자가 정확히 한 창에 포함
    - 창별 임베딩을 창의 토큰 수로 가중평균한 뒤 다시 L2 정규화
- 1단계 레시피 텍스트: `Recipe: {이름}. Steps: {조리 단계를 공백으로 연결}`
- 2단계 재료 텍스트: `Ingredient: {이름}. Category: {카테고리}.`

#### 1단계

`RecipePairGraphSAGE`: `Recipe —SIMILAR— Recipe` 그래프 위에서 레시피 임베딩을 만들고, 레시피 쌍의 0~1 유사도를 예측

1. 입력 투영: `Linear → LayerNorm → GELU` (768 → hidden 64)
2. GraphSAGE 2층 (`WeightedSAGELayer`): 자기 벡터 + 이웃 벡터의 **엣지 가중 평균**(이웃 가중치 합으로 나눔), 각 층 뒤 `LayerNorm → GELU`, 2층은 잔차 연결
3. 대칭 쌍 MLP: 두 레시피 임베딩 `l`, `r`로 `[|l−r|, l*r, (l+r)/2]`를 만들고, 고정된 입력(GTE) 임베딩의 코사인 유사도를 학습 쌍 기준으로 표준화해 덧붙인 뒤 `Linear → GELU → Dropout → Linear → sigmoid`
4. 0~4점 변환: `floor(5 × 유사도)` (경계 0.2 / 0.4 / 0.6 / 0.8)

- 레시피 임베딩은 L2 정규화하지 않으며, 쌍 유사도는 `score()`로 계산
- 체크포인트는 `src.engine.load_recipe_model()`로 읽으며, 체크포인트에 기록된 1단계 데이터 pt의 해시가 실제 데이터 파일과 다르면 거부

#### 2단계

`IngredientLinkEncoder`: 1단계 레시피 임베딩을 고정하고 `Recipe —CONTRIBUTION— Ingredient` 연결의 0~1 기여도를 학습.\
기존 연결의 기여도는 양방향 이웃 집계에서 가중치로 사용

```python
from src.engine import GTETextEncoder
from torch_geometric.data import HeteroData

encoder = GTETextEncoder()
recipe_texts = ["Recipe: tomato pasta. Steps: Boil the pasta. Heat the olive oil in a pan. Add pasta and tomato sauce into the pan and toss for a minute."]
ingredient_texts = ["Ingredient: tomato sauce.", "Ingredient: pasta.", "Ingredient: olive oil."]

data = HeteroData()
data["Recipe"].x = encoder.encode_texts(recipe_texts)
data["Ingredient"].x = encoder.encode_texts(ingredient_texts)
```

### 실행 방법

1. Python에서 `torch`, `torch-geometric`, `sentence-transformers` 설치
    - 최소 버전: `requirements.min.txt`
    - Colab 사용 시: `requirements.colab.txt` (또는 `colab_pipeline.ipynb` 사용)

```bash
pip install -r [파일명]
```

2. 프로젝트 루트에서 아래 명령어 순서대로 실행

```bash
# 1단계 (이미 학습된 기본 데이터·체크포인트를 쓸 경우 생략 가능)
python -m training.stage1 prepare --recipes data/recipes_raw_v3.csv --output tmp/stage1_raw_gte.pt
python -m training.stage1 train --data tmp/stage1_raw_gte.pt --out tmp/stage1

# 2단계, 평가, 추천 (1단계를 새로 학습했다면 --stage1-data / --stage1-checkpoint 지정)
python -m training.stage2 prepare
python -m training.stage2 train --stage1-data tmp/stage1_raw_gte.pt --stage1-checkpoint tmp/stage1/model.pt
python -m training.evaluate --stage1-data tmp/stage1_raw_gte.pt --stage1-checkpoint tmp/stage1/model.pt
python -m src.main --stage1-data tmp/stage1_raw_gte.pt --stage1-checkpoint tmp/stage1/model.pt
python -m unittest discover -s tests
```

- CUDA GPU를 사용할 수 있으면 GTE 인코딩·2단계 학습·추론을 GPU에서, 없으면 CPU에서 실행 (`src.engine.DEVICE`). CPU에서는 레시피 GTE 인코딩에 시간이 오래 걸림
- 1단계 학습(`training.stage1 train`)은 결과 재현을 위해 CPU 단일 스레드·결정론적 연산으로 실행
- 임베딩 pt와 체크포인트는 CPU 텐서로 저장하므로 GPU가 없는 환경에서도 불러올 수 있음

기본 경로 (`training/data.py`)

| 구분             | 경로                                        | 비고                                                |
| ---------------- | ------------------------------------------- | --------------------------------------------------- |
| 1단계 데이터     | `data/stage1_rebuild_raw_gte.pt`            | 레시피 순서·train/val/test 분할의 기준, 읽기 전용   |
| 1단계 체크포인트 | `checkpoints/stage1_rebuild_raw_gte_h64.pt` | 위 데이터로 학습한 최종 모델 (hidden 64), 읽기 전용 |
| 2단계 임베딩     | `tmp/stage2_embeddings.pt`                  | `stage2 prepare` 결과                               |
| 2단계 체크포인트 | `tmp/stage2.pt`                             | `stage2 train` 결과                                 |

- 2단계 체크포인트는 학습에 사용한 1단계 모델과 짝이 맞아야 함. 1단계를 새로 학습했다면 2단계도 같은 1단계 경로로 다시 학습

### 데이터 형식

모든 입력은 원본 데이터 CSV 및 임베딩을 담은 pt 파일 사용

| 단계  | 파일                                | 내용                                                                                               |
| ----- | ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1단계 | `data/recipes_raw_v3.csv`           | 1단계·기여도 CSV에 필요한 원문 레시피 (`id`, `name`, `steps`, `ingredients`), `stage1 select` 결과 |
| 1단계 | `data/labeled_pairs_masked_v3.csv`  | 레시피 쌍 유사도 정답 (`a_id`, `b_id`, `similarity` 0~4점)                                         |
| 1단계 | `data/stage1_rebuild_raw_gte.pt`    | 레시피 ID 순서(`recipe_ids`), GTE 임베딩(2914 × 768), train/val/test 쌍 분할(2235/478/478)         |
| 2단계 | `data/ingredients_category.csv`     | 재료별 카테고리 (`category`, `ingredient`)                                                         |
| 2단계 | `data/ingredients_contribution.csv` | 레시피–재료 기여도 정답 (`ingredient`, `recipe_id`, `contribution` 0~1)                            |
| 2단계 | `tmp/stage2_embeddings.pt`          | `stage2 prepare` 결과: 재료 이름 순서, 재료 GTE 임베딩, 기여도 train/test 분할                     |

2단계 그래프(`training.data.load_dataset`) 구성

- Recipe 노드: 1단계 pt의 `recipe_ids` 순서. `SIMILAR`는 1단계 학습 분할에서 1점 이상인 쌍을 `similarity / 4` 가중치로 양방향 연결
- Ingredient 노드: 카테고리 CSV와 기여도 CSV에 등장하는 재료 이름의 합집합(소문자·앞뒤 공백 정규화, 알파벳순). 카테고리 CSV에 없는 재료(예: `salt and pepper`)는 카테고리 없이 노드로 포함
- 재료 특징: `Ingredient: {이름}. Category: {카테고리}.` 텍스트의 GTE 임베딩 (카테고리의 `_`는 공백으로 바꾸고, 카테고리가 없는 재료는 `Ingredient: {이름}.`만 사용)
- `CONTRIBUTION`: 기여도 CSV의 학습 분할(기본 85%)만 메시지 전달 그래프에 포함하고, 테스트 분할(15%)은 평가에만 사용
- 1단계 데이터에 없는 레시피(평가 쌍에 등장하지 않는 레시피, 현재 `111303` 1개)의 기여도 행은 경고를 출력하고 건너뜀. 재료 임베딩에 없는 재료는 오류
- 노드별 원본 표기는 `data["Recipe"].raw_id`, `data["Ingredient"].ingredient_name`에 저장

#### 학습 데이터 제작 시 유의 사항

- 재료 특징(이름, 카테고리 등)과 레시피–재료 쌍별 기여도 정답 라벨(<-정빈님 마스킹 결과에 포함됨) 필요.

- 연결되지 않은 쌍의 예상 연결 점수를 계산하기 위한 목적이므로 연결이 없다고 해서 무조건적으로 negative로 라벨링하지 말 것

- 학습할 때 연결을 비운 입력도 함께 사용하여 직접 연결되지 않은 Ingredient도 자신의 feature를 통해 임베딩 생성 가능

- 테스트 정답 연결은 메시지 전달 그래프에 넣지 않는다 (`load_dataset`에서 검사). 학습 중에는 1단계와 같이 폴드를 번갈아 예측하며, 현재 예측하는 폴드의 연결은 그래프에서 제외

### 추론

1. 1단계 체크포인트는 `load_recipe_model()`, 2단계 체크포인트는 `IngredientLinkEncoder`로 복원하고 `recipe_embeddings = recipe_model(data)` 계산

2. `IngredientLinkEncoder.edge_weights(data, recipe_embeddings, pairs)` 호출
    - `pairs`:

        `[2, 후보 수]` 모양의 `torch.long` 텐서. 각 후보 연결의 예상 기여도(0~1)를 반환

3. 대체 재료는 `recommend_substitutes()`, 같은 재료가 들어가는 유사 레시피는 `recommend_recipes()`로 찾는다. 두 함수 모두 그래프를 변경하지 않음

- 예측한 연결 점수는 다음 계산의 이웃 집계에서 제외해 자기 자신을 강화하는 문제 방지

- 학습, 추론 시 전체 그래프를 메모리에 올려 실행

#### 기여도가 비슷한 대체 재료 찾기

`recommend_substitutes()`: `data/ingredients_category.csv`에서 원래 재료와 같은 카테고리인 재료만 후보로 하고, 원래 재료의 기여도 `cont_a`와 후보의 예측 기여도 `cont_b` 사이의 `abs(cont_b - cont_a)`가 작은 순으로 `(ingredient_ids, weights)` 반환 (여기서 반환되는 `weights`는 `cont_b`와 같음)

```python
ingredient_ids, weights = ingredient_model.recommend_substitutes(
    data, recipe_embeddings,
    recipe_id=0,
    ingredient_id=4,  # 대체할 원래 재료의 노드 ID
    top_k=5,
    # original_contribution=0.4,  # cont_a를 직접 알고 있다면 지정 가능
)
```

- 기준 기여도: `original_contribution`을 우선 사용하고, 생략하면 원래 레시피–재료의 실제 연결 가중치 사용. 실제 연결이 없으면 해당 쌍의 기여도를 모델로 예측 (`is_predicted=True`인 연결은 기준값으로 사용하지 않음)

- 원래 재료, 해당 레시피에 이미 연결된 재료, 카테고리 CSV에 없는 재료는 후보에서 제외. 같은 카테고리 후보를 모두 평가한 뒤 차이순으로 최대 `top_k`개 반환, 그래프는 변경하지 않음

- 카테고리를 재료 이름으로 찾으므로 그래프에 `data["Ingredient"].ingredient_name` 저장 필요. 원래 재료가 카테고리 CSV에 없으면 `ValueError` 발생. 다른 카테고리 파일은 `category_path`로 지정 가능

#### 같은 재료가 들어가는 유사 레시피 찾기

`recommend_recipes()`: `data/ingredients_contribution.csv`에서 `ingredient_id`번 재료가 실제로 들어가는 레시피를 모두 찾고, 그중 `recipe_id`번 레시피와 유사도가 높은 순으로 최대 `top_k`개를 `(recipe_ids, similarities)`로 반환. 모델 예측 연결은 사용하지 않음

- `recipe_model`(1단계 모델)을 넘기면 학습된 쌍 유사도(0~1, `RecipePairGraphSAGE.score`)로, 생략하면 1단계 임베딩의 코사인 유사도로 순위를 매김

CSV는 원본 레시피 ID와 재료 이름으로 기록되어 있으므로 노드 번호와의 대응이 그래프에 저장되어 있어야 함. `load_dataset()`으로 만든 그래프에는 자동으로 저장되며, 직접 그래프를 만들 때는 아래처럼 지정

```python
data["Recipe"].raw_id = artifact["recipe_ids"]          # i번 Recipe 노드의 원본 레시피 ID
data["Ingredient"].ingredient_name = ingredient_names   # j번 Ingredient 노드의 재료 이름
```

```python
recipe_ids, similarities = ingredient_model.recommend_recipes(
    data, recipe_embeddings,
    ingredient_id=4,  # 고정할 재료의 노드 ID
    recipe_id=0,      # 기준이 되는 원래 레시피의 노드 ID
    top_k=5,
    recipe_model=recipe_model,  # 1단계 쌍 유사도로 순위 (생략 시 코사인 유사도)
    # contribution_path="data/ingredients_contribution.csv",
)
```

- 반환되는 `recipe_ids`는 노드 인덱스

- 기준 레시피 자신과 그래프에 없는 CSV 레시피는 제외, 재료 이름은 대소문자·앞뒤 공백을 무시하고 비교

- 그래프는 변경하지 않음

---

## 1단계: 레시피 유사도 학습 (`training/stage1.py`)

- Recipe 노드: 유사도 평가 쌍에 등장하는 레시피 2914개 (순서는 `data/stage1_rebuild_raw_gte.pt` 기준)
- 라벨: 0~4점 평가를 0~1로 변환(`점수 / 4`), 분할은 점수별 비율을 유지한 train/val/test 70/15/15 (기준 pt에 저장된 분할 그대로 사용)
- 입력 텍스트: 레시피 이름과 전체 조리 단계(`steps`) 원문으로 구성, 512토큰을 넘으면 창으로 나눠 인코딩 (11개 레시피가 2개 창)

`steps`는 기존 문자열 목록과 구조화된 JSON 목록 모두 지원

- 구조화된 단계는 동작·대상·장소·시간·온도·조건을 영어 조리 지시문으로 조합, `RAW` 문장과 `[MASK:...]` 표기는 유지
- JSON 키와 괄호, 빈 필드, 반복되는 주어 `you`는 입력에서 제외
- 변환 방식이 바뀌면 `prepare`부터 다시 실행해 임베딩 생성 후 재학습 필요

```bash
# 명령어 예시
python -m training.stage1 select --source ../RAW_recipes_normalized.csv --output data/recipes_raw_v3.csv
python -m training.stage1 prepare --recipes data/recipes_raw_v3.csv --output tmp/stage1_raw_gte.pt
python -m training.stage1 train --data tmp/stage1_raw_gte.pt --out tmp/stage1
```

- `select`: 전체 레시피 CSV에서 기준 pt의 레시피(순서 유지)와 기여도 CSV에만 있는 레시피를 골라 `id,name,steps,ingredients` CSV로 저장
    - 옵션: `--source`(required), `--reference`, `--contributions`, `--output`

- `prepare`: GTE 모델을 처음 사용할 때 다운로드, 기준 pt의 레시피 순서대로 인코딩하고 기준 pt의 train/val/test 분할을 그대로 붙여 저장. 기준 벡터와의 코사인(min/mean)을 출력하며 1에 가까우면 정상
    - 옵션: `--recipes`, `--reference`(기본 `data/stage1_rebuild_raw_gte.pt`), `--output`(기본 `tmp/stage1_raw_gte.pt`), `--batch-size`

- `train`: `RecipePairGraphSAGE`를 학습해 `--out` 폴더(비어 있거나 없는 폴더)에 저장
    - `model.pt`: 검증 목표값이 가장 낮은 epoch의 체크포인트 (`load_recipe_model()` 형식)
    - `metrics.json`: 설정과 val/test 지표 (RMSE, MAE, 정답 비율, 1점 이내 비율, 2점 이상 positive의 Precision/Recall/F1, 혼동 행렬 등)
    - `history.json`: epoch별 학습/검증 지표
    - `test_predictions.csv`: `a_id,b_id,actual_score,predicted_similarity,predicted_score`
    - 옵션: `--data`, `--out`(required), `--hidden`(64), `--dropout`(0.15), `--epochs`(100), `--folds`(5), `--patience`(10), `--lr`(0.003), `--weight-decay`(0.001), `--ordinal-weight`(0.2), `--temperature`(0.1), `--no-raw-cosine-feature`, `--seed`(42)

학습 방식

- 손실: 0~1 유사도 RMSE + `0.2 × 순서형 손실`. 순서형 손실은 네 경계(0.2/0.4/0.6/0.8)마다 "점수 ≥ k" 여부를 temperature 0.1의 sigmoid로 분류하는 BCE
- 최적화: AdamW, 매 epoch 검증 목표값(RMSE + 0.2 × 순서형 손실)이 가장 낮은 모델만 저장, `--patience` epoch 동안 개선이 없으면 조기 종료

- `similarity >= 1`인 **학습 쌍**만 `SIMILAR` 메시지 전달용 연결로 사용, 양방향 엣지의 가중치는 `similarity / 4`

- 1단계 GraphSAGE 집계는 이웃 메시지의 가중합을 **이웃 가중치의 합**으로 나눈 가중 평균 (2단계 `WeightedSAGEConv`는 가중합을 이웃 수로 나눔)

- 0점 쌍도 낮은 유사도를 학습하는 정답으로 사용

- 학습 중에는 5개 폴드를 번갈아 예측하며, 현재 예측하는 폴드의 연결은 그래프에서 제외. 검증·테스트 쌍은 연결로 사용하지 않음

최종 모델 (`checkpoints/stage1_rebuild_raw_gte_h64.pt`): test RMSE 0.183, 정답 비율 56.1%, positive F1 0.930

- 인코더(GTE·BGE 등), 입력 텍스트 형식, hidden 크기를 비교한 실험 기록은 `STAGE1_MATRIX.md`, `STAGE1_HIDDEN_SWEEP.md` 참고
- 실험에 쓰인 이전 1단계 코드(`RecipeEncoder` 등)는 `training/stage1_backup.py`에 보관 (실험 스크립트·테스트용)

---

## 2단계: 레시피–재료 기여도

`IngredientLinkEncoder`: 1단계 레시피 임베딩을 고정하고 `Recipe —CONTRIBUTION— Ingredient` 관계의 0~1 기여도 예측

```bash
python -m training.stage2 prepare
python -m training.stage2 train
python -m training.evaluate
```

- `prepare`: 카테고리·기여도 CSV에서 재료 노드를 만들고 GTE로 인코딩, 기여도 연결을 train/test로 나누어 `tmp/stage2_embeddings.pt`에 저장
    - 옵션: `--categories`, `--contributions`, `--output`, `--batch-size`, `--test-ratio`(기본 0.15), `--seed`

- `train`: 1단계 체크포인트(`load_recipe_model()`)로 레시피 임베딩을 고정한 뒤 기여도를 학습해 `tmp/stage2.pt`에 저장. hidden 크기는 1단계 모델과 같게 맞춤. 매 epoch 학습 연결을 `--folds`개로 나누어 번갈아 예측하며, 예측하는 폴드의 연결은 입력 그래프에서 제외
    - 옵션: `--stage1-data`, `--data`, `--stage1-checkpoint`, `--output`, `--epochs`, `--folds`(기본 5), `--lr`, `--seed`

- `training.evaluate`: 학습 분할 연결만 있는 그래프에서 테스트 분할 기여도를 예측해 MAE·RMSE를 출력하고, 테스트 쌍 일부(`--show`, 기본 20)를 원본 레시피 ID·재료 이름과 함께 출력
    - 옵션: `--stage1-data`, `--data`, `--stage1-checkpoint`, `--stage2-checkpoint`, `--show`

---

## 대체 재료 기반 레시피 추천 (콘솔)

1·2단계 임베딩과 체크포인트를 모두 만든 뒤 실행

```bash
python -m src.main
```

1. 레시피 ID 입력
2. 기여도 CSV에 있는 해당 레시피의 재료(기여도 높은 순) 중 제외할 재료를 번호나 이름으로 선택
3. 같은 카테고리에서 기여도가 비슷한 대체 재료 후보(`recommend_substitutes()`) 중 하나 선택
4. 대체 재료가 들어가면서 원래 레시피와 유사한 레시피 목록(`recommend_recipes()`, 1단계 쌍 유사도 순) 출력

- 빈 입력(Enter)은 처음으로, `q`·`quit`·`exit`는 종료
- 추천용 그래프에는 기여도 CSV의 train·test 연결을 모두 포함 (레시피에 이미 들어 있는 재료가 대체 후보로 나오지 않도록)
- 카테고리 CSV에 없는 재료는 대체 재료를 찾을 수 없다는 안내를 출력
- 필요한 임베딩·체크포인트가 없으면 생성 명령을 안내하고 종료
- 1단계 체크포인트가 다른 1단계 데이터로 학습됐거나 1·2단계 hidden 크기가 다르면 안내 후 종료
- 파이프(예: PowerShell)로 넣은 입력의 BOM은 자동으로 제거
- 옵션: `--stage1-data`, `--data`, `--stage1-checkpoint`, `--stage2-checkpoint`, `--recipes`, `--contributions`, `--top-k`(기본 10)

---

## Colab

`colab_pipeline.ipynb`: `src/`, `training/`의 코드를 한 노트북에 모은 버전. `data/` 폴더를 `/content`에 올리고 위에서부터 순서대로 실행 (1단계 prepare → train → evaluate → ranking-metrics, 2단계 prepare → train → evaluate, 대화형 추천)

- 노트북의 1단계 학습은 GPU 연산 순서 차이로 로컬 실행 결과와 조금 다를 수 있음 (결정론적 연산 설정 미적용)

---

## 테스트

```bash
python -m unittest discover -s tests
```
