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
python -m training.stage2 prepare
python -m training.stage2 train
python -m training.evaluate
python -m src.main
python -m unittest discover -s tests
```

- CUDA GPU를 사용할 수 있으면 BGE 인코딩·학습·추론을 GPU에서, 없으면 CPU에서 실행 (`src.engine.DEVICE`)
- 임베딩 pt와 체크포인트는 CPU 텐서로 저장하므로 GPU가 없는 환경에서도 불러올 수 있음

임베딩 저장 경로 (`prepare` 결과)

- 1단계: `artifacts/stage1_embeddings_v3.pt`
- 2단계: `artifacts/stage2_embeddings.pt`

모델 저장 경로

- 1단계: `checkpoints/stage1.pt`
- 2단계: `checkpoints/stage2.pt`

사용 가능한 옵션

- `--epochs`
- `--lr`
- `--output`(체크포인트 저장할 경로 지정)
- `--stage1-checkpoint`(2단계 입력으로 들어갈 체크포인트 경로 지정)

### 데이터 형식

모든 입력은 원본 데이터 CSV 및 임베딩을 담은 pt 파일 사용

| 단계  | 파일                                         | 내용                                                                                    |
| ----- | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1단계 | `data/stage1/v3/recipes_masked_v3.csv`       | 모든 원본 레시피 고유 정보 (`id`, `name`, `steps` 등)                                   |
| 1단계 | `data/stage1/v3/labeled_pairs_masked_v3.csv` | 레시피 쌍 유사도 정답 (`a_id`, `b_id`, `similarity` 0~4점)                              |
| 1단계 | `artifacts/stage1_embeddings_v3.pt`          | `stage1 prepare` 결과: 레시피 ID 순서(`recipe_ids`), BGE 임베딩, train/val/test 쌍 분할 |
| 2단계 | `data/stage2/ingredients_category.csv`       | 재료별 카테고리 (`category`, `ingredient`)                                              |
| 2단계 | `data/stage2/ingredients_contribution.csv`   | 레시피–재료 기여도 정답 (`ingredient`, `recipe_id`, `contribution` 0~1)                 |
| 2단계 | `artifacts/stage2_embeddings.pt`             | `stage2 prepare` 결과: 재료 이름 순서, 재료 BGE 임베딩, 기여도 train/test 분할          |

2단계 그래프(`training.data.load_dataset`) 구성

- Recipe 노드: 1단계 pt의 `recipe_ids` 순서. `SIMILAR`는 1단계 학습 분할에서 1점 이상인 쌍을 `similarity / 4` 가중치로 양방향 연결
- Ingredient 노드: 카테고리 CSV와 기여도 CSV에 등장하는 재료 이름의 합집합(소문자·앞뒤 공백 정규화, 알파벳순). 카테고리 CSV에 없는 재료(예: `salt and pepper`)는 카테고리 없이 노드로 포함
- 재료 특징: `Ingredient: {이름}. Category: {카테고리}.` 텍스트의 BGE 임베딩 (카테고리의 `_`는 공백으로 바꾸고, 카테고리가 없는 재료는 `Ingredient: {이름}.`만 사용)
- `CONTRIBUTION`: 기여도 CSV의 학습 분할(기본 85%)만 메시지 전달 그래프에 포함하고, 테스트 분할(15%)은 평가에만 사용
- 노드별 원본 표기는 `data["Recipe"].raw_id`, `data["Ingredient"].ingredient_name`에 저장

#### 학습 데이터 제작 시 유의 사항

- 재료 특징(이름, 카테고리 등)과 레시피–재료 쌍별 기여도 정답 라벨(<-정빈님 마스킹 결과에 포함됨) 필요.

- 연결되지 않은 쌍의 예상 연결 점수를 계산하기 위한 목적이므로 연결이 없다고 해서 무조건적으로 negative로 라벨링하지 말 것

- 학습할 때 연결을 비운 입력도 함께 사용하여 직접 연결되지 않은 Ingredient도 자신의 feature를 통해 임베딩 생성 가능

- 테스트 정답 연결은 메시지 전달 그래프에 넣지 않는다 (`load_dataset`에서 검사). 학습 중에는 1단계와 같이 폴드를 번갈아 예측하며, 현재 예측하는 폴드의 연결은 그래프에서 제외

### 추론

1. 1단계(SIMILAR)와 2단계(CONTRIBUTION) 체크포인트로 모델 복원

2. `IngredientLinkEncoder.edge_weights(data, recipe_embeddings, pairs)` 호출
    - `pairs`:

        `[2, 후보 수]` 모양의 `torch.long` 텐서. 각 후보 연결의 예상 기여도(0~1)를 반환

3. 대체 재료는 `recommend_substitutes()`, 같은 재료가 들어가는 유사 레시피는 `recommend_recipes()`로 찾는다. 두 함수 모두 그래프를 변경하지 않음

- 예측한 연결 점수는 다음 계산의 이웃 집계에서 제외해 자기 자신을 강화하는 문제 방지

- 학습, 추론 시 전체 그래프를 메모리에 올려 실행

#### 기여도가 비슷한 대체 재료 찾기

`recommend_substitutes()`: `data/stage2/ingredients_category.csv`에서 원래 재료와 같은 카테고리인 재료만 후보로 하고, 원래 재료의 기여도 `cont_a`와 후보의 예측 기여도 `cont_b` 사이의 `abs(cont_b - cont_a)`가 작은 순으로 `(ingredient_ids, weights)` 반환 (여기서 반환되는 `weights`는 `cont_b`와 같음)

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

`recommend_recipes()`: `data/stage2/ingredients_contribution.csv`에서 `ingredient_id`번 재료가 실제로 들어가는 레시피를 모두 찾고, 그중 `recipe_id`번 레시피와 1단계 임베딩의 코사인 유사도가 높은 순으로 최대 `top_k`개를 `(recipe_ids, similarities)`로 반환. 모델 예측 연결은 사용하지 않음

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
    # contribution_path="data/stage2/ingredients_contribution.csv",
)
```

- 반환되는 `recipe_ids`는 노드 인덱스

- 기준 레시피 자신과 그래프에 없는 CSV 레시피는 제외, 재료 이름은 대소문자·앞뒤 공백을 무시하고 비교

- 그래프는 변경하지 않음

---

## 1단계: CSV 파일로 레시피 유사도 학습

- Recipe 노드: `data/stage1/v3/recipes_masked_v3.csv`의 전체 고유 레시피 (레시피 ID가 중복되면 첫 번째 행 사용)
- 라벨: `data/stage1/v3/labeled_pairs_masked_v3.csv`의 0~4점 평가를 0~1로 변환
- BGE 입력 텍스트: 레시피 이름과 전체 조리 단계(`steps`)로 구성

`steps`는 기존 문자열 목록과 구조화된 JSON 목록 모두 지원

- 구조화된 단계는 동작·대상·장소·시간·온도·조건을 영어 조리 지시문으로 조합, `RAW` 문장과 `[MASK:...]` 표기는 유지
- JSON 키와 괄호, 빈 필드, 반복되는 주어 `you`는 입력에서 제외
- 변환 방식이 바뀌면 `prepare`부터 다시 실행해 임베딩 생성 후 재학습 필요
- 긴 입력은 BGE의 최대 입력 길이에서 잘릴 수 있음 (단계별 분할 인코딩은 적용 X)

```bash
# 명령어 예시
pip install -r requirements.min.txt
python -m training.stage1 prepare
python -m training.stage1 train
python -m training.stage1 evaluate
python -m training.stage1 predict --a-id 172520 --b-id 306863
python -m training.stage1 top-similar --recipe-id 172520 --top-k 10
```

- `prepare`: BGE 모델을 처음 사용할 때 다운로드, 인코딩 결과와 점수별 비율을 유지하면서 70/15/15로 나눈 train/val/test 쌍을 `artifacts/stage1_embeddings_v3.pt`에 저장
    - 옵션: `--recipes`, `--pairs`, `--output`, `--seed`, `--batch-size`

- `train`: 매 epoch 검증 쌍의 0~1 유사도 RMSE를 계산하고 최저 검증 RMSE의 모델만 `checkpoints/stage1.pt`에 저장. 검증 RMSE가 연속 `--patience` epoch 동안 개선되지 않으면 조기 종료
    - 옵션: `--data`, `--output`, `--hidden`, `--epochs`, `--folds`, `--patience`(기본 10), `--lr`, `--seed`

- `evaluate`: 생성한 모델을 테스트 쌍으로 평가, 유사도 2점 이상을 positive로 간주하여 정확도 계산. 테스트 쌍의 `a_id,b_id,similarity,predicted_similarity,predicted_similarity_0_1`(정답·예측 0~4점, 예측 0~1 유사도)를 `data/stage1_predictions.csv`에 저장 (`--output`으로 경로 변경 가능)
    - 옵션: `--data`, `--checkpoint`, `--output`

- `predict`: 입력 그래프에 직접 연결되지 않은 두 레시피도 0~1 유사도로 평가
    - 옵션: `--data`, `--checkpoint`, `--a-id`(required), `--b-id`(required)

- `top-similar`: 지정한 레시피(`--recipe-id`)를 제외하고 전체 레시피 중 유사도가 높은 상위 `--top-k`개를 원본 ID, 이름, 0~1 예측 유사도와 함께 출력. 이름은 `--recipes` CSV에서 읽어옴
    - 옵션: `--data`, `--checkpoint`, `--recipes`, `--recipe-id`(required), `--top-k`(기본 10)

- CSV 1단계 학습 손실과 학습 로그의 오차는 정규화된 0~1 유사도 기준의 RMSE로 계산, `evaluate`의 테스트 오차도 같은 기준의 RMSE로 출력

- `similarity >= 1`인 **학습 쌍**만 `SIMILAR` 메시지 전달용 연결로 사용, 양방향 엣지의 가중치는 `similarity / 4`

- 두 그래프 집계층 모두 이웃 메시지의 가중합을 이웃 수로 나누어 집계하므로 이웃이 하나여도 유사도에 따라 메시지 크기가 달라짐

- 0점 쌍도 낮은 유사도를 학습하는 정답으로 사용

- 학습 중에는 5개 폴드를 번갈아 예측하며, 현재 예측하는 폴드의 연결은 그래프에서 제외. 검증·테스트 쌍은 연결로 사용하지 않음

---

## 2단계: 레시피–재료 기여도

`IngredientLinkEncoder`: 1단계 레시피 임베딩을 고정하고 `Recipe —CONTRIBUTION— Ingredient` 관계의 0~1 기여도 예측

```bash
python -m training.stage2 prepare
python -m training.stage2 train
python -m training.evaluate
```

- `prepare`: 카테고리·기여도 CSV에서 재료 노드를 만들고 BGE로 인코딩, 기여도 연결을 train/test로 나누어 `artifacts/stage2_embeddings.pt`에 저장
    - 옵션: `--categories`, `--contributions`, `--output`, `--batch-size`, `--test-ratio`(기본 0.15), `--seed`

- `train`: 1단계 체크포인트로 레시피 임베딩을 고정한 뒤 기여도를 학습해 `checkpoints/stage2.pt`에 저장. 매 epoch 학습 연결을 `--folds`개로 나누어 번갈아 예측하며, 예측하는 폴드의 연결은 입력 그래프에서 제외
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
4. 대체 재료가 들어가면서 원래 레시피와 유사한 레시피 목록(`recommend_recipes()`) 출력

- 빈 입력(Enter)은 처음으로, `q`·`quit`·`exit`는 종료
- 추천용 그래프에는 기여도 CSV의 train·test 연결을 모두 포함 (레시피에 이미 들어 있는 재료가 대체 후보로 나오지 않도록)
- 카테고리 CSV에 없는 재료는 대체 재료를 찾을 수 없다는 안내를 출력
- 필요한 임베딩·체크포인트가 없으면 생성 명령을 안내하고 종료
- 옵션: `--stage1-data`, `--data`, `--stage1-checkpoint`, `--stage2-checkpoint`, `--recipes`, `--contributions`, `--top-k`(기본 10)

---

## 테스트

```bash
python -m unittest discover -s tests
```
