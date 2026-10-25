# recipe-recommendation

소프트웨어응용 추천 시스템 설계 프로젝트

## 2단계 GraphSAGE 추천 엔진

#### 텍스트 입력 인코딩

`BGETextEncoder`: Recipe와 Ingredient 텍스트에 적용해 768차원 정규화 임베딩 생성

#### 1단계

`RecipeEncoder`: `Recipe —SIMILAR— Recipe` 연결만 사용해 레시피 임베딩과 레시피 쌍의 유사도 점수 학습

#### 2단계

`IngredientLinkEncoder`: 1단계 임베딩을 고정한 상태에서 실제로 연결된 `Recipe —MAIN/SUPPORT— Ingredient` 연결을 역할별(MAIN/SUPPORT) GraphSAGE로 집계하여 Recipe–Ingredient 후보들의 MAIN, SUPPORT 연결 확률(=edge 가중치)을 각각 학습

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
python -m training.train --stage 1 --data examples/toy_graph.json
python -m training.train --stage 2 --data examples/toy_graph.json
python -m training.evaluate --data examples/toy_graph.json
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

`recipe_features`, `ingredient_features`: 순서가 노드 ID인 2차원 숫자 배열 `similar_edges`: `[recipe_a, recipe_b, weight]` (`weight`는 0(전혀 유사하지 않음)~1(완전히 유사함) 사이의 실수)
`main_edges`, `support_edges`: `[recipe_id, ingredient_id]`

- 각 edge는 두 방향 중 하나만 기록하면 실제 그래프에서는 양방향으로 연결 생성

- MAIN과 SUPPORT는 독립된 관계로, 한 Recipe–Ingredient 쌍이 두 역할을 모두 가질 수 있음

- `MAIN` 과 `SUPPORT` 연결 동시에 저장(예시: 정답이 MAIN이면 `MAIN, 1`과 `SUPPORT, 0`으로 각각 기록), 연결 점수는 역할별로 독립 학습

`toy_graph.json`에서 `stage1_train/test` 항목은 `[recipe_a, recipe_b, label]`, `stage2_train/test`항목은 `[recipe_id, ingredient_id, "MAIN" 또는 "SUPPORT", label]` 형식

- 1단계 유사도 라벨: 0~1 사이 실수 (예: 0.75)
- 2단계 연결 라벨: 1(유효한 연결) 또는 0(검증된 부적합 연결, 필요시 추가 예정)

#### 학습 데이터 제작 시 유의 사항

- MAIN/SUPPORT의 실제 역할 유사도를 학습하기 위해 재료 특징(이름, 카테고리 등)과 역할별 정답 라벨 필요.

- 연결되지 않은 쌍의 예상 연결 점수를 계산하기 위한 목적이므로 연결이 없다고 해서 무조건적으로 negative로 라벨링하지 말 것

- 학습할 때 연결을 비운 입력도 함께 사용하여 직접 연결되지 않은 Ingredient도 자신의 feature를 통해 임베딩 생성 가능

- 학습 및 평가 대상으로 사용할 정답 라벨은 메시지 전달용 edge 목록에서 제외할 것 (+ data.py의 로더에서 검사함)

### 추론

1. SIMILAR, MAIN/SUPPORT 체크포인트 2개로 모델 복원

2. `IngredientLinkEncoder.edge_weights(data, recipe_embeddings, pairs, role)` 호출
    - `pairs`:

        `[2, 후보 수]` 모양의 `torch.long` 텐서. 각 후보 연결의 0~1 가중치를 반환하며 MAIN과 SUPPORT를 각각 호출해 비교 가능

3. `IngredientLinkEncoder.recommend_ingredients(data, recipe_embeddings, recipe_id, role, top_k, save_predicted_edges=False)` 호출

    `role`에 해당하는 edge로 이미 연결된 재료를 제외하고 `(ingredient_ids, weights)`를 가중치 점수가 높은 순으로 반환
    - `save_predicted_edges`:

        `True`: 반환할 추천 연결을 점수 계산 후 `data`에 추가, edge 속성 `edge_weight`에 가중치 기록, `is_predicted`에 예측 여부 기록

        `False` (기본값): 그래프를 변경하지 않음

- 예측한 연결 점수는 다음 계산의 이웃 집계에서 제외해 자기 자신을 강화하는 문제 방지

- 학습, 추론 시 전체 그래프를 메모리에 올려 실행

---

## CSV 파일로 1단계 레시피 유사도 학습

`data/recipes_masked.csv`의 전체 고유 레시피를 노드로 만들고, `data/labeled_pairs.csv`의 0~4점 평가를 0~1 라벨로 변환한다. 레시피 ID가 중복되면 첫 번째 행을 사용한다. BGE 입력 텍스트는 레시피 이름과 전체 조리 단계(`steps`)로 구성한다.

```bash
pip install -r requirements.min.txt
python -m training.stage1_csv prepare
python -m training.stage1_csv train
python -m training.stage1_csv evaluate
python -m training.stage1_csv predict --a-id 172520 --b-id 306863
```

- `prepare`: BGE 모델을 처음 사용할 때 다운로드, 인코딩 결과와 점수별 비율을 유지하면서 나눈 train/val/test 쌍을 `data/stage1.pt`에 저장
- `train`: 학습 모델 생성 (결과 파일: `checkpoints/stage1_csv.pt`)
- `evaluate`: 생성한 모델 검증/테스트, 유사도 2점 이상을 positive로 간주하여 정확도 계산
- `predict`: 입력 그래프에 직접 연결되지 않은 두 레시피도 0~1 유사도로 평가

`similarity >= 1`인 **학습 쌍**만 `SIMILAR` 메시지 전달용 연결로 사용한다. 0점 쌍도 낮은 유사도를 학습하는 정답으로 사용한다.

학습 중에는 5개 폴드를 번갈아 예측하며, 현재 예측하는 폴드의 연결을 그래프에서 제외한다. 검증·테스트 쌍은 연결로 사용하지 않는다.
