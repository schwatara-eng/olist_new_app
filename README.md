# 올리스트 배송 분석 대시보드

브라질 전자상거래 Olist의 주문·상품·리뷰 데이터를 분석한 Streamlit 앱입니다. 코랩에서 만든 요약표와 모델을 웹 화면에 연결하고, 분석 근거를 찾아 답하는 데이터 질문 기능을 추가했습니다.

## 화면 구성

| 탭 | 기능 |
| --- | --- |
| 주문·결제 | 주문 수, 결제 금액, 월별 추이 |
| 배송·리뷰 | 배송 상태별 평균 별점과 저평점 비율, 리뷰 감성 분석 |
| 상품·지역 | 상품과 지역별 주문 분포, 유사 상품 추천, 판매자 유형과 이탈 점검 기준 비교 |
| 데이터 질문 | 배송 상태별 평균 별점과 저평점 비율에 대한 답변, 집계 기준, 출처 |

## Windows에서 실행하기

VS Code에서 프로젝트 폴더를 열고, PowerShell 터미널에서 아래 명령을 실행합니다. 학습 환경은 Windows 11과 Python 3.14.7입니다. 저장된 모델을 읽을 때 버전 오류가 나면 모델을 만든 환경과 같은 scikit-learn 버전을 사용합니다.

```powershell
python -m pip install streamlit pandas altair joblib scikit-learn openai
```

데이터 질문 탭에서 AI 답변을 사용하려면 **앱을 실행할 PowerShell**에서 API 키를 입력합니다. 키는 입력할 때 화면에 표시되지 않으며, 현재 터미널 세션에서 사용됩니다.

```powershell
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "OpenAI API 키" -AsSecureString)).Password
python -m streamlit run app.py
```

터미널에 표시되는 **Local URL**을 브라우저에서 엽니다. API 키는 소스 코드나 GitHub 커밋에 포함하지 않습니다.

## 데이터와 모델 파일

[Olist 공개 데이터셋](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)의 원본 CSV는 프로젝트의 `data/` 폴더에 둡니다. 코랩에서 만든 아래 결과 파일은 `app.py`와 같은 폴더에 둡니다.

| 파일 | 용도 |
| --- | --- |
| `olist_delivery_summary.csv` | 배송 상태별 평균 별점 |
| `olist_low_rating_summary.csv` | 배송 상태별 저평점 비율 |
| `olist_seller_cluster_summary.csv` | 판매자 유형 요약 |
| `olist_churn_threshold_summary.csv` | 이탈 점수 기준별 시험 결과 |
| `olist_facts.csv` | 데이터 질문에 쓰는 사실표 |
| `olist_recommendation.pkl` | 유사 상품 추천 |
| `olist_sentiment.pkl` | 리뷰 감성 분석 |

## 데이터 질문 사용 예

**데이터 질문** 탭에 “지연 주문 중 별점 2점 이하는 얼마나 돼?”를 입력하고 **근거 찾기**를 누릅니다. 실습 데이터에서는 **62.4%**가 나오며, 집계 기준과 출처 `olist_low_rating_summary.csv (F02)`가 함께 표시됩니다.

이 비율은 배송 완료 주문 중 실제 도착일·예정일·별점이 모두 기록된 주문을 기준으로 계산했습니다. 예정일 다음 날부터 도착한 주문을 지연으로 분류했습니다.

현재 질문 기능은 키워드로 사실표에서 평균 별점 또는 저평점 비율에 해당하는 행을 찾고, 그 근거를 바탕으로 답변을 생성합니다. 근거를 찾지 못하면 안내를 표시합니다. 판매자 이탈 화면은 코랩에서 계산한 시험 결과를 비교하는 기능입니다.
