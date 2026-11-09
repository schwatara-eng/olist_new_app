<div align="center">

<sub>SLIDE 01 / 10</sub>

# 📦 올리스트 배송 분석 대시보드

브라질 전자상거래 **Olist** 주문·상품·리뷰 데이터 분석 · Streamlit 웹앱

🚀 **배포된 대시보드:** [Olist 배송 분석 대시보드](https://olistnewappgit-ib4dhdezq43pseivtvm6up.streamlit.app/)

`Python` · `Pandas` · `scikit-learn` · `Streamlit` · `Altair` · `OpenAI API`

</div>

<br>

---

<sub>SLIDE 02 / 10</sub>

## 🔎 질문 하나에서 출발했다

> **배송이 늦어지면, 고객은 별점을 얼마나 깎을까?**

코랩에서 주문·배송·리뷰 데이터를 정제하고 집계한 뒤,
그 결과표와 모델을 **웹 화면 하나**에 모았다.

```
수집 → 정제 → 분석 → 모델 → 웹 화면 → 근거 기반 질문
 Kaggle   Colab    Colab   .pkl   Streamlit    사실표 + LLM
```

<br>

---

<sub>SLIDE 03 / 10</sub>

## 📊 핵심 발견 — 지연은 별점을 반으로 깎는다

<div align="center">

| | 주문 수 | 평균 별점 | 별점 2점 이하 |
| :--- | ---: | ---: | ---: |
| ✅ 제때 도착 | 89,443 | **4.29점** | 9.2% |
| ⚠️ 지연 도착 | 6,381 | **2.27점** | **62.4%** |

</div>

- 평균 별점 차이 **2.02점**
- 지연 주문 **10건 중 6건**이 저평점

<sub>기준: 배송 완료 주문 중 실제 도착일·예정일·별점이 모두 기록된 주문. 예정일 다음 날부터 도착하면 ‘지연’.</sub>

<br>

---

<sub>SLIDE 04 / 10</sub>

## 🗂️ 화면 구성 — 탭 5개

| 탭 | 무엇을 보여주나 |
| --- | --- |
| 🧾 **주문·결제** | 주문 수, 결제 금액, 결제수단별 금액, 월별 주문 추이 |
| 🚚 **배송·리뷰** | 배송 상태별 평균 별점·저평점 비율, 리뷰 감성 분석 |
| 🗺️ **상품·고객 지역** | 카테고리 상위 10, 고객 지역별 베스트셀러 상품·주문·지도, 유사 상품 추천 |
| 🏪 **판매자** | 판매자 지역·유형, 90일 이탈 위험 예측과 시간순 평가 |
| 💬 **데이터 질문** | 임베딩으로 사실표 근거를 찾아 답변 + 집계 기준 + 출처 표시 |

<br>

---

<sub>SLIDE 05 / 10</sub>

## 🤖 모델 두 개를 화면에 연결

**리뷰 감성 분석** — `olist_sentiment.pkl`
리뷰 문장을 넣으면 긍정·부정을 판정

**유사 상품 추천** — `olist_recommendation.pkl`
상품을 고르면 비슷한 상품을 찾아 줌

<br>

---

<sub>SLIDE 06 / 10</sub>

## 🏪 판매자 유형 & 이탈 점검

**판매자 유형 (군집 3개)**

| 그룹 | 판매자 수 | 판매 상품 수 (중앙값) | 상품 판매액 (중앙값, R$) |
| :---: | ---: | ---: | ---: |
| 1 | 710 | 69.5개 | 8,078.49 |
| 2 | 1,210 | 10개 | 1,197.50 |
| 0 | 1,175 | 2개 | 149.70 |

<sub>판매액은 배송비를 제외한 상품 금액.</sub>

**이탈 점수 기준별 시험 결과** <sub>(코랩 시험 데이터, 실제 이탈 판매자 108명 기준)</sub>

| 기준 | 점검 명단 | 찾아낸 이탈 | 잘못 지목 | 놓친 이탈 |
| :---: | ---: | ---: | ---: | ---: |
| 0.50 | 263 | 87 | 176 | 21 |
| 0.60 | 157 | 64 | 93 | 44 |
| 0.70 | 0 | 0 | 0 | 108 |

> 기준을 높이면 헛짚는 경우는 줄지만, 놓치는 이탈이 늘어난다.

<br>

---

<sub>SLIDE 07 / 10</sub>

## 💬 데이터 질문 — 근거부터 찾고 답한다

```
질문 입력  →  임베딩으로 사실 검색  →  상위 3개 근거 선택  →  GPT가 근거만으로 답변
```

**예시**

> 🙋 지연 주문 중 별점 2점 이하는 얼마나 돼?
>
> 🤖 **62.4%** · 기준 표시 · 출처 `olist_low_rating_summary.csv (F02)`

- 근거를 못 찾으면 지어내지 않고 **안내 문구**를 보여 줌
- 사실표 범위: 배송·리뷰, 지역, 상품 카테고리, 결제 관련 사실 **F01–F08**

<br>

---

<sub>SLIDE 08 / 10</sub>

## ▶️ 실행하기 (Windows)

**1. 패키지 설치** — VS Code에서 폴더 열고 PowerShell에서

```powershell
python -m pip install streamlit pandas altair joblib scikit-learn openai
```

**2. API 키 입력 후 실행** — 키는 화면에 안 보이고, 현재 터미널에서만 쓰임

```powershell
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "OpenAI API 키" -AsSecureString)).Password
python -m streamlit run app.py
```

**3.** 터미널에 뜬 **Local URL**을 브라우저에서 열기

> 🔐 API 키는 소스 코드나 GitHub 커밋에 넣지 않는다.
> ⚙️ 학습 환경: Windows 11 · Python 3.14.7 — 모델 로딩 중 버전 오류가 나면 모델을 만든 환경과 같은 scikit-learn 버전을 쓴다.

<br>

---

<sub>SLIDE 09 / 10</sub>

## 📁 파일 구조

```
olist_new_app/
├─ app.py                              Streamlit 앱
├─ data/                               Olist 원본 CSV
├─ olist_delivery_summary.csv          배송 상태별 평균 별점
├─ olist_low_rating_summary.csv        배송 상태별 저평점 비율
├─ olist_seller_cluster_summary.csv    판매자 유형 요약
├─ olist_churn_threshold_summary.csv   이탈 점수 기준별 시험 결과
├─ olist_facts.csv                     임베딩 검색용 사실표
├─ olist_churn.py                      판매자 이탈 예측·평가
├─ requirements.txt                    앱 실행 패키지
├─ olist_recommendation.pkl            유사 상품 추천 모델
└─ olist_sentiment.pkl                 리뷰 감성 분석 모델
```

데이터 출처: [Olist 공개 데이터셋 (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

<br>

---

<div align="center">

<sub>SLIDE 10 / 10</sub>

## 🧭 다음 단계

</div>

- [x] 사실표 확장 — 지역·카테고리·결제 관련 사실 추가 (F03–F08)
- [x] 키워드 검색을 **임베딩 검색(RAG)** 으로 교체 — 사실표 상위 근거 3건과 출처 표시
- [x] 판매자 이탈 점검을 **시간순 학습·평가 화면**으로 발전 — 이후 90일 주문 중단 기준
- [ ] 집계·답변 기능을 **FastAPI**로 분리

<div align="center">

<br>

**배송 약속을 지키는 것 = 별점을 지키는 것**

</div>
