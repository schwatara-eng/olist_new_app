from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from openai import OpenAI, OpenAIError

st.set_page_config(page_title="Olist 배송 분석")

st.title("올리스트 배송 분석 대시보드")
order_tab, delivery_tab, product_tab, seller_tab, question_tab = st.tabs(
    ["주문·결제", "배송·리뷰", "상품·고객 지역", "판매자", "데이터 질문"]
)

def get_openai_client() -> OpenAI:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    return OpenAI(api_key=api_key) if api_key else OpenAI()


@st.cache_data(show_spinner=False)
def embed_fact_documents(documents: tuple[str, ...]) -> list[list[float]]:
    response = get_openai_client().embeddings.create(
        model="text-embedding-3-small",
        input=list(documents),
    )
    return [item.embedding for item in response.data]


@st.cache_data(show_spinner=False)
def get_seller_churn_report(data_dir: str):
    from olist_churn import build_churn_report

    return build_churn_report(data_dir)


with order_tab:

    st.subheader("주문·결제 요약")

    data_dir = Path(__file__).resolve().parent / "data"
    orders_path = data_dir / "olist_orders_dataset.csv"
    payments_path = data_dir / "olist_order_payments_dataset.csv"

    if orders_path.is_file() and payments_path.is_file():
        orders = pd.read_csv(
            orders_path,
            usecols=["order_id", "order_purchase_timestamp"],
        )
        payments = pd.read_csv(
            payments_path,
            usecols=["payment_type", "payment_value"],
        )

        order_count = orders["order_id"].nunique()
        payment_total = payments["payment_value"].sum()

        order_column, payment_column = st.columns(2)

        with order_column:
            st.metric("전체 주문 수", f"{order_count:,}건")

        with payment_column:
            st.metric("기록된 결제 금액 합계", f"R$ {payment_total:,.2f}")

        st.caption(
            "원본 주문·결제 CSV 전체 기준. "
            "배송 완료 주문만 골라 계산한 금액과 집계 범위가 달라."
        )

        st.subheader("결제수단별 결제 금액")

        payment_labels = {
            "credit_card": "신용카드",
            "boleto": "볼레투",
            "voucher": "바우처",
            "debit_card": "직불카드",
            "not_defined": "미분류",
        }

        payment_summary = (
            payments.groupby("payment_type", dropna=False, as_index=False)
            .agg(total_amount=("payment_value", "sum"))
        )

        payment_summary["method"] = (
            payment_summary["payment_type"]
            .replace(payment_labels)
            .fillna("미기록")
        )

        payment_chart = (
            alt.Chart(payment_summary)
            .mark_bar(color="#337CB0")
            .encode(
                x=alt.X("total_amount:Q", title="결제 금액 (R$)"),
                y=alt.Y("method:N", title="결제수단", sort="-x"),
                tooltip=[
                    alt.Tooltip("method:N", title="결제수단"),
                    alt.Tooltip("total_amount:Q", title="결제 금액 (R$)", format=",.2f"),
                ],
            )
        )

        st.altair_chart(payment_chart)
        st.caption("원본 결제 CSV 전체 기준. 나눠 결제한 금액은 각 결제수단에 따로 반영했어.")

        st.subheader("월별 주문 수")

        purchase_dates = pd.to_datetime(
            orders["order_purchase_timestamp"],
            errors="coerce",
        )

        monthly_orders = (
            orders.assign(month=purchase_dates.dt.to_period("M").astype("string"))
            .loc[purchase_dates.notna()]
            .groupby("month", as_index=False)
            .agg(order_count=("order_id", "nunique"))
            .sort_values("month")
        )

        monthly_chart = (
            alt.Chart(monthly_orders)
            .mark_line(point=True, color="#337CB0")
            .encode(
                x=alt.X("month:O", title="주문 연·월", sort="ascending"),
                y=alt.Y("order_count:Q", title="주문 수 (건)"),
                tooltip=[
                    alt.Tooltip("month:O", title="주문 연·월"),
                    alt.Tooltip("order_count:Q", title="주문 수", format=",d"),
                ],
            )
        )

        st.altair_chart(monthly_chart)
        st.caption(
            f"원본 주문 CSV 기준 · 주문 시각을 읽을 수 없는 행 "
            f"{purchase_dates.isna().sum():,}개 제외"
        )

    else:
        st.warning("data 폴더에 주문 CSV와 결제 CSV가 모두 있는지 확인해 줘.")

    st.divider()

with delivery_tab:

    summary_path = Path(__file__).with_name("olist_delivery_summary.csv")

    if not summary_path.is_file():
        st.error("app.py와 같은 폴더에 olist_delivery_summary.csv를 넣어 줘.")
        st.stop()

    summary = pd.read_csv(summary_path)
    group_order = ["제때 도착", "지연 도착"]

    st.caption(
        "배송이 완료되고 실제 도착일·예정일·별점이 모두 기록된 주문 기준. "
        "예정일 다음 날 이후에 도착한 주문을 지연으로 분류했어."
    )

    chart = (
        alt.Chart(summary)
        .mark_bar(color="#337CB0")
        .encode(
            x=alt.X("delivery_group:N", title="배송 상태", sort=group_order),
            y=alt.Y(
                "average_score:Q",
                title="평균 별점",
                scale=alt.Scale(domain=[0, 5]),
            ),
            tooltip=[
                alt.Tooltip("delivery_group:N", title="배송 상태"),
                alt.Tooltip("average_score:Q", title="평균 별점", format=".2f"),
                alt.Tooltip("order_count:Q", title="주문 수", format=",d"),
            ],
        )
    )

    st.altair_chart(chart)

    for group in group_order:
        matching_rows = summary.loc[summary["delivery_group"] == group]

        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            st.write(
                f"**{group}** · 평균 **{row['average_score']:.2f}점** "
                f"· 주문 {int(row['order_count']):,}건"
            )

    st.divider()

    st.subheader("별점 2점 이하 주문 비율")

    low_rating_path = Path(__file__).with_name("olist_low_rating_summary.csv")

    if low_rating_path.is_file():
        low_ratings = pd.read_csv(low_rating_path, encoding="utf-8-sig")

        low_rating_chart = (
            alt.Chart(low_ratings)
            .mark_bar(color="#337CB0")
            .encode(
                x=alt.X(
                    "delivery_group:N",
                    title="배송 상태",
                    sort=["제때 도착", "지연 도착"],
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "low_rating_pct:Q",
                    title="별점 2점 이하 주문 비율 (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("delivery_group:N", title="배송 상태"),
                    alt.Tooltip("low_rating_pct:Q", title="비율 (%)", format=".1f"),
                    alt.Tooltip("low_rating_count:Q", title="2점 이하 주문 수", format=",d"),
                    alt.Tooltip("order_count:Q", title="전체 집계 주문 수", format=",d"),
                ],
            )
        )

        st.altair_chart(low_rating_chart)
        st.caption("배송이 완료되고 실제 도착일·예정일·별점이 모두 기록된 주문 기준.")

        for row in low_ratings.itertuples(index=False):
            st.write(
                f"**{row.delivery_group}** · "
                f"{row.low_rating_count:,}건 / {row.order_count:,}건 "
                f"(**{row.low_rating_pct:.1f}%**)"
            )
    else:
        st.warning("app.py와 같은 폴더에 olist_low_rating_summary.csv를 넣어 줘.")

    st.divider()
    st.subheader("리뷰 감성 분석")
    st.caption("포르투갈어 리뷰 문장을 입력하면 긍정 또는 부정을 예측해.")

    sentiment_path = Path(__file__).with_name("olist_sentiment.pkl")

    if not sentiment_path.is_file():
        st.info("app.py와 같은 폴더에 olist_sentiment.pkl을 넣어 줘.")
    else:
        with st.form("sentiment_form"):
            review_text = st.text_area(
                "리뷰 문장",
                value="O produto não chegou.",
            )
            analyze_review = st.form_submit_button("리뷰 분석하기")

        if analyze_review:
            review_text = review_text.strip()

            if review_text:
                import joblib

                sentiment_model = joblib.load(sentiment_path)
                prediction = sentiment_model.predict([review_text])[0]

                st.write(f"**모델 예측 · {prediction}**")
            else:
                st.warning("리뷰 문장을 입력해 줘.")

with product_tab:
    st.subheader("주문 상품 수 상위 10개 카테고리")

    items = pd.read_csv(
        data_dir / "olist_order_items_dataset.csv",
        usecols=["product_id"],
    )

    products = pd.read_csv(
        data_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )

    translations = pd.read_csv(
        data_dir / "product_category_name_translation.csv",
        usecols=["product_category_name", "product_category_name_english"],
    )

    items_with_category = items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    items_with_category = items_with_category.merge(
        translations,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    items_with_category["category"] = (
        items_with_category["product_category_name_english"]
        .fillna(items_with_category["product_category_name"])
        .fillna("미분류")
    )

    category_summary = (
        items_with_category.groupby("category")
        .size()
        .reset_index(name="item_count")
        .nlargest(10, "item_count")
    )

    category_chart = (
        alt.Chart(category_summary)
        .mark_bar(color="#337CB0")
        .encode(
            x=alt.X("item_count:Q", title="주문 상품 수 (행)"),
            y=alt.Y("category:N", title="카테고리", sort="-x"),
            tooltip=[
                alt.Tooltip("category:N", title="카테고리"),
                alt.Tooltip("item_count:Q", title="주문 상품 수", format=",d"),
            ],
        )
    )

    st.altair_chart(category_chart)
    st.caption(
        "원본 주문 상품 CSV 전체 기준. 주문 상품 한 행을 1개로 셌어. "
        "영어 이름이 없으면 원래 카테고리명을 표시해."
    )

    st.divider()
    st.subheader("고객 지역별 주문 수")

    orders_for_region = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id"],
    )

    customers = pd.read_csv(
        data_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"],
    )

    orders_with_region = orders_for_region.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    orders_with_region["customer_state"] = (
        orders_with_region["customer_state"].fillna("미기록")
    )

    state_summary = (
        orders_with_region.groupby("customer_state", as_index=False)
        .agg(order_count=("order_id", "nunique"))
        .nlargest(10, "order_count")
    )

    state_chart = (
        alt.Chart(state_summary)
        .mark_bar(color="#337CB0")
        .encode(
            x=alt.X("order_count:Q", title="주문 수 (건)"),
            y=alt.Y(
                "customer_state:N",
                title="고객 지역 (주)",
                sort="-x",
                scale=alt.Scale(paddingInner=0.25, paddingOuter=0.1),
            ),
            tooltip=[
                alt.Tooltip("customer_state:N", title="고객 지역"),
                alt.Tooltip("order_count:Q", title="주문 수", format=",d"),
            ],
        )
        .properties(height=180)
    )

    st.altair_chart(state_chart)
    st.caption(
        "원본 주문 CSV 전체 기준 · 주문 수 상위 10개 지역 표시. "
        "고객표에 지역이 없으면 ‘미기록’으로 집계해."
    )

    st.divider()
    st.subheader("지역별 베스트셀러 상품")
    st.caption(
        "고객 주소의 주(state)를 기준으로 배송 완료 주문의 상품을 집계해. "
        "상품명은 데이터에 없어 상품 ID와 카테고리를 표시하고, "
        "각 주문 상품 행을 1개로 세며 상품 금액에서 배송비는 제외해."
    )

    sales_orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status"],
    )
    sales_orders = sales_orders.loc[
        sales_orders["order_status"].eq("delivered"),
        ["order_id", "customer_id"],
    ]

    sales_customers = pd.read_csv(
        data_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"],
    )
    sales_items = pd.read_csv(
        data_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "price"],
    )
    sales_products = pd.read_csv(
        data_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )
    sales_translations = pd.read_csv(
        data_dir / "product_category_name_translation.csv",
        usecols=["product_category_name", "product_category_name_english"],
    )

    regional_sales = sales_items.merge(
        sales_orders,
        on="order_id",
        how="inner",
        validate="many_to_one",
    ).merge(
        sales_customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    ).merge(
        sales_products,
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(
        sales_translations,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    regional_sales["customer_state"] = (
        regional_sales["customer_state"].fillna("미기록")
    )
    regional_sales["category"] = (
        regional_sales["product_category_name_english"]
        .fillna(regional_sales["product_category_name"])
        .fillna("미분류")
    )

    state_options = sorted(regional_sales["customer_state"].unique().tolist())
    selected_state = st.selectbox(
        "고객 지역 선택",
        state_options,
        key="regional_bestseller_state",
    )

    regional_best_sellers = (
        regional_sales.loc[regional_sales["customer_state"].eq(selected_state)]
        .groupby(["product_id", "category"], as_index=False)
        .agg(
            sold_items=("order_id", "size"),
            order_count=("order_id", "nunique"),
            product_sales=("price", "sum"),
        )
        .sort_values(
            ["sold_items", "order_count"],
            ascending=False,
        )
        .head(10)
        .reset_index(drop=True)
    )

    if regional_best_sellers.empty:
        st.info("선택한 지역에서 집계할 상품이 없어.")
    else:
        regional_best_sellers["row_order"] = range(len(regional_best_sellers))
        regional_best_sellers["product_id_short"] = (
            regional_best_sellers["product_id"].str.slice(0, 8) + "…"
        )
        row_sort = list(range(len(regional_best_sellers)))
        max_sold_items = int(regional_best_sellers["sold_items"].max())
        sales_tick_values = sorted(
            {
                0,
                round(max_sold_items * 0.25),
                round(max_sold_items * 0.5),
                round(max_sold_items * 0.75),
                max_sold_items,
            }
        )

        product_id_chart = (
            alt.Chart(regional_best_sellers)
            .mark_text(
                align="left",
                baseline="middle",
                color="#64748B",
                fontSize=11,
            )
            .encode(
                x=alt.value(0),
                y=alt.Y(
                    "row_order:O",
                    sort=row_sort,
                    axis=None,
                    scale=alt.Scale(paddingInner=0.25, paddingOuter=0.1),
                ),
                text=alt.Text("product_id_short:N"),
                tooltip=alt.Tooltip("product_id:N", title="상품 ID"),
            )
            .properties(
                width=70,
                height=180,
                title=alt.TitleParams(text="상품 ID", anchor="start", fontSize=12),
            )
        )

        category_chart = (
            alt.Chart(regional_best_sellers)
            .mark_text(
                align="left",
                baseline="middle",
                color="#64748B",
                fontSize=11,
            )
            .encode(
                x=alt.value(0),
                y=alt.Y("row_order:O", sort=row_sort, axis=None),
                text=alt.Text("category:N"),
                tooltip=alt.Tooltip("category:N", title="카테고리"),
            )
            .properties(
                width=150,
                height=180,
                title=alt.TitleParams(text="카테고리명", anchor="start", fontSize=12),
            )
        )

        bestseller_bars = (
            alt.Chart(regional_best_sellers)
            .mark_bar(color="#337CB0")
            .encode(
                x=alt.X(
                    "sold_items:Q",
                    title="판매 상품 수 (행)",
                    scale=alt.Scale(
                        domain=[0, max_sold_items],
                        nice=False,
                    ),
                    axis=alt.Axis(values=sales_tick_values),
                ),
                y=alt.Y(
                    "row_order:O",
                    sort=row_sort,
                    axis=None,
                    scale=alt.Scale(paddingInner=0.25, paddingOuter=0.1),
                ),
                tooltip=[
                    alt.Tooltip("product_id:N", title="상품 ID"),
                    alt.Tooltip("category:N", title="카테고리"),
                    alt.Tooltip("sold_items:Q", title="판매 상품 수", format=",d"),
                    alt.Tooltip("order_count:Q", title="주문 수", format=",d"),
                    alt.Tooltip(
                        "product_sales:Q",
                        title="상품 금액 (R$)",
                        format=",.2f",
                    ),
                ],
            )
            .properties(width=400, height=180)
        )

        bestseller_chart = alt.hconcat(
            product_id_chart,
            category_chart,
            bestseller_bars,
            spacing=8,
        ).resolve_scale(y="shared")

        st.altair_chart(bestseller_chart, use_container_width=False)
        st.dataframe(
            regional_best_sellers[
                ["product_id", "category", "sold_items", "order_count", "product_sales"]
            ].rename(
                columns={
                    "product_id": "상품 ID",
                    "category": "카테고리",
                    "sold_items": "판매 상품 수",
                    "order_count": "주문 수",
                    "product_sales": "상품 금액 합계 (R$)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.divider()
    st.subheader("고객 주문 분포 지도")

    map_customers = pd.read_csv(
        data_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix"],
    )

    locations = pd.read_csv(
        data_dir / "olist_geolocation_dataset.csv",
        usecols=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
        ],
    ).rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "latitude",
            "geolocation_lng": "longitude",
        }
    )

    # 같은 우편번호 앞자리에 좌표가 여러 개 있으면 중앙값으로 대표 좌표를 만들어.
    valid_locations = locations.loc[
        locations["latitude"].between(-90, 90)
        & locations["longitude"].between(-180, 180)
    ]

    zip_coordinates = (
        valid_locations.groupby("customer_zip_code_prefix", as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
    )

    orders_with_zip = orders_for_region.merge(
        map_customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    zip_orders = (
        orders_with_zip.groupby("customer_zip_code_prefix", as_index=False)
        .agg(order_count=("order_id", "nunique"))
    )

    map_data = zip_orders.merge(
        zip_coordinates,
        on="customer_zip_code_prefix",
        how="inner",
        validate="one_to_one",
    )

    if map_data.empty:
        st.info("우편번호로 연결할 수 있는 위치 기록이 없어.")
    else:
        # 주문 수를 지도에서 보기 좋은 점 크기로 바꿔.
        map_data["point_size"] = map_data["order_count"].pow(0.5) * 400

        st.map(
            map_data,
            latitude="latitude",
            longitude="longitude",
            size="point_size",
            zoom=3,
        )

        mapped_orders = int(map_data["order_count"].sum())
        total_orders = orders_for_region["order_id"].nunique()

        st.caption(
            f"전체 주문 {total_orders:,}건 중 위치를 연결한 {mapped_orders:,}건 표시. "
            "점의 위치는 고객 주소의 우편번호 앞자리별 대표 좌표야."
        )

    st.divider()
    st.subheader("비슷한 상품 찾기")

    import joblib

    recommendation_path = Path(__file__).with_name("olist_recommendation.pkl")

    if not recommendation_path.is_file():
        st.info("app.py와 같은 폴더에 olist_recommendation.pkl을 넣어 줘.")
    else:
        artifact = joblib.load(recommendation_path)

        products = artifact["products"]
        features = artifact["features"]
        scaler = artifact["scaler"]
        models = artifact["models_by_category"]

        category = st.selectbox("카테고리 선택", sorted(models))
        category_model = models[category]

        selected_row = st.selectbox(
            "기준 상품 선택",
            options=category_model["row_indices"].tolist(),
            format_func=lambda row: products.at[row, "product_id"],
        )

        if st.button("비슷한 상품 찾기"):
            selected_features = products.loc[[selected_row], features]
            scaled_features = scaler.transform(selected_features)

            row_indices = category_model["row_indices"]
            neighbor_positions = category_model["model"].kneighbors(
                scaled_features,
                n_neighbors=min(6, len(row_indices)),
                return_distance=False,
            )[0]

            recommended_rows = [
                row for row in row_indices[neighbor_positions]
                if row != selected_row
            ][:5]

            column_labels = {
                "product_id": "상품 ID",
                "product_weight_g": "무게 (g)",
                "product_length_cm": "길이 (cm)",
                "product_height_cm": "높이 (cm)",
                "product_width_cm": "너비 (cm)",
            }

            st.write("**기준 상품**")
            st.dataframe(
                products.loc[[selected_row], ["product_id", *features]]
                .rename(columns=column_labels),
                hide_index=True,
            )

            st.write("**비슷한 상품**")
            st.dataframe(
                products.loc[recommended_rows, ["product_id", *features]]
                .rename(columns=column_labels),
                hide_index=True,
            )


with seller_tab:
    st.subheader("판매자 이탈 위험 예측")
    st.caption(
        "최근 90일 판매 흐름으로 이후 90일 주문 중단 위험을 예측해. "
        "시간순으로 과거 구간을 나눠 평가하는 시뮬레이션이야."
    )

    try:
        with st.spinner("판매자 주문 이력을 학습하고 평가하는 중…"):
            predictions, churn_metrics = get_seller_churn_report(str(data_dir))

        threshold = st.slider(
            "이탈 위험 표시 기준",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key="seller_churn_threshold",
        )

        metric_columns = st.columns(4)
        metric_columns[0].metric("정밀도", f"{churn_metrics['precision']:.1%}")
        metric_columns[1].metric("재현율", f"{churn_metrics['recall']:.1%}")
        metric_columns[2].metric("F1", f"{churn_metrics['f1']:.1%}")
        metric_columns[3].metric("예측 기준일", churn_metrics["snapshot_date"])

        st.caption(
            f"학습 기준일까지 {churn_metrics['train_through']} · "
            f"시간순 평가 시작 {churn_metrics['test_from']} · "
            "최근 90일 활동, 직전 90일 변화, 마지막 주문 후 경과일을 사용해."
        )

        flagged = predictions.loc[
            predictions["churn_probability"].ge(threshold)
        ].copy()
        st.metric("선택한 기준의 점검 판매자", f"{len(flagged):,}명")

        labels = {
            "seller_id": "판매자 ID",
            "churn_probability": "이탈 위험 점수",
            "orders_recent_90d": "최근 90일 주문 수",
            "items_recent_90d": "최근 90일 상품 수",
            "revenue_recent_90d": "최근 90일 상품 판매액 (R$)",
            "orders_previous_90d": "직전 90일 주문 수",
            "revenue_previous_90d": "직전 90일 상품 판매액 (R$)",
            "days_since_last_order": "마지막 주문 후 일수",
            "churned_next_90d": "이후 90일 실제 이탈",
        }

        if flagged.empty:
            st.info("현재 기준을 넘는 판매자가 없어.")
        else:
            shown = flagged[list(labels)].rename(columns=labels)
            shown["이탈 위험 점수"] = shown["이탈 위험 점수"].map(
                lambda score: f"{score:.1%}"
            )
            shown["이후 90일 실제 이탈"] = shown[
                "이후 90일 실제 이탈"
            ].map({0: "유지", 1: "이탈"})
            st.dataframe(shown, hide_index=True, use_container_width=True)

        st.caption(
            "이 화면은 과거 주문 자료를 시간순으로 나눈 검증용 예측이야. "
            "표의 실제 이탈은 과거 평가 결과이며 현재 판매자 상태가 아니야."
        )
    except (FileNotFoundError, ValueError) as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"판매자 이탈 예측을 준비하지 못했어: {error}")

    st.subheader("판매자 유형")

    cluster_path = Path(__file__).with_name("olist_seller_cluster_summary.csv")

    if cluster_path.is_file():
        cluster_summary = pd.read_csv(cluster_path).rename(
            columns={
                "cluster": "그룹 번호",
                "seller_count": "판매자 수",
                "median_item_count": "판매 상품 수 중앙값",
                "median_item_value": "상품 판매액 중앙값 (R$)",
            }
        )

        st.dataframe(cluster_summary, hide_index=True)
        st.caption("판매액은 배송비를 제외한 상품 금액이야.")
    else:
        st.warning("판매자 유형 요약 CSV를 app.py와 같은 폴더에 넣어 줘.")

    st.subheader("판매자 지역별 주문 상품 수")

    items_for_region = pd.read_csv(
        data_dir / "olist_order_items_dataset.csv",
        usecols=["seller_id"],
    )

    sellers = pd.read_csv(
        data_dir / "olist_sellers_dataset.csv",
        usecols=["seller_id", "seller_state"],
    )

    items_with_seller_region = items_for_region.merge(
        sellers,
        on="seller_id",
        how="left",
        validate="many_to_one",
    )

    items_with_seller_region["seller_state"] = (
        items_with_seller_region["seller_state"].fillna("미기록")
    )

    seller_state_summary = (
        items_with_seller_region.groupby("seller_state")
        .size()
        .reset_index(name="item_count")
        .nlargest(10, "item_count")
    )

    seller_state_chart = (
        alt.Chart(seller_state_summary)
        .mark_bar(color="#337CB0")
        .encode(
            x=alt.X("item_count:Q", title="주문 상품 수 (행)"),
            y=alt.Y("seller_state:N", title="판매자 지역 (주)", sort="-x"),
            tooltip=[
                alt.Tooltip("seller_state:N", title="판매자 지역"),
                alt.Tooltip("item_count:Q", title="주문 상품 수", format=",d"),
            ],
        )
    )

    st.altair_chart(seller_state_chart)
    st.caption(
        "원본 주문 상품 CSV 전체 기준 · 상위 10개 지역 표시. "
        "판매자 지역이 없으면 ‘미기록’으로 집계해."
    )

    st.divider()

with question_tab:
    st.subheader("올리스트 데이터에 질문하기")
    st.caption(
        "질문의 뜻과 가까운 사실을 임베딩으로 찾고, 검색된 근거 안에서 답해."
    )

    facts_path = Path(__file__).with_name("olist_facts.csv")

    with st.form("olist_question_form"):
        question = st.text_input(
            "질문",
            placeholder="신용카드 결제 금액과 지연 배송의 별점 차이를 알려 줘.",
        )
        submitted = st.form_submit_button("근거 찾기")

    if submitted:
        if not question.strip():
            st.warning("질문을 입력해 줘.")
        elif not facts_path.is_file():
            st.warning("app.py와 같은 폴더에 olist_facts.csv를 넣어 줘.")
        else:
            try:
                facts = pd.read_csv(facts_path, encoding="utf-8-sig").fillna("")
                documents = tuple(
                    f"분야: {row['분야']}\n내용: {row['내용']}\n"
                    f"기준: {row['기준']}\n출처: {row['출처']}"
                    for row in facts.to_dict(orient="records")
                )

                with st.spinner("질문과 사실표를 임베딩으로 비교하는 중…"):
                    document_vectors = embed_fact_documents(documents)
                    query_vector = get_openai_client().embeddings.create(
                        model="text-embedding-3-small",
                        input=question.strip(),
                    ).data[0].embedding

                query_length = sum(value * value for value in query_vector) ** 0.5
                ranked = []
                for index, document_vector in enumerate(document_vectors):
                    doc_length = sum(value * value for value in document_vector) ** 0.5
                    similarity = sum(
                        query_value * document_value
                        for query_value, document_value in zip(query_vector, document_vector)
                    ) / max(query_length * doc_length, 1e-12)
                    ranked.append((similarity, index))

                ranked.sort(reverse=True)
                matches = [(score, facts.iloc[index]) for score, index in ranked[:3]]
                context = "\n\n".join(
                    f"[{fact['사실ID']}] {fact['내용']} "
                    f"집계 기준: {fact['기준']} 출처: {fact['출처']}"
                    for _, fact in matches
                )

                with st.spinner("찾은 근거를 바탕으로 답변을 만드는 중…"):
                    response = get_openai_client().chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "제공된 사실만 사용해 한국어로 짧게 답해. "
                                    "수치·단위·집계 범위를 바꾸지 마. "
                                    "근거가 부족하면 부족하다고 말하고 추측하지 마."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"질문: {question}\n\n검색된 사실:\n{context}",
                            },
                        ],
                    )

                st.write(response.choices[0].message.content)
                st.subheader("검색된 근거")
                for score, fact in matches:
                    with st.expander(
                        f"{fact['사실ID']} · {fact['분야']} · 유사도 {score:.3f}"
                    ):
                        st.write(fact["내용"])
                        st.caption(f"집계 기준 · {fact['기준']}")
                        st.caption(f"출처 · {fact['출처']}")
            except OpenAIError as error:
                st.error(
                    "임베딩 검색이나 답변을 준비하지 못했어. "
                    f"OPENAI_API_KEY와 API 상태를 확인해 줘. ({error})"
                )
            except (KeyError, ValueError, TypeError) as error:
                st.error(f"사실표 형식을 읽지 못했어: {error}")
