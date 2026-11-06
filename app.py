from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from openai import OpenAI, OpenAIError

st.set_page_config(page_title="Olist 배송 분석")

st.title("배송 지연과 리뷰 별점")
order_tab, delivery_tab, product_tab, question_tab = st.tabs(
    ["주문·결제", "배송·리뷰", "상품·지역", "데이터 질문"]
)

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
    st.subheader("판매자 이탈 점검")

    churn_summary_path = Path(__file__).with_name(
        "olist_churn_threshold_summary.csv"
    )

    if churn_summary_path.is_file():
        churn_summary = pd.read_csv(
            churn_summary_path,
            dtype={"이탈 점수 기준": str},
        ).set_index("이탈 점수 기준")

        threshold = st.selectbox(
            "이탈 점수 기준",
            churn_summary.index.tolist(),
            key="seller_churn_threshold",
        )

        result = churn_summary.loc[threshold]
        metrics = st.columns(2)

        metrics[0].metric("점검 명단", f"{result['점검 명단']:,}명")
        metrics[1].metric("찾아낸 이탈", f"{result['찾아낸 이탈']:,}명")
        metrics[0].metric("잘못 지목", f"{result['잘못 지목']:,}명")
        metrics[1].metric("놓친 이탈", f"{result['놓친 이탈']:,}명")

        st.caption("코랩 시험 데이터에서 실제 이탈한 판매자 108명 기준이야.")
    else:
        st.warning("이탈 점수 비교 CSV를 app.py와 같은 폴더에 넣어 줘.")

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
            y=alt.Y("customer_state:N", title="고객 지역 (주)", sort="-x"),
            tooltip=[
                alt.Tooltip("customer_state:N", title="고객 지역"),
                alt.Tooltip("order_count:Q", title="주문 수", format=",d"),
            ],
        )
    )

    st.altair_chart(state_chart)
    st.caption(
        "원본 주문 CSV 전체 기준 · 주문 수 상위 10개 지역 표시. "
        "고객표에 지역이 없으면 ‘미기록’으로 집계해."
    )

    st.divider()
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

with question_tab:
    st.subheader("올리스트 데이터에 질문하기")

    facts_path = Path(__file__).with_name("olist_facts.csv")

    with st.form("olist_question_form"):
        question = st.text_input(
            "질문",
            placeholder="지연 주문 중 별점 2점 이하는 얼마나 돼?",
        )
        submitted = st.form_submit_button("근거 찾기")

    if submitted:
        normalized_question = "".join(question.split())
        fact_id = None

        if any(word in normalized_question for word in ("2점이하", "저평점")):
            fact_id = "F02"
        elif any(word in normalized_question for word in ("평균별점", "평균평점")):
            fact_id = "F01"

        if fact_id is None:
            st.info("현재 참고할 수 있는 데이터에서 이 질문에 답할 근거를 찾지 못했어요.")
        elif not facts_path.is_file():
            st.warning("olist_facts.csv를 app.py와 같은 폴더에 넣어 줘.")
        else:
            facts = pd.read_csv(facts_path)
            matches = facts.loc[facts["사실ID"].eq(fact_id)]

            if matches.empty:
                st.info("사실표에서 해당 근거를 찾지 못했어.")
            else:
                fact = matches.iloc[0]

                try:
                    with st.spinner("근거를 바탕으로 답변을 만드는 중…"):
                        response = OpenAI().chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "제공된 근거만 사용해 한국어로 짧게 답해. "
                                        "수치를 정확히 전달해. "
                                        "근거로 답할 수 없으면 근거가 부족하다고 말해."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"질문: {question}\n"
                                        f"분석 결과: {fact['내용']}\n"
                                        f"집계 기준: {fact['기준']}"
                                    ),
                                },
                            ],
                        )

                    st.write(response.choices[0].message.content)
                except OpenAIError:
                    st.warning("AI 답변을 가져오지 못했어. 아래는 사실표에서 찾은 근거야.")
                    st.write(fact["내용"])

                st.caption(f"집계 기준 · {fact['기준']}")
                st.caption(f"출처 · {fact['출처']} ({fact['사실ID']})")

