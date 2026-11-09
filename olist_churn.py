from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


FEATURES = [
    "orders_recent_90d",
    "items_recent_90d",
    "revenue_recent_90d",
    "orders_previous_90d",
    "items_previous_90d",
    "revenue_previous_90d",
    "days_since_last_order",
]


def _seller_totals(window: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame(
            columns=[
                "seller_id",
                f"orders_{suffix}",
                f"items_{suffix}",
                f"revenue_{suffix}",
            ]
        )

    return (
        window.groupby("seller_id", as_index=False)
        .agg(
            **{
                f"orders_{suffix}": ("order_id", "nunique"),
                f"items_{suffix}": ("item_count", "sum"),
                f"revenue_{suffix}": ("revenue", "sum"),
            }
        )
    )


def build_churn_report(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Train on earlier 90-day snapshots and score a later historical snapshot."""
    data_path = Path(data_dir)
    orders_path = data_path / "olist_orders_dataset.csv"
    items_path = data_path / "olist_order_items_dataset.csv"

    if not orders_path.is_file() or not items_path.is_file():
        raise FileNotFoundError(
            "data 폴더에 주문 CSV와 주문 상품 CSV가 모두 필요해."
        )

    orders = pd.read_csv(
        orders_path,
        usecols=["order_id", "order_purchase_timestamp", "order_status"],
    )
    orders["purchase_date"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )
    orders = orders.loc[
        orders["purchase_date"].notna()
        & orders["order_status"].eq("delivered"),
        ["order_id", "purchase_date"],
    ]

    items = pd.read_csv(
        items_path,
        usecols=["order_id", "seller_id", "price"],
    ).dropna(subset=["seller_id"])

    seller_orders = (
        items.groupby(["seller_id", "order_id"], as_index=False)
        .agg(
            item_count=("order_id", "size"),
            revenue=("price", "sum"),
        )
        .merge(
            orders,
            on="order_id",
            how="inner",
            validate="many_to_one",
        )
    )

    if seller_orders.empty:
        raise ValueError("배송 완료 주문에서 판매자 활동을 찾지 못했어.")

    first_date = seller_orders["purchase_date"].min()
    last_date = seller_orders["purchase_date"].max()
    first_anchor = (first_date + pd.Timedelta(days=180)).normalize()
    last_anchor = (last_date - pd.Timedelta(days=90)).normalize()
    anchors = pd.date_range(first_anchor, last_anchor, freq="MS")

    if len(anchors) < 6:
        raise ValueError("90일 예측 구간을 만들 수 있는 주문 이력이 부족해.")

    snapshots = []

    for anchor in anchors:
        recent_start = anchor - pd.Timedelta(days=90)
        previous_start = anchor - pd.Timedelta(days=180)
        future_end = anchor + pd.Timedelta(days=90)

        recent = seller_orders.loc[
            seller_orders["purchase_date"].ge(recent_start)
            & seller_orders["purchase_date"].lt(anchor)
        ]
        previous = seller_orders.loc[
            seller_orders["purchase_date"].ge(previous_start)
            & seller_orders["purchase_date"].lt(recent_start)
        ]
        future = seller_orders.loc[
            seller_orders["purchase_date"].ge(anchor)
            & seller_orders["purchase_date"].lt(future_end)
        ]

        if recent.empty:
            continue

        current = (
            _seller_totals(recent, "recent_90d")
            .merge(
                recent.groupby("seller_id", as_index=False)
                .agg(last_order_date=("purchase_date", "max")),
                on="seller_id",
                how="left",
                validate="one_to_one",
            )
            .merge(
                _seller_totals(previous, "previous_90d"),
                on="seller_id",
                how="left",
                validate="one_to_one",
            )
        )

        previous_columns = [
            "orders_previous_90d",
            "items_previous_90d",
            "revenue_previous_90d",
        ]
        current[previous_columns] = current[previous_columns].fillna(0)
        current["days_since_last_order"] = (
            anchor - current["last_order_date"]
        ).dt.days

        future_sellers = set(future["seller_id"].unique())
        current["churned_next_90d"] = (
            ~current["seller_id"].isin(future_sellers)
        ).astype(int)
        current["snapshot_date"] = anchor
        snapshots.append(current)

    if len(snapshots) < 6:
        raise ValueError("모델 학습에 필요한 판매자 활동 구간이 부족해.")

    records = pd.concat(snapshots, ignore_index=True).dropna(subset=FEATURES)
    dates = sorted(records["snapshot_date"].unique())
    split_at = max(1, int(len(dates) * 0.8))
    split_at = min(split_at, len(dates) - 1)
    training_dates = set(dates[:split_at])
    testing_dates = set(dates[split_at:])

    train = records.loc[records["snapshot_date"].isin(training_dates)]
    test = records.loc[records["snapshot_date"].isin(testing_dates)]

    if train["churned_next_90d"].nunique() < 2:
        raise ValueError("초기 학습 구간에 이탈·유지 사례가 모두 있어야 해.")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train["churned_next_90d"])

    test = test.copy()
    test["churn_probability"] = model.predict_proba(test[FEATURES])[
        :, list(model.classes_).index(1)
    ]
    test["predicted_churn"] = (test["churn_probability"] >= 0.5).astype(int)

    matrix = confusion_matrix(
        test["churned_next_90d"],
        test["predicted_churn"],
        labels=[0, 1],
    )
    final_date = test["snapshot_date"].max()
    predictions = test.loc[test["snapshot_date"].eq(final_date)].copy()
    predictions = predictions.sort_values(
        "churn_probability",
        ascending=False,
    )

    metrics = {
        "precision": precision_score(
            test["churned_next_90d"],
            test["predicted_churn"],
            zero_division=0,
        ),
        "recall": recall_score(
            test["churned_next_90d"],
            test["predicted_churn"],
            zero_division=0,
        ),
        "f1": f1_score(
            test["churned_next_90d"],
            test["predicted_churn"],
            zero_division=0,
        ),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "snapshot_date": pd.Timestamp(final_date).strftime("%Y-%m-%d"),
        "train_through": pd.Timestamp(max(training_dates)).strftime("%Y-%m-%d"),
        "test_from": pd.Timestamp(min(testing_dates)).strftime("%Y-%m-%d"),
        "feature_importance": dict(
            sorted(
                zip(FEATURES, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }

    return predictions, metrics
