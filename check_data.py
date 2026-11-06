from pathlib import Path

import pandas as pd


data_dir = Path(__file__).resolve().parent / "data"

checks = [
    ("olist_orders_dataset.xlsx", "olist_orders_dataset.csv", ["order_id"]),
    ("olist_orders_dataset(1).xlsx", "olist_orders_dataset.csv", ["order_id"]),
    ("olist_orders_dataset(2).xlsx", "olist_orders_dataset.csv", ["order_id"]),
    ("olist_orders_dataset(3).xlsx", "olist_orders_dataset.csv", ["order_id"]),
    (
        "olist_order_items_dataset.xlsx",
        "olist_order_items_dataset.csv",
        ["order_id", "order_item_id"],
    ),
]

csv_tables = {}

for excel_name, csv_name, keys in checks:
    print(f"\n--- {excel_name} ---")

    if not (data_dir / excel_name).is_file() or not (data_dir / csv_name).is_file():
        print("비교할 파일이 없어. data 폴더와 파일명을 확인해 줘.")
        continue

    if csv_name not in csv_tables:
        csv_tables[csv_name] = pd.read_csv(
            data_dir / csv_name,
            dtype="string",
        )

    csv_table = csv_tables[csv_name]
    excel_table = pd.read_excel(data_dir / excel_name, dtype="string")

    print(f"엑셀 {len(excel_table):,}행 / CSV {len(csv_table):,}행")
    print("엑셀 열:", ", ".join(excel_table.columns))
    print("CSV 열:", ", ".join(csv_table.columns))

    if any(key not in excel_table or key not in csv_table for key in keys):
        print("비교에 필요한 번호 열이 없어.")
        continue

    excel_rows = excel_table.dropna(subset=keys).set_index(keys)
    csv_rows = csv_table.dropna(subset=keys).set_index(keys)

    excel_ids = excel_rows.index.unique()
    csv_ids = csv_rows.index.unique()
    shared_ids = excel_ids.intersection(csv_ids)

    print(f"양쪽에 있는 고유 번호 조합: {len(shared_ids):,}개")
    print(f"엑셀에만 있는 번호 조합: {len(excel_ids.difference(csv_ids)):,}개")
    print(f"CSV에만 있는 번호 조합: {len(csv_ids.difference(excel_ids)):,}개")

    excel_unique = excel_rows.loc[~excel_rows.index.duplicated(keep=False)]
    csv_unique = csv_rows.loc[~csv_rows.index.duplicated(keep=False)]
    comparable_ids = excel_unique.index.intersection(csv_unique.index)
    common_columns = excel_unique.columns.intersection(csv_unique.columns)

    if len(common_columns) > 0:
        excel_values = excel_unique.loc[comparable_ids, common_columns]
        csv_values = csv_unique.loc[comparable_ids, common_columns]
        same_values = excel_values.eq(csv_values).fillna(False)
        both_blank = excel_values.isna() & csv_values.isna()
        differences = (~(same_values | both_blank)).any(axis=1)
        if excel_name == "olist_order_items_dataset.xlsx":
            different_cells = ~(same_values | both_blank)
            differences_by_column = different_cells.sum().sort_values(ascending=False)

            for column in ["price", "freight_value"]:
                excel_amount = pd.to_numeric(excel_values[column], errors="coerce")
                csv_amount = pd.to_numeric(csv_values[column], errors="coerce")

                both_numeric = excel_amount.notna() & csv_amount.notna()
                different_amounts = both_numeric & excel_amount.ne(csv_amount)

                needs_review = ~both_numeric & ~(
                    excel_values[column].isna() & csv_values[column].isna()
                )

                print(f"{column} · 실제 금액이 다른 행: {different_amounts.sum():,}개")
                print(f"{column} · 한쪽 누락 또는 숫자 변환 실패: {needs_review.sum():,}개")

            print("\n[주문 상품 파일 · 열별 표기 차이]")
            print(differences_by_column.to_string())

            for column in differences_by_column[differences_by_column > 0].index:
                different_rows = different_cells[column]
                examples = pd.DataFrame(
                    {
                        "엑셀 값": excel_values.loc[different_rows, column],
                        "CSV 값": csv_values.loc[different_rows, column],
                    }
                ).head(3)

                print(f"\n[{column} · 차이 예시]")
                print(examples.to_string())

        print(f"공통 열의 값 표기가 다른 기록: {differences.sum():,}개")
        print(f"중복 번호 때문에 값 비교에서 제외: {len(shared_ids) - len(comparable_ids):,}개")

print("\n--- 원본 주문표와 수업 통합표 대조 ---")

original_order_ids = set(
    pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        usecols=["order_id"],
        dtype={"order_id": "string"},
    )["order_id"].dropna()
)

master_order_ids = set(
    pd.read_csv(
        data_dir / "olist_master_data.csv",
        usecols=["order_id"],
        dtype={"order_id": "string"},
    )["order_id"].dropna()
)

print(f"원본 고유 주문 수: {len(original_order_ids):,}건")
print(f"통합표 고유 주문 수: {len(master_order_ids):,}건")
print(f"원본에만 있는 주문 번호: {len(original_order_ids - master_order_ids):,}개")
print(f"통합표에만 있는 주문 번호: {len(master_order_ids - original_order_ids):,}개")

