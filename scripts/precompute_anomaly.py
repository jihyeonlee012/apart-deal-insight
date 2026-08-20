"""
사전 계산 스크립트 — 이상치 탐지 모델 3개 실행 후 결과를 data/cache/ 에 저장합니다.

실행 방법 (프로젝트 루트에서):
    python scripts/precompute_anomaly.py

저장 파일 목록 (data/cache/):
    anomaly_summary.json            전국 단일 모델 KPI
    anomaly_top30.parquet           전국 TOP 30 특이거래
    anomaly_region_counts.parquet   지역별 특이거래 건수 TOP 20
    anomaly_normal_ppy.parquet      정상거래 평당가 샘플 (박스플롯용)
    anomaly_anomaly_ppy.parquet     특이거래 평당가 (박스플롯용)
    anomaly_scores.parquet          특이거래 score 분포 (히스토그램용)

    seoul_kpi.json                  서울 구별 모델 KPI
    seoul_summary.parquet           구별 요약 테이블
    seoul_top_by_district.parquet   구별 TOP 10 특이거래

    location_kpi.json               전국 시군구별 모델 KPI
    location_top10.parquet          시군구별 TOP 1 → 전국 TOP 10
    location_summary.parquet        시군구별 요약 테이블
    location_anomalies.parquet      전국 특이거래 전체 (시군구 선택 조회용)
    location_anomalies_map.parquet  전국 특이거래 지도용 (위도/경도/거래연도/거래월 포함)
"""
 
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from models.anomaly.anomaly_transaction_model import AnomalyTransactionModel
from models.anomaly.seoul_anomaly_transaction_model import SeoulDistrictAnomalyModel
from models.anomaly.location_anomaly_transaction_model import LocationAnomalyModel


# =============================================================================
# 데이터 로드 — DB 캐시 → CSV 순서로 폴백
# =============================================================================

def load_data() -> pd.DataFrame:
    try:
        from utils.db import load_apart_deals
        print("  데이터 로드: DB/Parquet 캐시 사용")
        df = load_apart_deals()
    except Exception as e:
        raise RuntimeError(f"DB/Parquet 로드 실패: {e}")

    df["거래금액"] = df["거래금액"].astype(str).str.replace(",", "", regex=False).astype(float)

    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"  중복 제거: {before:,}건 → {len(df):,}건 ({before - len(df):,}건 제거)")
    return df


# =============================================================================
# 모델 1: 전국 단일 (AnomalyTransactionModel)
# =============================================================================

def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """행 순서를 DB/CSV 관계없이 고정하여 sample(random_state=42)이 재현 가능하도록 정렬."""
    sort_keys = [c for c in ["거래일", "시군구", "아파트", "거래금액", "전용면적"] if c in df.columns]
    return df.sort_values(sort_keys).reset_index(drop=True)


def run_anomaly(df: pd.DataFrame) -> None:
    print("\n[1/3] 전국 단일 모델 (AnomalyTransactionModel)")
    t = time.time()

    df_fit = df
    model = AnomalyTransactionModel(contamination=0.03, sample_size=200_000, random_state=42)
    model.fit_from_dataframe(df_fit)
    print(f"  학습 완료 ({time.time()-t:.0f}초)")

    batch_size  = 100_000
    total_batch = (len(df_fit) - 1) // batch_size + 1
    print(f"  전체 데이터 탐지 중 (총 {total_batch}개 배치, 배치당 {batch_size:,}건)...")
    parts = []
    for i in range(0, len(df_fit), batch_size):
        batch_no = i // batch_size + 1
        chunk    = df_fit.iloc[i: i + batch_size].copy()
        print(f"    배치 {batch_no}/{total_batch}  [{i:,} ~ {i+len(chunk):,}건]")
        parts.append(model.detect_from_dataframe(chunk))

    result = pd.concat(parts, ignore_index=True)
    result["anomaly_rank"] = (
        result["anomaly_score"].rank(method="first", ascending=True).astype(int)
    )

    anomaly_mask = result["anomaly_raw_label"] == -1
    normal_mask  = result["anomaly_raw_label"] == 1

    # ── KPI
    summary = {
        "total_count":     int(len(result)),
        "anomaly_count":   int(anomaly_mask.sum()),
        "normal_count":    int(normal_mask.sum()),
        "anomaly_ratio":   float(anomaly_mask.mean()),
        "avg_ppy_normal":  float(result.loc[normal_mask,  "평당가"].mean()),
        "avg_ppy_anomaly": float(result.loc[anomaly_mask, "평당가"].mean()),
    }
    with open(CACHE_DIR / "anomaly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── TOP 30 테이블
    top30 = result[anomaly_mask].sort_values("anomaly_score").head(30).reset_index(drop=True)
    top30.to_parquet(CACHE_DIR / "anomaly_top30.parquet", index=False)

    # ── 지역별 특이거래 건수 TOP 20
    region_counts = (
        result.loc[anomaly_mask, "시군구"]
        .value_counts().head(20).reset_index()
    )
    region_counts.columns = ["시군구", "특이거래수"]
    region_counts.to_parquet(CACHE_DIR / "anomaly_region_counts.parquet", index=False)

    # ── 박스플롯용 평당가 샘플
    normal_ppy  = result.loc[normal_mask,  "평당가"].dropna()
    anomaly_ppy = result.loc[anomaly_mask, "평당가"].dropna()

    normal_vis = normal_ppy[
        normal_ppy <= float(normal_ppy.quantile(0.99))
    ].sample(min(10_000, len(normal_ppy)), random_state=42)
    anomaly_vis = anomaly_ppy[anomaly_ppy <= float(anomaly_ppy.quantile(0.99))]

    pd.DataFrame({"평당가": normal_vis}).to_parquet(
        CACHE_DIR / "anomaly_normal_ppy.parquet", index=False
    )
    pd.DataFrame({"평당가": anomaly_vis}).to_parquet(
        CACHE_DIR / "anomaly_anomaly_ppy.parquet", index=False
    )

    # ── score 분포 (히스토그램)
    pd.DataFrame({"anomaly_score": result.loc[anomaly_mask, "anomaly_score"].dropna()}).to_parquet(
        CACHE_DIR / "anomaly_scores.parquet", index=False
    )

    print(f"  저장 완료 ({time.time()-t:.0f}초) - 6개 파일")


# =============================================================================
# 모델 2: 서울 구별 (SeoulDistrictAnomalyModel)
# =============================================================================

def run_seoul(df: pd.DataFrame) -> None:
    print("\n[2/3] 서울 구별 모델 (SeoulDistrictAnomalyModel)")
    t = time.time()

    df_fit = df
    model = SeoulDistrictAnomalyModel(contamination=0.03, random_state=42)
    model.fit_from_dataframe(df_fit)
    print(f"  학습 완료 ({time.time()-t:.0f}초)")

    result       = model.detect_from_dataframe(df_fit)
    anomaly_mask = result["anomaly_raw_label"] == -1
    normal_mask  = result["anomaly_raw_label"] == 1

    # summarize_by_district / top_anomalies_by_district 결과를 result에서 직접 도출 (재탐지 방지)
    result_notna = result[result["anomaly_raw_label"].notna()]
    rows = []
    for district, group in result_notna.groupby("구명"):
        anomaly = group[group["anomaly_raw_label"] == -1]
        normal  = group[group["anomaly_raw_label"] == 1]
        total   = len(group)
        ac      = len(anomaly)
        rows.append({
            "구명": district,
            "총거래수": total,
            "특이거래수": ac,
            "특이거래비율(%)": round(ac / total * 100, 2) if total else 0.0,
            "정상거래_평균평당가": round(float(normal["평당가"].mean()), 0) if len(normal) else np.nan,
            "특이거래_평균평당가": round(float(anomaly["평당가"].mean()), 0) if ac else np.nan,
            "평균평당가_차이": round(float(anomaly["평당가"].mean() - normal["평당가"].mean()), 0) if (ac and len(normal)) else np.nan,
            "특이거래_평균거래금액": round(float(anomaly["거래금액"].mean()), 0) if ac else np.nan,
            "특이거래_최저score": round(float(anomaly["anomaly_score"].min()), 4) if ac else np.nan,
        })
    summary = pd.DataFrame(rows).set_index("구명").sort_values("특이거래비율(%)", ascending=False)

    top_by_district = (
        result[anomaly_mask]
        .sort_values("anomaly_score", ascending=True)
        .groupby("구명", group_keys=False)
        .head(10)
        .sort_values(["구명", "anomaly_score"])
        .reset_index(drop=True)
    )

    kpi = {
        "total_count":     int(len(result)),
        "anomaly_count":   int(anomaly_mask.sum()),
        "normal_count":    int(normal_mask.sum()),
        "anomaly_ratio":   float(anomaly_mask.mean()),
        "avg_ppy_normal":  float(result.loc[normal_mask,  "평당가"].mean()) if normal_mask.any()  else 0.0,
        "avg_ppy_anomaly": float(result.loc[anomaly_mask, "평당가"].mean()) if anomaly_mask.any() else 0.0,
    }
    with open(CACHE_DIR / "seoul_kpi.json", "w", encoding="utf-8") as f:
        json.dump(kpi, f, ensure_ascii=False, indent=2)

    summary.reset_index().to_parquet(CACHE_DIR / "seoul_summary.parquet", index=False)
    top_by_district.to_parquet(CACHE_DIR / "seoul_top_by_district.parquet", index=False)

    print(f"  저장 완료 ({time.time()-t:.0f}초) - 3개 파일")


# =============================================================================
# 모델 3: 전국 시군구별 (LocationAnomalyModel)
# =============================================================================

def run_location(df: pd.DataFrame) -> None:
    print("\n[3/3] 전국 시군구별 모델 (LocationAnomalyModel)")
    t = time.time()

    df_fit = df
    model = LocationAnomalyModel(contamination=0.03, random_state=42)
    model.fit_from_dataframe(df_fit)
    print(f"  학습 완료 ({time.time()-t:.0f}초)")

    result        = model.detect_from_dataframe(df_fit)
    anomaly_mask  = result["anomaly_raw_label"] == -1
    detected_mask = result["anomaly_raw_label"].notna()

    # summarize_by_location / top_anomalies_top1_per_region 결과를 result에서 직접 도출 (재탐지 방지)
    result_notna = result[result["anomaly_raw_label"].notna()]
    rows = []
    for sigungu, group in result_notna.groupby("시군구"):
        anomaly = group[group["anomaly_raw_label"] == -1]
        normal  = group[group["anomaly_raw_label"] == 1]
        total   = len(group)
        ac      = len(anomaly)
        rows.append({
            "시군구": sigungu,
            "총거래수": total,
            "특이거래수": ac,
            "특이거래비율(%)": round(ac / total * 100, 2) if total else 0.0,
            "정상거래_평균평당가": round(float(normal["평당가"].mean()), 0) if len(normal) else np.nan,
            "특이거래_평균평당가": round(float(anomaly["평당가"].mean()), 0) if ac else np.nan,
            "평균평당가_차이": round(float(anomaly["평당가"].mean() - normal["평당가"].mean()), 0) if (ac and len(normal)) else np.nan,
            "특이거래_평균거래금액": round(float(anomaly["거래금액"].mean()), 0) if ac else np.nan,
            "특이거래_최저score": round(float(anomaly["anomaly_score"].min()), 4) if ac else np.nan,
        })
    summary = pd.DataFrame(rows).set_index("시군구").sort_values("특이거래비율(%)", ascending=False)

    top10 = (
        result[anomaly_mask]
        .sort_values("anomaly_score", ascending=True)
        .groupby("시군구", group_keys=False)
        .head(1)
        .sort_values("anomaly_score", ascending=True)
        .head(10)
        .reset_index(drop=True)
    )

    best_score = float(top10["anomaly_score"].min()) if len(top10) > 0 else float("nan")

    kpi = {
        "total_count":    int(len(result)),
        "anomaly_count":  int(anomaly_mask.sum()),
        "detected_count": int(detected_mask.sum()),
        "location_count": int(len(summary)),
        "best_score":     None if np.isnan(best_score) else best_score,
    }
    with open(CACHE_DIR / "location_kpi.json", "w", encoding="utf-8") as f:
        json.dump(kpi, f, ensure_ascii=False, indent=2)

    top10.to_parquet(CACHE_DIR / "location_top10.parquet", index=False)
    summary.reset_index().to_parquet(CACHE_DIR / "location_summary.parquet", index=False)

    # 시군구 선택 조회용 — 특이거래 행만 저장
    anomaly_rows = result[anomaly_mask].copy()
    keep_cols = [c for c in [
        "시군구", "아파트", "거래금액", "전용면적", "층",
        "평당가", "anomaly_score", "anomaly_rank"
    ] if c in anomaly_rows.columns]
    anomaly_rows[keep_cols].to_parquet(CACHE_DIR / "location_anomalies.parquet", index=False)

    # 이상치 지도(시계열 슬라이더)용 — 위도/경도/거래시점 포함 저장
    map_cols = [c for c in [
        "시군구", "아파트", "거래금액", "전용면적", "층",
        "평당가", "위도", "경도", "거래연도", "거래월",
        "anomaly_score", "anomaly_rank"
    ] if c in anomaly_rows.columns]
    anomaly_rows[map_cols].dropna(subset=["위도", "경도"]).to_parquet(
        CACHE_DIR / "location_anomalies_map.parquet", index=False
    )

    print(f"  저장 완료 ({time.time()-t:.0f}초) - 5개 파일")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    total_start = time.time()
    print("=" * 60)
    print("이상치 탐지 사전 계산 시작")
    print(f"저장 경로: {CACHE_DIR}")
    print("=" * 60)

    print("\n데이터 로드 중...")
    df = load_data()
    print(f"  로드 완료: {len(df):,}건")
    df = _sort_df(df)

    run_anomaly(df)
    run_seoul(df)
    run_location(df)

    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"전체 완료 - 총 소요 시간: {elapsed/60:.1f}분")
    print(f"저장된 파일 수: 14개")
    print("이제 Streamlit 앱이 캐시 파일을 즉시 로드합니다.")
    print("=" * 60)
