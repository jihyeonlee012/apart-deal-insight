# 경로: app/pages/10_이상치분석.py

import json
import os
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import page_header, section_badge

st.markdown("""
<style>
.metric-card {
    background: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
}
.metric-label {
    font-size: 11px;
    color: #6B7280;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #172B4D;
    margin: 8px 0 4px;
}
.metric-sub {
    font-size: 12px;
    color: #9CA3AF;
}
.intro-box {
    background: linear-gradient(135deg, #FFF7ED 0%, #FEF3C7 100%);
    border: 1px solid #FCD34D;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #92400E;
    line-height: 1.8;
}
.notice-box {
    background: #FFF1F2;
    border-left: 4px solid #F43F5E;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 13px;
    color: #881337;
}
.no-cache-box {
    background: #F8FAFC;
    border: 1.5px dashed #94A3B8;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
    font-size: 14px;
    color: #475569;
    line-height: 1.9;
}
</style>
""", unsafe_allow_html=True)

page_header("이상 거래 탐지 분석")

st.markdown("""
<div class="intro-box">
    <b>IsolationForest 기반 이상 거래 탐지</b> — 아파트 실거래 데이터에서 일반적인 거래 패턴과 다른 <b>특이 거래</b>를 탐지합니다.<br>
    <b>전국 단일 모델</b>: 전국 평균 대비 절대 이상치 탐지 (초대형·초고가 거래 위주)<br>
    <b>서울 구별 모델</b>: 서울 25개 구마다 독립 학습, 구 내부 기준 상대 이상치 탐지<br>
    <b>전국 시군구별 모델</b>: 246개 시군구 내부 패턴 기준 상대 이상치 탐지 (지방 소도시 포함)
</div>
<div class="notice-box">
    ⚠️ 탐지 결과는 <b>통계적 이상치</b>이며 실제 불법·허위 거래와는 무관합니다.
    "모델 기준 특이 거래" 또는 "일반 패턴과 다른 거래"로 해석하세요.
</div>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR    = PROJECT_ROOT / "data" / "cache"

# =============================================================================
# 캐시 파일 로드 함수 (모델 학습 없음 — 즉시 반환)
# =============================================================================

ANOMALY_FILES = [
    "anomaly_summary.json", "anomaly_top30.parquet",
    "anomaly_region_counts.parquet", "anomaly_normal_ppy.parquet",
    "anomaly_anomaly_ppy.parquet",   "anomaly_scores.parquet",
]
SEOUL_FILES = [
    "seoul_kpi.json", "seoul_summary.parquet", "seoul_top_by_district.parquet",
]
LOCATION_FILES = [
    "location_kpi.json", "location_top10.parquet",
    "location_summary.parquet", "location_anomalies.parquet",
]
LOCATION_MAP_FILES = ["location_anomalies_map.parquet"]


def _cache_ready(file_list: list[str]) -> bool:
    return all((CACHE_DIR / f).exists() for f in file_list)


def _no_cache_msg(script: str = "scripts/precompute_anomaly.py") -> None:
    st.markdown(f"""
    <div class="no-cache-box">
        📂 <b>사전 계산 파일이 없습니다.</b><br>
        아래 명령어를 <b>프로젝트 루트</b>에서 한 번 실행한 뒤 페이지를 새로고침하세요.<br><br>
        <code style="background:#E2E8F0;padding:4px 10px;border-radius:6px;font-size:13px;">
            python {script}
        </code><br><br>
        소요 시간: 약 5~10분 (이후 즉시 로드)
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner="분석 결과 불러오는 중...")
def load_anomaly_cache():
    summary    = json.loads((CACHE_DIR / "anomaly_summary.json").read_text(encoding="utf-8"))
    top30      = pd.read_parquet(CACHE_DIR / "anomaly_top30.parquet")
    region_cnt = pd.read_parquet(CACHE_DIR / "anomaly_region_counts.parquet")
    normal_ppy = pd.read_parquet(CACHE_DIR / "anomaly_normal_ppy.parquet")["평당가"]
    anomaly_ppy= pd.read_parquet(CACHE_DIR / "anomaly_anomaly_ppy.parquet")["평당가"]
    scores     = pd.read_parquet(CACHE_DIR / "anomaly_scores.parquet")["anomaly_score"]
    return summary, top30, region_cnt, normal_ppy, anomaly_ppy, scores


@st.cache_data(show_spinner="분석 결과 불러오는 중...")
def load_seoul_cache():
    kpi             = json.loads((CACHE_DIR / "seoul_kpi.json").read_text(encoding="utf-8"))
    summary         = pd.read_parquet(CACHE_DIR / "seoul_summary.parquet")
    top_by_district = pd.read_parquet(CACHE_DIR / "seoul_top_by_district.parquet")
    return kpi, summary, top_by_district


@st.cache_data(show_spinner="분석 결과 불러오는 중...")
def load_location_cache():
    kpi       = json.loads((CACHE_DIR / "location_kpi.json").read_text(encoding="utf-8"))
    top10     = pd.read_parquet(CACHE_DIR / "location_top10.parquet")
    summary   = pd.read_parquet(CACHE_DIR / "location_summary.parquet")
    anomalies = pd.read_parquet(CACHE_DIR / "location_anomalies.parquet")
    return kpi, top10, summary, anomalies


@st.cache_data(show_spinner="지도 데이터 불러오는 중...")
def load_location_map_cache() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / "location_anomalies_map.parquet")
    df["위도"] = df["위도"].astype(float)
    df["경도"] = df["경도"].astype(float)
    df["연월"] = df["거래연도"].astype(int).astype(str) + "-" + df["거래월"].astype(int).astype(str).str.zfill(2)
    return df

# =============================================================================
# 공통 헬퍼
# =============================================================================

def _metric_card(col, label, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def _fmt_price(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].apply(lambda x: f"{x:,.0f}만원" if pd.notna(x) else "-")
    return out


def _fmt_score(df: pd.DataFrame, col: str = "anomaly_score") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    return out


def _anomaly_color(score: float, vmin: float, vmax: float) -> str:
    """anomaly_score가 낮을수록(더 특이할수록) 진한 빨강."""
    if vmax == vmin:
        return "#DC2626"
    ratio = (score - vmin) / (vmax - vmin)
    if ratio <= 0.20: return "#7F1D1D"
    if ratio <= 0.40: return "#B91C1C"
    if ratio <= 0.60: return "#DC2626"
    if ratio <= 0.80: return "#F87171"
    return "#FCA5A5"

# =============================================================================
# 탭 구성
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 전국 단일 모델",
    "🏙️ 서울 구별 모델",
    "📍 전국 시군구별 모델",
    "🗺️ 이상치 지도 (시계열)",
])

# ─────────────────────────────────────────────────────────────────────────────
# 탭 1: 전국 단일 모델
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not _cache_ready(ANOMALY_FILES):
        _no_cache_msg()
    else:
        summary_a, top30_a, region_cnt_a, normal_ppy_a, anomaly_ppy_a, scores_a = load_anomaly_cache()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI 카드
        section_badge("📊", "전국 기준 이상치 탐지 요약")
        c1, c2, c3, c4 = st.columns(4)
        _metric_card(c1, "전체 거래 건수",      f"{summary_a['total_count']:,}건",        "분석 대상")
        _metric_card(c2, "특이 거래 건수",      f"{summary_a['anomaly_count']:,}건",       f"전체의 {summary_a['anomaly_ratio']*100:.1f}%")
        _metric_card(c3, "정상거래 평균 평당가", f"{summary_a['avg_ppy_normal']:,.0f}만원",  "평당가 기준")
        _metric_card(c4, "특이거래 평균 평당가", f"{summary_a['avg_ppy_anomaly']:,.0f}만원", "평당가 기준")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TOP 30 테이블
        section_badge("🏆", "전국 기준 특이거래 TOP 30", color="#DC2626")
        display_cols = [c for c in [
            "시군구", "아파트", "거래금액", "전용면적", "층",
            "평당가", "anomaly_score", "anomaly_rank"
        ] if c in top30_a.columns]
        display_top30 = top30_a[display_cols].copy()
        for col in ["거래금액", "평당가"]:
            display_top30 = _fmt_price(display_top30, col)
        display_top30 = _fmt_score(display_top30)
        st.dataframe(display_top30, use_container_width=True, height=420)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 지역별 특이거래 건수 TOP 20
        section_badge("🗺️", "지역별 특이거래 건수 TOP 20", color="#7C3AED")
        fig_region = go.Figure(go.Bar(
            x=region_cnt_a["특이거래수"],
            y=region_cnt_a["시군구"],
            orientation="h",
            marker_color="#EF4444",
            text=region_cnt_a["특이거래수"].apply(lambda x: f"{x:,}건"),
            textposition="outside",
        ))
        fig_region.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=90, t=20, b=20),
            xaxis=dict(title="특이거래 건수", showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(autorange="reversed"),
            font=dict(size=11),
        )
        st.plotly_chart(fig_region, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 정상 vs 특이거래 평당가 박스플롯
        section_badge("📈", "정상거래 vs 특이거래 평당가 분포", color="#059669")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=normal_ppy_a,  name="정상거래", marker_color="#3B82F6", boxmean="sd"))
        fig_box.add_trace(go.Box(y=anomaly_ppy_a, name="특이거래", marker_color="#EF4444", boxmean="sd"))
        fig_box.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=60, r=20),
            yaxis=dict(title="평당가 (만원)", showgrid=True, gridcolor="#F0F4F8"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
        st.caption("시각화를 위해 각 그룹의 99 퍼센타일 이하 데이터만 표시합니다.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── anomaly_score 분포 히스토그램
        section_badge("📉", "Anomaly Score 분포 (특이거래)", color="#6B7280")
        fig_hist = go.Figure(go.Histogram(
            x=scores_a, nbinsx=50, marker_color="#F97316", opacity=0.8,
        ))
        fig_hist.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=40, l=60, r=20),
            xaxis=dict(title="anomaly_score (낮을수록 더 특이)", showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(title="거래 건수", showgrid=True, gridcolor="#F0F4F8"),
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"특이거래 score 범위: {scores_a.min():.4f} ~ {scores_a.max():.4f}  |  "
            "전국 단일 모델 기준 TOP권: -0.03 이하"
        )

        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 탭 2: 서울 구별 모델
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not _cache_ready(SEOUL_FILES):
        _no_cache_msg()
    else:
        kpi_s, summary_s, top_by_district_s = load_seoul_cache()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI 카드
        section_badge("📊", "서울 구별 이상치 탐지 요약")
        c1, c2, c3, c4 = st.columns(4)
        _metric_card(c1, "서울 총거래 건수",    f"{kpi_s['total_count']:,}건",           "분석 대상")
        _metric_card(c2, "특이거래 건수",        f"{kpi_s['anomaly_count']:,}건",         f"전체의 {kpi_s['anomaly_ratio']*100:.1f}%")
        _metric_card(c3, "정상거래 평균 평당가", f"{kpi_s['avg_ppy_normal']:,.0f}만원",   "구별 모델 기준")
        _metric_card(c4, "특이거래 평균 평당가", f"{kpi_s['avg_ppy_anomaly']:,.0f}만원",  "구별 모델 기준")

        st.markdown("<br>", unsafe_allow_html=True)

        x_gu = "구명" if "구명" in summary_s.columns else summary_s.columns[0]

        # ── 구별 평당가 차이 바차트
        section_badge("💰", "구별 정상거래 vs 특이거래 평당가 차이", color="#059669")
        if "평균평당가_차이" in summary_s.columns:
            ppy_sorted = summary_s.sort_values("평균평당가_차이", ascending=False)
            fig_ppy = go.Figure(go.Bar(
                x=ppy_sorted[x_gu],
                y=ppy_sorted["평균평당가_차이"],
                marker_color="#F97316",
                text=ppy_sorted["평균평당가_차이"].apply(lambda x: f"+{x:,.0f}만원" if x >= 0 else f"{x:,.0f}만원"),
                textposition="outside",
            ))
            fig_ppy.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=80, l=20, r=20),
                yaxis=dict(title="평당가 차이 (만원)", showgrid=True, gridcolor="#F0F4F8"),
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig_ppy, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 구 선택 → TOP 10
        section_badge("🔍", "구별 특이거래 상세 조회", color="#DC2626")
        if "구명" in top_by_district_s.columns:
            gu_list     = sorted(top_by_district_s["구명"].unique().tolist())
            selected_gu = st.selectbox("조회할 구를 선택하세요", gu_list, key="gu_select")
            gu_df       = top_by_district_s[top_by_district_s["구명"] == selected_gu].copy()

            display_cols = [c for c in [
                "구명", "아파트", "거래금액", "전용면적", "층",
                "평당가", "anomaly_score", "anomaly_rank"
            ] if c in gu_df.columns]
            display_gu = gu_df[display_cols].copy()
            for col in ["거래금액", "평당가"]:
                display_gu = _fmt_price(display_gu, col)
            display_gu = _fmt_score(display_gu)

            st.caption(f"{selected_gu} 특이거래 TOP 10 (구 내 anomaly_score 기준)")
            st.dataframe(display_gu.reset_index(drop=True), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 구별 최강 이상치 score
        section_badge("📉", "구별 최강 이상치 Score 비교", color="#6B7280")
        if "특이거래_최저score" in summary_s.columns:
            score_sorted = summary_s.sort_values("특이거래_최저score")
            fig_score = go.Figure(go.Bar(
                x=score_sorted["특이거래_최저score"],
                y=score_sorted[x_gu],
                orientation="h",
                marker_color="#EF4444",
                text=score_sorted["특이거래_최저score"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
            ))
            fig_score.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=90, t=20, b=20),
                xaxis=dict(title="최저 anomaly_score (낮을수록 더 특이)", showgrid=True, gridcolor="#F0F4F8"),
                yaxis=dict(autorange="reversed"),
                font=dict(size=11),
            )
            st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})
            st.caption("서울 구별 모델 score 범위: -0.08 ~ -0.16 수준 (전국 단일 모델과 직접 비교 불가)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 서울 구별 요약 테이블
        section_badge("📋", "서울 구별 특이거래 요약 테이블", color="#6B7280")
        display_summary_s = summary_s.copy()
        for col in ["정상거래_평균평당가", "특이거래_평균평당가", "평균평당가_차이", "특이거래_평균거래금액"]:
            if col in display_summary_s.columns:
                display_summary_s[col] = display_summary_s[col].apply(
                    lambda x: f"{x:,.0f}만원" if pd.notna(x) else "-"
                )
        st.dataframe(display_summary_s, use_container_width=True, height=460)

        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 탭 3: 전국 시군구별 모델
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not _cache_ready(LOCATION_FILES):
        _no_cache_msg()
    else:
        kpi_l, top10_l, summary_l, anomalies_l = load_location_cache()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI 카드
        section_badge("📊", "전국 시군구별 이상치 탐지 요약")
        best_score = kpi_l.get("best_score")
        c1, c2, c3, c4 = st.columns(4)
        _metric_card(c1, "전체 거래 건수",   f"{kpi_l['total_count']:,}건",                             "분석 대상")
        _metric_card(c2, "특이거래 건수",     f"{kpi_l['anomaly_count']:,}건",                           f"탐지 대상의 {kpi_l['anomaly_count']/kpi_l['detected_count']*100:.1f}%")
        _metric_card(c3, "분석 시군구 수",    f"{kpi_l['location_count']}개",                            "246개 중 (30건 미만 제외)")
        _metric_card(c4, "최강 이상치 Score", f"{best_score:.4f}" if best_score is not None else "-",    "값이 작을수록 더 특이")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TOP 10 테이블
        section_badge("🏆", "시군구별 TOP 1 이상치 → 전국 TOP 10", color="#DC2626")
        display_cols = [c for c in [
            "시군구", "아파트", "거래금액", "전용면적", "층",
            "평당가", "anomaly_score", "anomaly_rank", "total_rank"
        ] if c in top10_l.columns]
        display_top10 = top10_l[display_cols].copy()
        for col in ["거래금액", "평당가"]:
            display_top10 = _fmt_price(display_top10, col)
        display_top10 = _fmt_score(display_top10)
        st.dataframe(display_top10.reset_index(drop=True), use_container_width=True)

        # ── TOP 10 바차트
        if len(top10_l) > 0 and "anomaly_score" in top10_l.columns:
            label_col = "아파트" if "아파트" in top10_l.columns else "시군구"
            labels = top10_l.apply(
                lambda r: f"{r.get('시군구','')}  {r.get(label_col,'')}", axis=1
            )
            fig_top10 = go.Figure(go.Bar(
                x=top10_l["anomaly_score"],
                y=labels,
                orientation="h",
                marker_color="#EF4444",
                text=top10_l["anomaly_score"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
            ))
            fig_top10.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=90, t=20, b=20),
                xaxis=dict(title="anomaly_score (낮을수록 더 특이)", showgrid=True, gridcolor="#F0F4F8"),
                yaxis=dict(autorange="reversed"),
                font=dict(size=11),
            )
            st.plotly_chart(fig_top10, use_container_width=True, config={"displayModeBar": False})
            st.caption("전국 시군구별 모델 score 범위: -0.13 ~ -0.16 수준 (다른 모델 score와 직접 비교 불가)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 시군구별 특이거래 비율 TOP 20
        section_badge("🗺️", "시군구별 특이거래 비율 TOP 20", color="#7C3AED")
        if "특이거래비율(%)" in summary_l.columns:
            top20_ratio = summary_l.nlargest(20, "특이거래비율(%)")
            fig_ratio = go.Figure(go.Bar(
                x=top20_ratio["특이거래비율(%)"],
                y=top20_ratio["시군구"],
                orientation="h",
                marker_color="#8B5CF6",
                text=top20_ratio["특이거래비율(%)"].apply(lambda x: f"{x:.2f}%"),
                textposition="outside",
            ))
            fig_ratio.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=80, t=20, b=20),
                xaxis=dict(title="특이거래 비율 (%)", showgrid=True, gridcolor="#F0F4F8"),
                yaxis=dict(autorange="reversed"),
                font=dict(size=11),
            )
            st.plotly_chart(fig_ratio, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 시군구 선택 → 상세 조회
        section_badge("🔍", "시군구별 특이거래 상세 조회", color="#059669")
        if "시군구" in anomalies_l.columns:
            sigungu_list = sorted(anomalies_l["시군구"].dropna().unique().tolist())
            selected_sg  = st.selectbox("조회할 시군구를 선택하세요", sigungu_list, key="sg_select")

            sg_result = (
                anomalies_l[anomalies_l["시군구"] == selected_sg]
                .sort_values("anomaly_score")
                .head(20)
                .copy()
            )

            if sg_result.empty:
                st.info(f"{selected_sg}에서 탐지된 특이거래가 없거나 학습 데이터가 부족합니다.")
            else:
                display_cols = [c for c in [
                    "시군구", "아파트", "거래금액", "전용면적", "층",
                    "평당가", "anomaly_score", "anomaly_rank"
                ] if c in sg_result.columns]
                display_sg = sg_result[display_cols].copy()
                for col in ["거래금액", "평당가"]:
                    display_sg = _fmt_price(display_sg, col)
                display_sg = _fmt_score(display_sg)
                st.caption(f"{selected_sg} 특이거래 TOP 20 (시군구 내 anomaly_score 기준)")
                st.dataframe(display_sg.reset_index(drop=True), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 전국 시군구별 요약 테이블
        section_badge("📋", "전국 시군구별 특이거래 요약 테이블", color="#6B7280")
        display_summary_l = summary_l.copy()
        for col in ["정상거래_평균평당가", "특이거래_평균평당가", "평균평당가_차이", "특이거래_평균거래금액"]:
            if col in display_summary_l.columns:
                display_summary_l[col] = display_summary_l[col].apply(
                    lambda x: f"{x:,.0f}만원" if pd.notna(x) else "-"
                )
        st.dataframe(display_summary_l, use_container_width=True, height=500)

        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 탭 4: 이상치 지도 (시계열 슬라이더)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    if not _cache_ready(LOCATION_MAP_FILES):
        _no_cache_msg()
    else:
        map_df = load_location_map_cache()

        st.markdown("<br>", unsafe_allow_html=True)

        section_badge("🗺️", "기간별 특이거래 분포 지도")
        st.caption(
            "슬라이더로 기간을 옮기면 그 기간에 발생한 특이거래만 지도에 표시됩니다. "
            "점 색이 진할수록 시군구 내부 기준으로 더 특이한 거래입니다."
        )

        period_options = sorted(map_df["연월"].unique().tolist())

        preset_col, slider_col = st.columns([1, 4])
        with preset_col:
            st.markdown("<br>", unsafe_allow_html=True)
            surge_preset = st.button("📈 2020~2021 급등기만 보기", use_container_width=True)

        if "anomaly_map_range" not in st.session_state:
            st.session_state.anomaly_map_range = (period_options[0], period_options[-1])

        if surge_preset:
            surge_start = next((p for p in period_options if p >= "2020-01"), period_options[0])
            surge_end   = next((p for p in reversed(period_options) if p <= "2021-12"), period_options[-1])
            st.session_state.anomaly_map_range = (surge_start, surge_end)

        with slider_col:
            start_period, end_period = st.select_slider(
                "기간 선택 (거래연도-월)",
                options=period_options,
                value=st.session_state.anomaly_map_range,
                key="anomaly_map_range",
            )

        period_df = map_df[(map_df["연월"] >= start_period) & (map_df["연월"] <= end_period)]

        c1, c2, c3 = st.columns(3)
        _metric_card(c1, "선택 기간", f"{start_period} ~ {end_period}", "거래연도-월 기준")
        _metric_card(c2, "표시 특이거래 건수", f"{len(period_df):,}건", f"전체 {len(map_df):,}건 중")
        _metric_card(
            c3, "최강 이상치 Score",
            f"{period_df['anomaly_score'].min():.4f}" if len(period_df) else "-",
            "값이 작을수록 더 특이",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if period_df.empty:
            st.info("선택한 기간에는 탐지된 특이거래가 없습니다.")
        else:
            MAX_MARKERS = 3000
            plotted = period_df
            if len(period_df) > MAX_MARKERS:
                plotted = period_df.sort_values("anomaly_score", ascending=True).head(MAX_MARKERS)
                st.caption(f"⚠ 표시 성능을 위해 가장 특이한 {MAX_MARKERS:,}건만 지도에 표시합니다.")

            vmin = float(plotted["anomaly_score"].min())
            vmax = float(plotted["anomaly_score"].max())

            m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles="cartodbpositron")
            cluster = MarkerCluster().add_to(m)

            for row in plotted.itertuples():
                folium.CircleMarker(
                    location=[row.위도, row.경도],
                    radius=5,
                    color=_anomaly_color(row.anomaly_score, vmin, vmax),
                    fill=True,
                    fill_color=_anomaly_color(row.anomaly_score, vmin, vmax),
                    fill_opacity=0.85,
                    weight=1,
                    popup=folium.Popup(
                        f"<b>{row.시군구} · {row.아파트}</b><br>"
                        f"거래연월: {row.연월}<br>"
                        f"거래금액: {row.거래금액:,.0f}만원<br>"
                        f"평당가: {row.평당가:,.0f}만원<br>"
                        f"anomaly_score: {row.anomaly_score:.4f}",
                        max_width=260,
                    ),
                ).add_to(cluster)

            st_folium(m, width="100%", height=620, returned_objects=[], key=f"anomaly_map_{start_period}_{end_period}")

        st.markdown("<br>", unsafe_allow_html=True)
 