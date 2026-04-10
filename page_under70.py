# ──────────────────────────────────────────────────────────────
#  page_under70.py
#  70점 미만 QA 모니터링 교차분석 페이지
#  Google Sheets gid=2055211445 에서 로드
# ──────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials

# ── 색상 토큰 (main.py 동일) ─────────────────────────────────
C_PRIMARY = "#6366f1"
C_SUCCESS = "#22c55e"
C_WARNING = "#f59e0b"
C_DANGER  = "#ef4444"
C_BG_CARD = "#ffffff"
C_BORDER  = "#e2e8f0"
C_TEXT    = "#1e293b"
C_TEXT_SUB = "#64748b"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Google Sheets 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHEET_ID = "1ujtxIKZJRR9vIC1TS5GWWEtM9luChlDJk4NwTqeYB2Q"
GID      = 2055211445          # 두 번째 시트

@st.cache_data(ttl=600, show_spinner="모니터링 시트 로딩 중…")
def load_monitoring_sheet():
    """gid=2055211445 시트를 DataFrame으로 반환"""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        # gid 로 워크시트 찾기
        ws = None
        for w in sh.worksheets():
            if w.id == GID:
                ws = w
                break
        if ws is None:
            ws = sh.get_worksheet(1)   # fallback: 두 번째 시트
        rows = ws.get_all_values()
    except Exception as e:
        st.error(f"시트 로드 실패: {e}")
        return pd.DataFrame()

    if len(rows) < 7:
        st.error("시트에 데이터가 부족합니다.")
        return pd.DataFrame()

    # ── 헤더 행 스킵 & 데이터 시작점 탐색 ──
    # 첫 번째 컬럼이 숫자(회신월)인 행이 데이터 시작
    data_start = None
    for i, row in enumerate(rows):
        val = str(row[0]).strip()
        if val.isdigit() and int(val) > 0:
            data_start = i
            break
    if data_start is None:
        st.error("데이터 시작 행을 찾을 수 없습니다.")
        return pd.DataFrame()

    data_rows = rows[data_start:]

    # ── 컬럼 수 확인 후 매핑 (위치 기반) ──
    ncols = len(data_rows[0]) if data_rows else 0
    # 필요한 핵심 컬럼만 위치 기반으로 잡기
    # 엑셀 기준 컬럼 인덱스 (0-based):
    COL_IDX = {
        "회신월":     0,
        "발송월":     1,
        "회신주차":   2,
        "발송주차":   3,
        "발송일자":   4,
        "회신일자":   5,
        "사업자":     6,
        "브랜드":     7,
        "채널":       8,
        "상담사":     9,
        "입사일":     10,
        "상담사근속": 11,
        "상담유형대": 12,
        "상담유형중": 13,
        "상담유형소": 14,
        "키워드":     15,
        "긍정부정":   16,
        "유형":       17,
        "총합":       18,
        "Q1":         19,
        "Q2":         20,
        "Q3":         21,
        "친절점수":   22,
        "만족점수":   23,
        "최종점수":   24,
        "만족율":     25,
        "WK":         26,
        "상담이력KEY": 27,
        # ── QA 모니터링 ──
        "문의유형":     28,
        "귀책분류":     29,
        "문의불만사유": 30,
        # 정확성(30) 서브항목
        "정확한안내":   31,   # (10)
        "프로세스":     32,   # (10)
        "전산처리":     33,   # (10)
        # 숙련도(20) 서브항목
        "맞춤설명":     34,   # (10)
        "문의파악":     35,   # (5)
        "숙련도_채널":  36,   # Call_음성숙련도 / Chat_대기 (5)
        # 친절도(30) 서브항목
        "친절도_감정":  37,   # Call_전반적인감정연출 / Chat_양해 (10)
        "친절도_경청":  38,   # Call_경청 / Chat_즉각호응 (15)
        "언어표현":     39,   # (5)
        # 약속이행(20) 서브항목
        "약속불이행":   40,   # (10)
        "약속지연이행": 41,   # (5)
        "약속시간누락": 42,   # (5)
        # 이행점수
        "이행점수":     43,
        # 상세분석
        "상세분석":     44,
        "피드백여부":   45,
        "주문번호":     46,
    }

    # 컬럼 추출
    parsed = []
    for row in data_rows:
        r = {}
        for col_name, idx in COL_IDX.items():
            if idx < len(row):
                val = str(row[idx]).strip()
                r[col_name] = val if val and val.lower() != "nan" else ""
            else:
                r[col_name] = ""
        parsed.append(r)

    df = pd.DataFrame(parsed)

    # ── 빈 행 제거 (상담이력KEY 기준 — 머지셀 빈 행) ──
    df = df[df["상담이력KEY"].str.strip() != ""].copy()
    df = df[df["최종점수"].str.strip() != ""].copy()

    # ── 숫자 변환 ──
    for c in ["최종점수", "친절점수", "만족점수", "이행점수",
              "회신월", "발송월", "회신주차", "발송주차", "WK"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QA 항목별 차감 점수 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 텍스트가 있으면 차감, 없으면 0
QA_SUB_ITEMS = {
    "정확성(30)": {
        "정확한안내": 10,
        "프로세스":   10,
        "전산처리":   10,
    },
    "숙련도(20)": {
        "맞춤설명":    10,
        "문의파악":     5,
        "숙련도_채널":  5,
    },
    "친절도(30)": {
        "친절도_감정": 10,
        "친절도_경청": 15,
        "언어표현":     5,
    },
    "약속이행(20)": {
        "약속불이행":   10,
        "약속지연이행":  5,
        "약속시간누락":  5,
    },
}

def calc_deductions(df: pd.DataFrame) -> pd.DataFrame:
    """각 서브항목에 텍스트 있으면 해당 배점만큼 차감, 대분류별 차감합, 이행총점 계산"""
    df = df.copy()

    for category, items in QA_SUB_ITEMS.items():
        cat_deduct = f"{category}_차감"
        df[cat_deduct] = 0
        for col, score in items.items():
            flag_col = f"{col}_감점"
            df[flag_col] = df[col].apply(lambda x: 1 if str(x).strip() != "" else 0)
            df[cat_deduct] = df[cat_deduct] + df[flag_col] * score

    df["QA총차감"] = (
        df["정확성(30)_차감"] +
        df["숙련도(20)_차감"] +
        df["친절도(30)_차감"] +
        df["약속이행(20)_차감"]
    )
    df["QA이행점수"] = 100 - df["QA총차감"]
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  페이지 렌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_under70():
    st.markdown("## 🔍 70점 미만 QA 모니터링 교차분석")
    st.caption("CSAT 최종점수 70점 미만 건의 모니터링 이행평가 · 귀책 · 항목별 차감 현황")

    df_raw = load_monitoring_sheet()
    if df_raw.empty:
        st.warning("데이터를 불러올 수 없습니다.")
        return

    df = calc_deductions(df_raw)

    # ── 필터 ──────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        months = sorted(df["회신월"].dropna().unique())
        sel_months = st.multiselect("회신월", months, default=months)
    with col_f2:
        agents = sorted(df["상담사"].unique())
        sel_agents = st.multiselect("상담사", agents, default=agents)
    with col_f3:
        channels = sorted(df["채널"].unique())
        sel_channels = st.multiselect("채널", channels, default=channels)

    mask = (
        df["회신월"].isin(sel_months) &
        df["상담사"].isin(sel_agents) &
        df["채널"].isin(sel_channels)
    )
    df_f = df[mask].copy()

    if df_f.empty:
        st.info("필터 조건에 맞는 데이터가 없습니다.")
        return

    # ── 1. 전체 KPI ──────────────────────────────────────────
    st.markdown("### 📊 전체 현황")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("모니터링 건수", f"{len(df_f)}건")
    k2.metric("평균 CSAT 최종점수", f"{df_f['최종점수'].mean():.1f}")
    k3.metric("평균 QA 이행점수", f"{df_f['QA이행점수'].mean():.1f}")
    k4.metric("평균 QA 차감점수", f"{df_f['QA총차감'].mean():.1f}")

    # ── 2. 귀책 분류 비중 ────────────────────────────────────
    st.markdown("### 📌 귀책 분류 현황")
    tab_all, tab_ch = st.tabs(["전체", "채널별"])

    with tab_all:
        blame = df_f["귀책분류"].value_counts().reset_index()
        blame.columns = ["귀책분류", "건수"]
        blame["비율"] = (blame["건수"] / blame["건수"].sum() * 100).round(1)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(blame, names="귀책분류", values="건수",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="귀책 분류 비중")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(blame, use_container_width=True, hide_index=True)

    with tab_ch:
        blame_ch = (
            df_f.groupby(["채널", "귀책분류"])
            .size()
            .reset_index(name="건수")
        )
        fig2 = px.bar(blame_ch, x="채널", y="건수", color="귀책분류",
                      barmode="group", title="채널별 귀책 분류",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ── 3. 불만사유 Top 10 ───────────────────────────────────
    st.markdown("### 💢 문의불만사유 Top 10")
    reason = df_f[df_f["문의불만사유"] != ""]["문의불만사유"].value_counts().head(10).reset_index()
    reason.columns = ["불만사유", "건수"]
    fig3 = px.bar(reason, x="건수", y="불만사유", orientation="h",
                  color="건수", color_continuous_scale="Reds",
                  title="문의불만사유 빈도")
    fig3.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

    # ── 4. QA 항목별 차감 분석 ───────────────────────────────
    st.markdown("### 📋 QA 항목별 차감 현황")

    tab_cat, tab_sub = st.tabs(["대분류별", "세부항목별"])

    with tab_cat:
        cat_cols = ["정확성(30)_차감", "숙련도(20)_차감", "친절도(30)_차감", "약속이행(20)_차감"]
        cat_means = df_f[cat_cols].mean().reset_index()
        cat_means.columns = ["항목", "평균차감"]
        cat_means["항목"] = cat_means["항목"].str.replace("_차감", "")
        fig4 = px.bar(cat_means, x="항목", y="평균차감",
                      color="평균차감", color_continuous_scale="OrRd",
                      title="대분류별 평균 차감 점수")
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

    with tab_sub:
        sub_flags = [c for c in df_f.columns if c.endswith("_감점")]
        sub_counts = df_f[sub_flags].sum().reset_index()
        sub_counts.columns = ["항목", "차감건수"]
        sub_counts["항목"] = sub_counts["항목"].str.replace("_감점", "")
        sub_counts = sub_counts.sort_values("차감건수", ascending=False)
        fig5 = px.bar(sub_counts, x="차감건수", y="항목", orientation="h",
                      color="차감건수", color_continuous_scale="YlOrRd",
                      title="세부항목 차감 빈도")
        fig5.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, use_container_width=True)

    # ── 5. 상담사별 성과 ─────────────────────────────────────
    st.markdown("### 👤 상담사별 QA 성과")

    agent_agg = (
        df_f.groupby("상담사")
        .agg(
            건수=("상담이력KEY", "count"),
            평균_CSAT=("최종점수", "mean"),
            평균_QA이행=("QA이행점수", "mean"),
            평균_차감=("QA총차감", "mean"),
            정확성차감=("정확성(30)_차감", "mean"),
            숙련도차감=("숙련도(20)_차감", "mean"),
            친절도차감=("친절도(30)_차감", "mean"),
            약속이행차감=("약속이행(20)_차감", "mean"),
        )
        .round(1)
        .sort_values("평균_QA이행")
        .reset_index()
    )
    st.dataframe(agent_agg, use_container_width=True, hide_index=True)

    # 상담사별 레이더 차트 (선택형)
    sel_agent = st.selectbox("상담사 상세 보기", agent_agg["상담사"].tolist())
    if sel_agent:
        row = agent_agg[agent_agg["상담사"] == sel_agent].iloc[0]
        cats = ["정확성", "숙련도", "친절도", "약속이행"]
        vals = [row["정확성차감"], row["숙련도차감"], row["친절도차감"], row["약속이행차감"]]
        max_vals = [30, 20, 30, 20]
        earned = [m - v for m, v in zip(max_vals, vals)]

        fig6 = go.Figure()
        fig6.add_trace(go.Scatterpolar(
            r=earned + [earned[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=sel_agent,
            line_color=C_PRIMARY,
        ))
        fig6.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
            title=f"{sel_agent} — QA 항목별 획득 점수",
            height=400,
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── 6. 귀책 × CSAT 교차 ─────────────────────────────────
    st.markdown("### 🔀 귀책 × CSAT 점수 교차분석")
    cross = (
        df_f.groupby("귀책분류")
        .agg(건수=("상담이력KEY", "count"),
             평균_CSAT=("최종점수", "mean"),
             평균_QA이행=("QA이행점수", "mean"))
        .round(1)
        .sort_values("건수", ascending=False)
        .reset_index()
    )
    st.dataframe(cross, use_container_width=True, hide_index=True)

    fig7 = px.scatter(
        df_f, x="최종점수", y="QA이행점수",
        color="귀책분류", hover_data=["상담사", "채널", "문의불만사유"],
        title="CSAT 최종점수 vs QA 이행점수 (귀책별)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig7.update_layout(height=450)
    st.plotly_chart(fig7, use_container_width=True)

    # ── 7. 귀책별 월 추이 ────────────────────────────────────
    st.markdown("### 📈 월별 귀책 건수 추이")
    monthly_blame = (
        df_f.groupby(["회신월", "귀책분류"])
        .size()
        .reset_index(name="건수")
    )
    fig8 = px.line(monthly_blame, x="회신월", y="건수", color="귀책분류",
                   markers=True, title="월별 귀책 분류 추이",
                   color_discrete_sequence=px.colors.qualitative.Set2)
    fig8.update_layout(height=400)
    st.plotly_chart(fig8, use_container_width=True)

    # ── 8. 상세 데이터 테이블 ────────────────────────────────
    st.markdown("### 📄 상세 데이터")
    show_cols = [
        "회신월", "발송일자", "상담사", "채널", "사업자", "브랜드",
        "상담유형대", "최종점수", "친절점수", "만족점수",
        "귀책분류", "문의불만사유", "QA이행점수", "QA총차감",
        "정확성(30)_차감", "숙련도(20)_차감", "친절도(30)_차감", "약속이행(20)_차감",
        "상세분석", "피드백여부",
    ]
    existing = [c for c in show_cols if c in df_f.columns]
    st.dataframe(
        df_f[existing].sort_values("QA이행점수"),
        use_container_width=True,
        hide_index=True,
        height=500,
    )
