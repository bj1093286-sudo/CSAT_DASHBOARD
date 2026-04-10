# ──────────────────────────────────────────────────────────────
# page_under70.py
# 같은 레포에 두고 main.py 에서 import + 메뉴 1줄 추가
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── 색상 토큰 (main.py 와 통일) ─────────────────────────────
C_PRIMARY = "#6366f1"
C_SUCCESS = "#22c55e"
C_WARNING = "#f59e0b"
C_DANGER  = "#ef4444"
C_BG_CARD = "#ffffff"
C_BORDER  = "#e2e8f0"
C_TEXT    = "#1e293b"
C_TEXT_SUB = "#64748b"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 채널별 감점 기준표
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 항목명 → (대분류, 배점)
ITEM_META = {
    "정확한안내":   ("정확성", 10),
    "프로세스":     ("정확성", 10),
    "전산처리":     ("정확성", 10),
    "맞춤설명":     ("숙련도", 10),
    "문의파악":     ("숙련도",  5),
    "숙련도채널":   ("숙련도",  5),
    "감정연출양해": ("친절도", 10),
    "경청호응":     ("친절도", 15),
    "언어표현":     ("친절도",  5),
    "약속불이행":   ("약속이행", 10),
    "약속지연이행": ("약속이행",  5),
    "시간안내누락": ("약속이행",  5),
}

CATEGORY_FULL = {"정확성": 30, "숙련도": 20, "친절도": 30, "약속이행": 20}
TOTAL_FULL = 100

# 채널별 유효 감점 텍스트 목록
DEDUCT_PHONE = {
    "정확한안내":   ["오안내", "일부답변 누락", "임의안내"],
    "프로세스":     ["필수안내 사항 누락", "프로세스 미준수"],
    "전산처리":     ["오처리", "전산처리 누락"],
    "맞춤설명":     ["설명 미흡"],
    "문의파악":     ["추가탐색 누락", "니즈 파악"],
    "숙련도채널":   ["상담흐름 끊김", "발음 부정확/오탈자", "응대속도 미흡", "대기 문제"],
    "감정연출양해": ["사무적인 응대", "상황에 맞지 않는 응대", "분위기 반전",
                     "사과/양해멘트 누락/부족"],
    "경청호응":     ["일방적인 응대", "성급한 개입/말자름", "호응 누락"],
    "언어표현":     ["불손한 어투", "어벽/사족어/내부용어 등"],
    "약속불이행":   ["약속 불이행"],
    "약속지연이행": ["약속 지연이행"],
    "시간안내누락": ["약속시간 안내 누락"],
}

DEDUCT_CHAT = {
    "정확한안내":   ["오안내", "일부답변 누락", "임의안내"],
    "프로세스":     ["필수안내 사항 누락", "프로세스 미준수"],
    "전산처리":     ["오처리", "전산처리 누락"],
    "맞춤설명":     ["설명 미흡"],
    "문의파악":     ["추가탐색 누락", "니즈 파악"],
    "숙련도채널":   ["답변 지연"],
    "감정연출양해": ["사과, 양해멘트 누락/부족", "사과/양해멘트 누락/부족", "쿠션어 누락"],
    "경청호응":     ["호응어 누락"],
    "언어표현":     ["사무적 / 기계적 응대", "내부/축양 용어", "띄어쓰기/맞춤법 오류"],
    "약속불이행":   ["약속 불이행"],
    "약속지연이행": ["약속 지연이행"],
    "시간안내누락": ["약속시간 안내 누락"],
}

ITEMS_ORDER = [
    "정확한안내", "프로세스", "전산처리",
    "맞춤설명", "문의파악", "숙련도채널",
    "감정연출양해", "경청호응", "언어표현",
    "약속불이행", "약속지연이행", "시간안내누락",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Google Sheets 에서 두 번째 시트 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QA_SHEET_NAME = "모니터링_QA"

# 시트 컬럼명 → 코드 내부 키 매핑
QA_COL_MAP = {
    "상담이력KEY": "상담이력KEY",
    "상담이력 KEY": "상담이력KEY",
    "문의유형":     "문의유형",
    "문의 유형":    "문의유형",
    "귀책분류":     "귀책분류",
    "귀책 분류":    "귀책분류",
    "문의불만사유": "문의불만사유",
    "문의 불만사유": "문의불만사유",
    "정확한안내":   "정확한안내",
    "프로세스":     "프로세스",
    "전산처리":     "전산처리",
    "맞춤설명":     "맞춤설명",
    "문의파악":     "문의파악",
    "숙련도채널":   "숙련도채널",
    "감정연출양해": "감정연출양해",
    "경청호응":     "경청호응",
    "언어표현":     "언어표현",
    "약속불이행":   "약속불이행",
    "약속지연이행": "약속지연이행",
    "시간안내누락": "시간안내누락",
    "상세분석":     "상세분석",
    "피드백여부":   "피드백여부",
    "피드백 여부":  "피드백여부",
}


@st.cache_data(ttl=600, show_spinner="모니터링 QA 시트 로드 중…")
def load_qa_sheet(spreadsheet_id: str) -> pd.DataFrame:
    """Google Sheets 의 두 번째 탭(모니터링_QA)을 CSV export 로 읽기"""
    import urllib.parse
    encoded_name = urllib.parse.quote(QA_SHEET_NAME)
    url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={encoded_name}"
    )
    try:
        df = pd.read_csv(url, dtype=str)
    except Exception as e:
        st.warning(f"모니터링_QA 시트를 읽을 수 없습니다: {e}")
        return pd.DataFrame()

    # 컬럼 정규화
    rename = {}
    for c in df.columns:
        key = c.strip().replace("\n", "").replace(" ", "")
        for src, dst in QA_COL_MAP.items():
            if key == src.replace(" ", ""):
                rename[c] = dst
                break
    df.rename(columns=rename, inplace=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 점수 계산 로직
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _has_text(val) -> bool:
    """셀에 감점 텍스트가 있는지 판별"""
    if pd.isna(val):
        return False
    s = str(val).strip()
    if s == "" or s.lower() == "nan" or s == "-":
        return False
    return True


def calc_qa_scores(df_qa: pd.DataFrame, df_csat: pd.DataFrame) -> pd.DataFrame:
    """
    QA 시트 + CSAT 시트를 조인하고
    채널 정보 기반으로 감점 계산 → 항목별 득점 / 대분류 득점 / 총점 산출
    """
    if df_qa.empty:
        return pd.DataFrame()

    # CSAT 에서 필요한 컬럼만 가져오기
    csat_cols = []
    for c in ["상담이력KEY", "채널", "사업자", "브랜드", "상담사",
              "상담사근속", "친절점수", "만족점수", "최종점수",
              "발송일자", "회신월", "발송월", "회신주차", "WK",
              "긍정부정", "유형", "키워드", "Q3",
              "상담유형대", "상담유형중", "상담유형소"]:
        if c in df_csat.columns:
            csat_cols.append(c)

    df_csat_slim = df_csat[csat_cols].copy()

    # 상담이력KEY 타입 통일
    for d in [df_qa, df_csat_slim]:
        if "상담이력KEY" in d.columns:
            d["상담이력KEY"] = d["상담이력KEY"].astype(str).str.strip()

    # 조인
    merged = df_qa.merge(df_csat_slim, on="상담이력KEY", how="left")

    # 채널 정규화
    if "채널" in merged.columns:
        merged["채널구분"] = merged["채널"].apply(
            lambda x: "채팅" if "채팅" in str(x) or "chat" in str(x).lower()
            else "전화"
        )
    else:
        merged["채널구분"] = "전화"

    # ── 항목별 득점 계산 ──
    for item in ITEMS_ORDER:
        _, full_score = ITEM_META[item]
        col_score = f"{item}_점수"

        if item in merged.columns:
            merged[col_score] = merged.apply(
                lambda row: 0 if _has_text(row.get(item)) else full_score,
                axis=1,
            )
        else:
            merged[col_score] = full_score

    # ── 대분류 합산 ──
    for cat, full in CATEGORY_FULL.items():
        sub_items = [it for it, (c, _) in ITEM_META.items() if c == cat]
        score_cols = [f"{it}_점수" for it in sub_items]
        existing = [c for c in score_cols if c in merged.columns]
        if existing:
            merged[f"{cat}_득점"] = merged[existing].astype(float).sum(axis=1)
        else:
            merged[f"{cat}_득점"] = float(full)

    # 이행총점 계산
    cat_score_cols = [f"{cat}_득점" for cat in CATEGORY_FULL]
    merged["이행총점_계산"] = merged[cat_score_cols].astype(float).sum(axis=1)

    # 날짜 파싱
    if "발송일자" in merged.columns:
        merged["발송일자_dt"] = pd.to_datetime(merged["발송일자"], errors="coerce")
        merged["발송월_parsed"] = merged["발송일자_dt"].dt.to_period("M").astype(str)
        merged["발송주_parsed"] = (
            merged["발송일자_dt"].dt.isocalendar().year.astype(str)
            + "-W"
            + merged["발송일자_dt"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

    # 최종점수 숫자 변환
    if "최종점수" in merged.columns:
        merged["최종점수"] = pd.to_numeric(merged["최종점수"], errors="coerce")

    return merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. UI 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _kpi_card(label, value, sub="", color=C_PRIMARY):
    st.markdown(f"""
    <div style="background:{C_BG_CARD};border:1px solid {C_BORDER};
                border-radius:12px;padding:20px;text-align:center;
                border-top:4px solid {color};">
        <div style="color:{C_TEXT_SUB};font-size:13px;">{label}</div>
        <div style="color:{color};font-size:28px;font-weight:700;
                    margin:4px 0;">{value}</div>
        <div style="color:{C_TEXT_SUB};font-size:12px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def _section(title):
    st.markdown(f"""
    <div style="border-left:4px solid {C_PRIMARY};padding-left:12px;
                margin:28px 0 16px 0;">
        <span style="font-size:18px;font-weight:700;color:{C_TEXT};">
            {title}</span>
    </div>
    """, unsafe_allow_html=True)


def _color_score(val, full):
    """점수 → 색상 배지 HTML"""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return str(val)
    if v >= full:
        bg, fg = "#dcfce7", "#166534"
    elif v >= full * 0.5:
        bg, fg = "#fef9c3", "#854d0e"
    else:
        bg, fg = "#fee2e2", "#991b1b"
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:6px;font-weight:600;">{v:.0f}</span>')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 메인 페이지 함수 (main.py 에서 호출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_under70_analysis(df_csat: pd.DataFrame, spreadsheet_id: str):
    """
    70점 미만 모니터링 교차분석 페이지
    Parameters
    ----------
    df_csat : 기존 CSAT DataFrame (prepare_data_from_df 에서 나온 것)
    spreadsheet_id : Google Sheets 문서 ID
    """

    st.markdown(
        f'<h2 style="color:{C_TEXT};">📋 70점 미만 모니터링 · 이행 교차분석</h2>',
        unsafe_allow_html=True,
    )

    # ── QA 시트 로드 ──
    df_qa = load_qa_sheet(spreadsheet_id)
    if df_qa.empty:
        st.info(
            f"Google Sheets 에 **'{QA_SHEET_NAME}'** 탭이 없거나 비어 있습니다.\n\n"
            "같은 스프레드시트에 두 번째 시트를 만들고 데이터를 넣어주세요.\n\n"
            "**필수 컬럼:** 상담이력KEY, 귀책분류, 정확한안내, 프로세스, "
            "전산처리, 맞춤설명, 문의파악, 숙련도채널, 감정연출양해, "
            "경청호응, 언어표현, 약속불이행, 약속지연이행, 시간안내누락, "
            "상세분석, 피드백여부"
        )
        return

    # ── 점수 계산 ──
    df = calc_qa_scores(df_qa, df_csat)
    if df.empty:
        st.warning("조인 결과가 비어 있습니다. 상담이력KEY 를 확인해주세요.")
        return

    # ── 70점 미만 필터 ──
    df_under = df[df["최종점수"] < 70].copy() if "최종점수" in df.columns else df.copy()

    # ━━━━━━━━━━━ 필터 사이드바 ━━━━━━━━━━━━
    with st.expander("🔍 필터", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)

        # 기간 필터
        if "발송일자_dt" in df_under.columns:
            min_dt = df_under["발송일자_dt"].dropna().min()
            max_dt = df_under["발송일자_dt"].dropna().max()
            if pd.notna(min_dt) and pd.notna(max_dt):
                with fc1:
                    date_range = st.date_input(
                        "기간", [min_dt, max_dt],
                        min_value=min_dt, max_value=max_dt,
                    )
                if len(date_range) == 2:
                    df_under = df_under[
                        (df_under["발송일자_dt"] >= pd.Timestamp(date_range[0]))
                        & (df_under["발송일자_dt"] <= pd.Timestamp(date_range[1]))
                    ]

        # 채널 필터
        if "채널구분" in df_under.columns:
            with fc2:
                ch_opts = ["전체"] + sorted(df_under["채널구분"].dropna().unique().tolist())
                ch_sel = st.selectbox("채널", ch_opts)
            if ch_sel != "전체":
                df_under = df_under[df_under["채널구분"] == ch_sel]

        # 귀책 필터
        if "귀책분류" in df_under.columns:
            with fc3:
                g_opts = ["전체"] + sorted(df_under["귀책분류"].dropna().unique().tolist())
                g_sel = st.selectbox("귀책", g_opts)
            if g_sel != "전체":
                df_under = df_under[df_under["귀책분류"] == g_sel]

        # 상담사 필터
        if "상담사" in df_under.columns:
            with fc4:
                a_opts = ["전체"] + sorted(df_under["상담사"].dropna().unique().tolist())
                a_sel = st.selectbox("상담사", a_opts)
            if a_sel != "전체":
                df_under = df_under[df_under["상담사"] == a_sel]

    if df_under.empty:
        st.info("조건에 해당하는 데이터가 없습니다.")
        return

    # ━━━━━━━━━━━ KPI 요약 ━━━━━━━━━━━━
    _section("개요")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("총 건수", f"{len(df_under):,}건", color=C_DANGER)
    with k2:
        avg_csat = df_under["최종점수"].mean() if "최종점수" in df_under.columns else 0
        _kpi_card("평균 CSAT", f"{avg_csat:.1f}점", color=C_WARNING)
    with k3:
        avg_qa = df_under["이행총점_계산"].mean()
        _kpi_card("평균 이행점수", f"{avg_qa:.1f}점", color=C_PRIMARY)
    with k4:
        if "귀책분류" in df_under.columns:
            top_g = df_under["귀책분류"].value_counts().idxmax()
            _kpi_card("최다 귀책", top_g, color=C_DANGER)
        else:
            _kpi_card("최다 귀책", "-")
    with k5:
        if "피드백여부" in df_under.columns:
            fb_rate = (df_under["피드백여부"].str.strip().str.upper() == "O").mean() * 100
            _kpi_card("피드백 완료율", f"{fb_rate:.1f}%", color=C_SUCCESS)
        else:
            _kpi_card("피드백 완료율", "-")

    # ━━━━━━━━━━━ 귀책 비중 분석 ━━━━━━━━━━━━
    if "귀책분류" in df_under.columns:
        _section("귀책 비중 분석")

        tab_g1, tab_g2, tab_g3 = st.tabs(["📊 전체 비중", "📡 채널별 비중", "📅 기간별 추이"])

        with tab_g1:
            g_counts = df_under["귀책분류"].value_counts().reset_index()
            g_counts.columns = ["귀책", "건수"]
            g_counts["비율"] = (g_counts["건수"] / g_counts["건수"].sum() * 100).round(1)

            gc1, gc2 = st.columns([1, 1])
            with gc1:
                fig = px.pie(
                    g_counts, names="귀책", values="건수",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.4,
                )
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=350)
                st.plotly_chart(fig, use_container_width=True)
            with gc2:
                st.dataframe(
                    g_counts.style.format({"비율": "{:.1f}%"}),
                    use_container_width=True, hide_index=True, height=350,
                )

        with tab_g2:
            if "채널구분" in df_under.columns:
                ch_g = (
                    df_under.groupby(["채널구분", "귀책분류"])
                    .size().reset_index(name="건수")
                )
                fig2 = px.bar(
                    ch_g, x="채널구분", y="건수", color="귀책분류",
                    barmode="group",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig2.update_layout(margin=dict(t=30, b=0), height=400)
                st.plotly_chart(fig2, use_container_width=True)

                # 채널별 비율 테이블
                ch_pivot = pd.crosstab(
                    df_under["채널구분"], df_under["귀책분류"],
                    margins=True, margins_name="합계",
                )
                st.dataframe(ch_pivot, use_container_width=True)

        with tab_g3:
            if "발송월_parsed" in df_under.columns:
                period_g = (
                    df_under.groupby(["발송월_parsed", "귀책분류"])
                    .size().reset_index(name="건수")
                )
                fig3 = px.bar(
                    period_g, x="발송월_parsed", y="건수", color="귀책분류",
                    barmode="stack",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig3.update_layout(
                    xaxis_title="월", margin=dict(t=30, b=0), height=400,
                )
                st.plotly_chart(fig3, use_container_width=True)

    # ━━━━━━━━━━━ 이행 항목별 감점 분석 ━━━━━━━━━━━━
    _section("이행 항목별 감점 현황")

    tab_i1, tab_i2, tab_i3 = st.tabs(["📉 항목별 감점률", "👤 상담사별 감점", "🔥 감점 사유 Top"])

    with tab_i1:
        deduct_data = []
        for item in ITEMS_ORDER:
            cat, full = ITEM_META[item]
            score_col = f"{item}_점수"
            if score_col in df_under.columns:
                deducted = (df_under[score_col].astype(float) < full).sum()
                rate = deducted / len(df_under) * 100 if len(df_under) > 0 else 0
                deduct_data.append({
                    "대분류": cat, "항목": item, "배점": full,
                    "감점건수": deducted, "감점률(%)": round(rate, 1),
                })
        df_deduct = pd.DataFrame(deduct_data)
        if not df_deduct.empty:
            fig4 = px.bar(
                df_deduct, x="항목", y="감점률(%)", color="대분류",
                color_discrete_map={
                    "정확성": "#6366f1", "숙련도": "#f59e0b",
                    "친절도": "#ef4444", "약속이행": "#22c55e",
                },
                text="감점률(%)",
            )
            fig4.update_layout(margin=dict(t=30, b=0), height=420)
            fig4.update_traces(textposition="outside", texttemplate="%{text}%")
            st.plotly_chart(fig4, use_container_width=True)

            st.dataframe(
                df_deduct.style.format({"감점률(%)": "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )

    with tab_i2:
        if "상담사" in df_under.columns:
            agent_scores = []
            for agent, grp in df_under.groupby("상담사"):
                row = {"상담사": agent, "건수": len(grp)}
                row["평균이행점수"] = round(grp["이행총점_계산"].mean(), 1)
                if "최종점수" in grp.columns:
                    row["평균CSAT"] = round(grp["최종점수"].mean(), 1)
                for cat in CATEGORY_FULL:
                    col = f"{cat}_득점"
                    if col in grp.columns:
                        row[f"평균_{cat}"] = round(grp[col].astype(float).mean(), 1)
                agent_scores.append(row)
            df_agent = pd.DataFrame(agent_scores).sort_values("평균이행점수")

            fig5 = px.bar(
                df_agent, x="상담사", y="평균이행점수",
                color="평균이행점수",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                text="평균이행점수",
            )
            fig5.update_layout(margin=dict(t=30, b=0), height=400)
            fig5.update_traces(textposition="outside", texttemplate="%{text}")
            st.plotly_chart(fig5, use_container_width=True)

            st.dataframe(df_agent, use_container_width=True, hide_index=True)

    with tab_i3:
        # 감점 사유별 빈도
        reason_counts = []
        for item in ITEMS_ORDER:
            if item in df_under.columns:
                vals = df_under[item].dropna().astype(str)
                vals = vals[vals.str.strip() != ""]
                vals = vals[vals.str.lower() != "nan"]
                for v in vals:
                    for reason in [r.strip() for r in v.split("/") if r.strip()]:
                        reason_counts.append({
                            "대분류": ITEM_META[item][0],
                            "항목": item,
                            "감점사유": reason,
                        })
        if reason_counts:
            df_reasons = pd.DataFrame(reason_counts)
            df_reason_agg = (
                df_reasons.groupby(["대분류", "항목", "감점사유"])
                .size().reset_index(name="건수")
                .sort_values("건수", ascending=False)
            )
            fig6 = px.bar(
                df_reason_agg.head(20),
                x="건수", y="감점사유", color="대분류",
                orientation="h",
                color_discrete_map={
                    "정확성": "#6366f1", "숙련도": "#f59e0b",
                    "친절도": "#ef4444", "약속이행": "#22c55e",
                },
            )
            fig6.update_layout(
                margin=dict(t=30, b=0), height=500,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig6, use_container_width=True)

    # ━━━━━━━━━━━ CSAT ↔ 이행 교차분석 ━━━━━━━━━━━━
    _section("CSAT ↔ 이행 교차분석")

    tab_c1, tab_c2, tab_c3 = st.tabs(
        ["🔄 CSAT vs 이행 상관", "📊 귀책별 점수 비교", "📋 상세 데이터"]
    )

    with tab_c1:
        if "최종점수" in df_under.columns:
            fig7 = px.scatter(
                df_under, x="이행총점_계산", y="최종점수",
                color="귀책분류" if "귀책분류" in df_under.columns else None,
                hover_data=["상담사", "상담이력KEY"] if "상담사" in df_under.columns else None,
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.7,
            )
            fig7.update_layout(
                xaxis_title="이행총점",
                yaxis_title="CSAT 최종점수",
                margin=dict(t=30, b=0), height=450,
            )
            st.plotly_chart(fig7, use_container_width=True)

            # 상관계수
            corr = df_under[["이행총점_계산", "최종점수"]].dropna().corr().iloc[0, 1]
            st.markdown(
                f"**이행총점 ↔ CSAT 상관계수:** `{corr:.3f}`"
            )

    with tab_c2:
        if "귀책분류" in df_under.columns:
            g_score = (
                df_under.groupby("귀책분류")
                .agg(
                    건수=("상담이력KEY", "count"),
                    평균CSAT=("최종점수", "mean"),
                    평균이행=("이행총점_계산", "mean"),
                    **{
                        f"평균_{cat}": (f"{cat}_득점", "mean")
                        for cat in CATEGORY_FULL
                        if f"{cat}_득점" in df_under.columns
                    },
                )
                .round(1)
                .reset_index()
                .sort_values("건수", ascending=False)
            )
            st.dataframe(g_score, use_container_width=True, hide_index=True)

            fig8 = px.bar(
                g_score.melt(
                    id_vars="귀책분류",
                    value_vars=["평균CSAT", "평균이행"],
                    var_name="구분", value_name="점수",
                ),
                x="귀책분류", y="점수", color="구분",
                barmode="group",
                color_discrete_map={"평균CSAT": C_PRIMARY, "평균이행": C_WARNING},
                text="점수",
            )
            fig8.update_layout(margin=dict(t=30, b=0), height=400)
            fig8.update_traces(textposition="outside", texttemplate="%{text:.1f}")
            st.plotly_chart(fig8, use_container_width=True)

    with tab_c3:
        # 상세 테이블
        display_cols = ["상담이력KEY"]
        for c in ["발송일자", "채널구분", "사업자", "브랜드", "상담사",
                   "최종점수", "귀책분류", "이행총점_계산",
                   "정확성_득점", "숙련도_득점", "친절도_득점", "약속이행_득점",
                   "문의불만사유", "피드백여부", "상세분석"]:
            if c in df_under.columns:
                display_cols.append(c)

        st.dataframe(
            df_under[display_cols].sort_values(
                "이행총점_계산", ascending=True
            ),
            use_container_width=True, hide_index=True, height=500,
        )

    # ━━━━━━━━━━━ 대분류별 히트맵 ━━━━━━━━━━━━
    if "상담사" in df_under.columns:
        _section("상담사 × 이행대분류 히트맵")
        heatmap_data = (
            df_under.groupby("상담사")[
                [f"{cat}_득점" for cat in CATEGORY_FULL if f"{cat}_득점" in df_under.columns]
            ]
            .mean().round(1)
        )
        heatmap_data.columns = [c.replace("_득점", "") for c in heatmap_data.columns]

        fig9 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale=[[0, "#ef4444"], [0.5, "#fef9c3"], [1, "#22c55e"]],
            text=heatmap_data.values,
            texttemplate="%{text:.1f}",
            textfont={"size": 12},
        ))
        fig9.update_layout(
            margin=dict(t=30, b=0, l=0, r=0),
            height=max(300, len(heatmap_data) * 35 + 80),
        )
        st.plotly_chart(fig9, use_container_width=True)
