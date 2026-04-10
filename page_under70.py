# ──────────────────────────────────────────────────────────────
#  page_under70.py
#  70점 미만 QA 모니터링 교차분석 페이지 (v3 - 디벨롭)
# ──────────────────────────────────────────────────────────────

import re
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# ── 색상 토큰 ─────────────────────────────────────────────────
C_PRIMARY   = "#6366f1"
C_PRIMARY_L = "#818cf8"
C_SUCCESS   = "#22c55e"
C_SUCCESS_L = "#86efac"
C_WARNING   = "#f59e0b"
C_WARNING_L = "#fcd34d"
C_DANGER    = "#ef4444"
C_DANGER_L  = "#fca5a5"
C_BG_CARD   = "#ffffff"
C_BG_PAGE   = "#f8fafc"
C_BORDER    = "#e2e8f0"
C_TEXT      = "#1e293b"
C_TEXT_SUB  = "#64748b"

CHANNEL_COLORS = {
    "전화 IN": "#6366f1", "전화 OUT": "#818cf8",
    "채팅": "#f59e0b", "이메일": "#22c55e",
    "게시판": "#ec4899", "기타": "#94a3b8",
}

SHEET_ID = "1ujtxIKZJRR9vIC1TS5GWWEtM9luChlDJk4NwTqeYB2Q"
GID      = 2055211445


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  0. 글로벌 CSS (Noto Sans KR + KPI 카드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');

    html, body, [class*="css"], .stMarkdown, .stDataFrame,
    .stSelectbox, .stMultiSelect, .stTextInput, .stMetric,
    h1, h2, h3, h4, h5, h6, p, span, div, label, td, th {
        font-family: 'Noto Sans KR', sans-serif !important;
    }
    .kpi-row { display: flex; gap: 14px; margin-bottom: 18px; }
    .kpi-bar {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
        transition: transform .15s;
    }
    .kpi-bar:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.10); }
    .kpi-bar-top { height: 6px; }
    .kpi-bar-body { padding: 16px 18px 14px; }
    .kpi-label {
        font-size: 12px; font-weight: 500; color: #64748b;
        margin-bottom: 4px; letter-spacing: -0.02em;
    }
    .kpi-value {
        font-size: 28px; font-weight: 900; letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .kpi-sub { font-size: 12px; color: #64748b; margin-top: 2px; }
    .section-divider {
        border: none; border-top: 2px solid #e2e8f0;
        margin: 32px 0 24px;
    }
    .section-title {
        font-size: 20px; font-weight: 700; color: #1e293b;
        margin-bottom: 4px; letter-spacing: -0.02em;
    }
    .section-caption { font-size: 13px; color: #64748b; margin-bottom: 16px; }
    .ch-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 700; color: #fff; margin-right: 4px;
    }
    .profile-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 20px; margin-bottom: 16px;
    }
    .profile-card h4 { margin: 0 0 8px; color: #1e293b; }
    .profile-tag {
        display: inline-block; background: #6366f1; color: #fff;
        padding: 2px 10px; border-radius: 12px; font-size: 11px;
        font-weight: 600; margin: 2px 4px 2px 0;
    }
    .profile-tag.warn { background: #ef4444; }
    .profile-tag.ok   { background: #22c55e; }
    </style>
    """, unsafe_allow_html=True)


def kpi_card_html(label, value, sub="", color=C_PRIMARY):
    return f"""
    <div class="kpi-bar">
        <div class="kpi-bar-top" style="background:{color};"></div>
        <div class="kpi-bar-body">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    </div>
    """


def render_kpi_row(cards_html: list):
    inner = "".join(cards_html)
    st.markdown(f'<div class="kpi-row">{inner}</div>', unsafe_allow_html=True)


def section_header(title, caption=""):
    html = f'<hr class="section-divider"><div class="section-title">{title}</div>'
    if caption:
        html += f'<div class="section-caption">{caption}</div>'
    st.markdown(html, unsafe_allow_html=True)


def ch_color(channel):
    return CHANNEL_COLORS.get(channel, "#94a3b8")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. 시트 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=600, show_spinner="모니터링 시트 로딩 중…")
def load_monitoring_sheet():
    try:
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
        df_raw = pd.read_csv(url, dtype=str, header=0)
    except Exception as e:
        st.error(f"시트 로드 실패: {e}")
        return pd.DataFrame()

    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df_raw = df_raw.astype(str)

    data_start = None
    for i in range(len(df_raw)):
        val = str(df_raw.iloc[i, 0]).strip()
        if val and (val.replace("월", "").isdigit() or val.isdigit()):
            data_start = i
            break
    if data_start is None:
        st.error("데이터 시작 행을 찾을 수 없습니다.")
        return pd.DataFrame()

    data = df_raw.iloc[data_start:].copy().reset_index(drop=True)
    ncols = data.shape[1]

    COL_IDX = {
        "회신월": 0, "발송월": 1, "회신주차": 2, "발송주차": 3,
        "발송일자": 4, "회신일자": 5, "사업자": 6, "브랜드": 7,
        "채널": 8, "상담사": 9, "입사일": 10, "상담사근속": 11,
        "상담유형대": 12, "상담유형중": 13, "상담유형소": 14,
        "키워드": 15, "긍정부정": 16, "유형": 17,
        "총합": 18, "Q1": 19, "Q2": 20, "Q3": 21,
        "친절점수": 22, "만족점수": 23, "최종점수": 24,
        "만족율": 25, "WK": 26, "상담이력KEY": 27,
        "문의유형": 28, "귀책분류": 29, "문의불만사유": 30,
        "정확한안내": 31, "프로세스": 32, "전산처리": 33,
        "맞춤설명": 34, "문의파악": 35, "숙련도_채널": 36,
        "친절도_감정": 37, "친절도_경청": 38, "언어표현": 39,
        "약속불이행": 40, "약속지연이행": 41, "약속시간누락": 42,
        "이행점수": 43, "상세분석": 44, "피드백여부": 45,
        "주문번호": 46,
    }

    rename = {}
    for col_name, idx in COL_IDX.items():
        if idx < ncols:
            rename[data.columns[idx]] = col_name
    data.rename(columns=rename, inplace=True)

    for c in data.columns:
        data[c] = data[c].apply(
            lambda x: "" if str(x).strip().lower() in ("nan", "none", "") else str(x).strip()
        )

    data = data[data["상담이력KEY"] != ""].copy()
    data = data[data["최종점수"] != ""].copy()

    if "회신월" in data.columns:
        data["회신월"] = data["회신월"].str.replace("월", "").str.strip()
        data["회신월"] = pd.to_numeric(data["회신월"], errors="coerce")

    for c in ["최종점수", "친절점수", "만족점수", "이행점수"]:
        if c in data.columns:
            data[c] = data[c].str.replace("%", "").str.strip()
            data[c] = pd.to_numeric(data[c], errors="coerce")

    for c in ["발송월", "회신주차", "발송주차", "WK"]:
        if c in data.columns:
            data[c] = data[c].str.replace("월", "").str.replace("WK", "").str.strip()
            data[c] = pd.to_numeric(data[c], errors="coerce")

    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. QA 차감 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QA_SUB_ITEMS = {
    "정확성(30)": {"정확한안내": 10, "프로세스": 10, "전산처리": 10},
    "숙련도(20)": {"맞춤설명": 10, "문의파악": 5, "숙련도_채널": 5},
    "친절도(30)": {"친절도_감정": 10, "친절도_경청": 15, "언어표현": 5},
    "약속이행(20)": {"약속불이행": 10, "약속지연이행": 5, "약속시간누락": 5},
}

SUB_LABELS = {
    "정확한안내": "정확한안내(10)", "프로세스": "프로세스(10)", "전산처리": "전산처리(10)",
    "맞춤설명": "맞춤설명(10)", "문의파악": "문의파악(5)", "숙련도_채널": "숙련도/채널(5)",
    "친절도_감정": "감정연출/양해(10)", "친절도_경청": "경청/즉각호응(15)", "언어표현": "언어표현(5)",
    "약속불이행": "약속불이행(10)", "약속지연이행": "약속지연이행(5)", "약속시간누락": "시간안내누락(5)",
}


def calc_deductions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for category, items in QA_SUB_ITEMS.items():
        cat_deduct = f"{category}_차감"
        df[cat_deduct] = 0
        for col, score in items.items():
            flag_col = f"{col}_감점"
            df[flag_col] = df[col].apply(lambda x: 1 if str(x).strip() != "" else 0)
            df[cat_deduct] = df[cat_deduct] + df[flag_col] * score

    df["QA총차감"] = (
        df["정확성(30)_차감"] + df["숙련도(20)_차감"] +
        df["친절도(30)_차감"] + df["약속이행(20)_차감"]
    )
    df["QA이행점수"] = 100 - df["QA총차감"]
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 상세분석(AS열) 종합 분석 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_analysis_keywords(texts: pd.Series, top_n=15) -> pd.DataFrame:
    """상세분석 텍스트에서 핵심 키워드/패턴 추출"""
    stop = {"있음", "없음", "함", "됨", "임", "것", "등", "및", "더", "로", "을", "를", "이", "가",
            "의", "에", "은", "는", "한", "하", "고", "도", "다", "수", "중", "대", "해", "안"}
    words = []
    for t in texts.dropna():
        t = str(t).strip()
        if not t:
            continue
        tokens = re.findall(r'[가-힣]{2,}', t)
        words.extend([w for w in tokens if w not in stop])
    if not words:
        return pd.DataFrame(columns=["키워드", "빈도"])
    freq = Counter(words).most_common(top_n)
    return pd.DataFrame(freq, columns=["키워드", "빈도"])


def build_agent_profile(df_agent: pd.DataFrame, agent_name: str) -> dict:
    """상담사 1인의 종합 프로파일 생성"""
    n = len(df_agent)
    profile = {
        "상담사": agent_name,
        "건수": n,
        "평균CSAT": round(df_agent["최종점수"].mean(), 1) if n else 0,
        "평균QA이행": round(df_agent["QA이행점수"].mean(), 1) if n else 0,
        "주요귀책": df_agent["귀책분류"].mode().iloc[0] if n and not df_agent["귀책분류"].mode().empty else "-",
    }

    # 가장 많이 차감된 대분류
    cat_cols = {"정확성(30)_차감": "정확성", "숙련도(20)_차감": "숙련도",
                "친절도(30)_차감": "친절도", "약속이행(20)_차감": "약속이행"}
    cat_means = {v: round(df_agent[k].mean(), 1) for k, v in cat_cols.items() if k in df_agent.columns}
    if cat_means:
        worst_cat = max(cat_means, key=cat_means.get)
        profile["최다차감분류"] = worst_cat
        profile["최다차감점수"] = cat_means[worst_cat]
    else:
        profile["최다차감분류"] = "-"
        profile["최다차감점수"] = 0

    # 가장 많이 차감된 세부항목
    sub_flags = [c for c in df_agent.columns if c.endswith("_감점")]
    if sub_flags:
        sub_sums = df_agent[sub_flags].sum()
        worst_sub = sub_sums.idxmax().replace("_감점", "")
        profile["최다차감항목"] = SUB_LABELS.get(worst_sub, worst_sub)
        profile["최다차감항목건수"] = int(sub_sums.max())
    else:
        profile["최다차감항목"] = "-"
        profile["최다차감항목건수"] = 0

    # 채널별 성과
    ch_perf = {}
    for ch in df_agent["채널"].unique():
        ch_df = df_agent[df_agent["채널"] == ch]
        ch_perf[ch] = {
            "건수": len(ch_df),
            "평균CSAT": round(ch_df["최종점수"].mean(), 1),
            "평균QA이행": round(ch_df["QA이행점수"].mean(), 1),
            "상담사귀책건수": int((ch_df["귀책분류"] == "상담사").sum()),
        }
    profile["채널별성과"] = ch_perf

    # 상세분석 키워드
    kw = extract_analysis_keywords(df_agent["상세분석"], top_n=10)
    profile["상세분석키워드"] = kw

    # 종합 코멘트 자동생성
    comments = []
    if profile["평균QA이행"] < 70:
        comments.append(f"QA이행점수 평균 {profile['평균QA이행']}점으로 심각한 수준")
    elif profile["평균QA이행"] < 85:
        comments.append(f"QA이행점수 평균 {profile['평균QA이행']}점으로 개선 필요")
    if profile["최다차감점수"] >= 10:
        comments.append(f"'{profile['최다차감분류']}' 영역 평균 {profile['최다차감점수']}점 차감 — 집중 코칭 필요")
    if profile["최다차감항목건수"] >= 3:
        comments.append(f"'{profile['최다차감항목']}' 반복 지적 {profile['최다차감항목건수']}건")

    # 채널 취약점
    for ch, perf in ch_perf.items():
        if perf["건수"] >= 2 and perf["평균QA이행"] < 75:
            comments.append(f"⚠️ [{ch}] 채널 QA이행 {perf['평균QA이행']}점 — 채널별 취약")
        if perf["건수"] >= 2 and perf["상담사귀책건수"] / perf["건수"] >= 0.7:
            pct = round(perf["상담사귀책건수"] / perf["건수"] * 100)
            comments.append(f"⚠️ [{ch}] 채널 상담사 귀책율 {pct}%")

    profile["종합코멘트"] = comments if comments else ["현재 특이사항 없음"]
    return profile


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 페이지 렌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_under70():
    inject_custom_css()

    st.markdown("## 📋 70점 미만 QA 모니터링 교차분석")
    st.caption("CSAT 최종점수 70점 미만 · 모니터링 이행평가 · 귀책 · 차감사유 · 고객코멘트 · 상담사 종합 프로파일")

    df_raw = load_monitoring_sheet()
    if df_raw.empty:
        st.warning("데이터를 불러올 수 없습니다.")
        return

    df = calc_deductions(df_raw)

    # ── 필터 ──────────────────────────────────────────────────
    with st.expander("🔍 필터 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            months = sorted(df["회신월"].dropna().unique())
            sel_months = st.multiselect("회신월", months, default=months, key="u70_m")
        with c2:
            agents = sorted(df["상담사"].unique())
            sel_agents = st.multiselect("상담사", agents, default=agents, key="u70_a")
        with c3:
            channels = sorted(df["채널"].unique())
            sel_ch = st.multiselect("채널", channels, default=channels, key="u70_c")
        with c4:
            blames = sorted(df["귀책분류"].dropna().unique())
            sel_blame = st.multiselect("귀책분류", blames, default=blames, key="u70_b")

    mask = (
        df["회신월"].isin(sel_months) &
        df["상담사"].isin(sel_agents) &
        df["채널"].isin(sel_ch) &
        df["귀책분류"].isin(sel_blame)
    )
    df_f = df[mask].copy()

    if df_f.empty:
        st.info("필터 조건에 맞는 데이터가 없습니다.")
        return

    # ══════════════════════════════════════════════════════════
    #  SECTION 1 — 전체 KPI (예쁜 카드 바)
    # ══════════════════════════════════════════════════════════
    section_header("전체 현황", "필터 적용 기준 모니터링 핵심 지표")

    total = len(df_f)
    avg_csat = df_f["최종점수"].mean()
    avg_qa = df_f["QA이행점수"].mean()
    avg_ded = df_f["QA총차감"].mean()
    agent_blame_cnt = int((df_f["귀책분류"] == "상담사").sum())
    agent_blame_pct = agent_blame_cnt / total * 100 if total else 0

    render_kpi_row([
        kpi_card_html("모니터링 건수", f"{total}건", "전체 필터 적용", C_PRIMARY),
        kpi_card_html("평균 CSAT", f"{avg_csat:.1f}점",
                      "양호" if avg_csat >= 70 else "주의", C_SUCCESS if avg_csat >= 70 else C_DANGER),
        kpi_card_html("평균 QA이행", f"{avg_qa:.1f}점",
                      "양호" if avg_qa >= 85 else "개선필요", C_SUCCESS if avg_qa >= 85 else C_WARNING),
        kpi_card_html("평균 차감", f"{avg_ded:.1f}점", "낮을수록 양호", C_WARNING),
        kpi_card_html("상담사 귀책", f"{agent_blame_pct:.1f}%", f"{agent_blame_cnt}건 / {total}건", C_DANGER),
    ])

    # ── 채널별 KPI ──
    st.markdown("")
    st.markdown("**채널별 KPI**")
    ch_groups = df_f.groupby("채널")
    ch_cards = []
    for ch_name in sorted(df_f["채널"].unique()):
        ch_df = ch_groups.get_group(ch_name)
        ch_n = len(ch_df)
        ch_csat = ch_df["최종점수"].mean()
        ch_qa = ch_df["QA이행점수"].mean()
        ch_blame = int((ch_df["귀책분류"] == "상담사").sum())
        ch_blame_p = ch_blame / ch_n * 100 if ch_n else 0
        color = ch_color(ch_name)
        ch_cards.append(kpi_card_html(
            f"📞 {ch_name}",
            f"{ch_n}건",
            f"CSAT {ch_csat:.1f} · QA {ch_qa:.1f} · 귀책 {ch_blame_p:.0f}%",
            color
        ))
    if ch_cards:
        render_kpi_row(ch_cards)

    # ══════════════════════════════════════════════════════════
    #  SECTION 2 — 귀책 분류
    # ══════════════════════════════════════════════════════════
    section_header("귀책 분류 현황", "전체 · 채널별 · 월별 추이")
    tab_all, tab_ch, tab_trend = st.tabs(["전체 비중", "채널별", "월별 추이"])

    with tab_all:
        blame = df_f["귀책분류"].value_counts().reset_index()
        blame.columns = ["귀책분류", "건수"]
        blame["비율(%)"] = (blame["건수"] / blame["건수"].sum() * 100).round(1)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(blame, names="귀책분류", values="건수",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=350, font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(blame, use_container_width=True, hide_index=True)

    with tab_ch:
        blame_ch = df_f.groupby(["채널", "귀책분류"]).size().reset_index(name="건수")
        fig2 = px.bar(blame_ch, x="채널", y="건수", color="귀책분류",
                      barmode="group", color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig2, use_container_width=True)

        # 채널별 귀책 비율 테이블
        ch_blame_piv = df_f.groupby(["채널", "귀책분류"]).size().unstack(fill_value=0)
        ch_blame_piv["합계"] = ch_blame_piv.sum(axis=1)
        for col in ch_blame_piv.columns[:-1]:
            ch_blame_piv[f"{col}(%)"] = (ch_blame_piv[col] / ch_blame_piv["합계"] * 100).round(1)
        st.dataframe(ch_blame_piv, use_container_width=True)

    with tab_trend:
        mt = df_f.groupby(["회신월", "귀책분류"]).size().reset_index(name="건수")
        fig_t = px.line(mt, x="회신월", y="건수", color="귀책분류",
                        markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_t.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_t, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 3 — 불만사유 & 차감사유 텍스트 상세
    # ══════════════════════════════════════════════════════════
    section_header("문의불만사유 & 차감사유 상세", "불만사유 Top 15 · 차감사유 텍스트 빈도 · 상세분석 키워드")
    tab_reason, tab_deduct_detail, tab_analysis_kw = st.tabs(
        ["불만사유 Top 15", "차감사유 텍스트 빈도", "상세분석(AS) 키워드"]
    )

    with tab_reason:
        reason = df_f[df_f["문의불만사유"] != ""]["문의불만사유"].value_counts().head(15).reset_index()
        reason.columns = ["불만사유", "건수"]
        fig3 = px.bar(reason, x="건수", y="불만사유", orientation="h",
                      color="건수", color_continuous_scale="Reds")
        fig3.update_layout(height=450, yaxis=dict(autorange="reversed"),
                           font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab_deduct_detail:
        sub_cols = list(SUB_LABELS.keys())
        all_reasons = []
        for _, row in df_f.iterrows():
            for col in sub_cols:
                val = str(row.get(col, "")).strip()
                if val:
                    all_reasons.append({"항목": SUB_LABELS.get(col, col), "차감사유": val})
        if all_reasons:
            df_reasons = pd.DataFrame(all_reasons)
            reason_freq = df_reasons.groupby(["항목", "차감사유"]).size().reset_index(name="건수")
            reason_freq = reason_freq.sort_values("건수", ascending=False)
            st.dataframe(reason_freq.head(30), use_container_width=True, hide_index=True)

            fig_sun = px.sunburst(
                reason_freq.head(50), path=["항목", "차감사유"], values="건수",
                color="건수", color_continuous_scale="YlOrRd",
                title="항목별 차감사유 분포"
            )
            fig_sun.update_layout(height=500, font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("차감 사유 텍스트가 없습니다.")

    with tab_analysis_kw:
        kw_df = extract_analysis_keywords(df_f["상세분석"], top_n=20)
        if not kw_df.empty:
            fig_kw = px.bar(kw_df, x="빈도", y="키워드", orientation="h",
                            color="빈도", color_continuous_scale="Purples",
                            title="상세분석(AS열) 핵심 키워드 Top 20")
            fig_kw.update_layout(height=500, yaxis=dict(autorange="reversed"),
                                 font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_kw, use_container_width=True)

            # 채널별 상세분석 키워드 비교
            st.markdown("**채널별 상세분석 키워드 비교**")
            ch_kw_tabs = st.tabs(sorted(df_f["채널"].unique()))
            for tab_obj, ch_name in zip(ch_kw_tabs, sorted(df_f["채널"].unique())):
                with tab_obj:
                    ch_kw = extract_analysis_keywords(
                        df_f[df_f["채널"] == ch_name]["상세분석"], top_n=10
                    )
                    if not ch_kw.empty:
                        fig_ckw = px.bar(ch_kw, x="빈도", y="키워드", orientation="h",
                                         color="빈도", color_continuous_scale="Blues",
                                         title=f"[{ch_name}] 상세분석 키워드 Top 10")
                        fig_ckw.update_layout(height=350, yaxis=dict(autorange="reversed"),
                                              font=dict(family="Noto Sans KR"))
                        st.plotly_chart(fig_ckw, use_container_width=True)
                    else:
                        st.info(f"[{ch_name}] 상세분석 텍스트가 없습니다.")
        else:
            st.info("상세분석 데이터가 없습니다.")

    # ══════════════════════════════════════════════════════════
    #  SECTION 4 — QA 항목별 차감
    # ══════════════════════════════════════════════════════════
    section_header("QA 항목별 차감 현황", "대분류별 · 세부항목별 · 상담사×항목 히트맵 · 채널별 비교")
    tab_cat, tab_sub, tab_heatmap, tab_ch_qa = st.tabs(
        ["대분류별", "세부항목별", "상담사×항목 히트맵", "채널별 항목 비교"]
    )

    with tab_cat:
        cat_cols = ["정확성(30)_차감", "숙련도(20)_차감", "친절도(30)_차감", "약속이행(20)_차감"]
        cat_means = df_f[cat_cols].mean().reset_index()
        cat_means.columns = ["항목", "평균차감"]
        cat_means["항목"] = cat_means["항목"].str.replace("_차감", "")
        fig4 = px.bar(cat_means, x="항목", y="평균차감",
                      color="평균차감", color_continuous_scale="OrRd",
                      title="대분류별 평균 차감 점수")
        fig4.update_layout(height=350, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig4, use_container_width=True)

    with tab_sub:
        sub_flags = [c for c in df_f.columns if c.endswith("_감점")]
        sub_counts = df_f[sub_flags].sum().reset_index()
        sub_counts.columns = ["항목", "차감건수"]
        sub_counts["항목"] = sub_counts["항목"].str.replace("_감점", "").map(
            lambda x: SUB_LABELS.get(x, x)
        )
        sub_counts = sub_counts.sort_values("차감건수", ascending=False)
        fig5 = px.bar(sub_counts, x="차감건수", y="항목", orientation="h",
                      color="차감건수", color_continuous_scale="YlOrRd")
        fig5.update_layout(height=500, yaxis=dict(autorange="reversed"),
                           font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig5, use_container_width=True)

    with tab_heatmap:
        sub_flag_cols = [c for c in df_f.columns if c.endswith("_감점")]
        hm = df_f.groupby("상담사")[sub_flag_cols].mean().round(2) * 100
        hm.columns = [SUB_LABELS.get(c.replace("_감점", ""), c.replace("_감점", "")) for c in hm.columns]
        fig_hm = px.imshow(
            hm, text_auto=".0f", aspect="auto",
            color_continuous_scale="YlOrRd",
            title="상담사별 항목 감점율 (%)",
            labels=dict(x="QA 항목", y="상담사", color="감점율(%)"),
        )
        fig_hm.update_layout(height=max(350, len(hm) * 35),
                             font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_ch_qa:
        # 채널별 대분류 차감 비교
        ch_cat = df_f.groupby("채널")[cat_cols].mean().round(1)
        ch_cat.columns = [c.replace("_차감", "") for c in ch_cat.columns]
        ch_cat_long = ch_cat.reset_index().melt(id_vars="채널", var_name="항목", value_name="평균차감")
        fig_chqa = px.bar(ch_cat_long, x="채널", y="평균차감", color="항목",
                          barmode="group", color_discrete_sequence=[C_DANGER, C_WARNING, C_PRIMARY, C_SUCCESS],
                          title="채널별 대분류 평균 차감 비교")
        fig_chqa.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_chqa, use_container_width=True)

        # 채널별 세부항목 히트맵
        ch_sub_hm = df_f.groupby("채널")[sub_flag_cols].mean().round(2) * 100
        ch_sub_hm.columns = [SUB_LABELS.get(c.replace("_감점", ""), c.replace("_감점", "")) for c in ch_sub_hm.columns]
        fig_chsub = px.imshow(
            ch_sub_hm, text_auto=".0f", aspect="auto",
            color_continuous_scale="YlOrRd",
            title="채널별 세부항목 감점율 (%)",
            labels=dict(x="QA 항목", y="채널", color="감점율(%)"),
        )
        fig_chsub.update_layout(height=max(250, len(ch_sub_hm) * 50),
                                font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_chsub, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 5 — 고객 코멘트(Q3) vs 귀책 교차분석
    # ══════════════════════════════════════════════════════════
    section_header("고객 코멘트(Q3) vs 실제 귀책 교차분석",
                   "고객 긍정인데 상담사 귀책? 부정인데 고객/IBR 귀책? 불일치 케이스 분석")
    tab_mismatch, tab_sentiment = st.tabs(["불일치 케이스", "긍정/부정 × 귀책"])

    with tab_mismatch:
        pos_agent = df_f[(df_f["긍정부정"] == "긍정") & (df_f["귀책분류"] == "상담사")]
        neg_not_agent = df_f[(df_f["긍정부정"] == "부정") & (df_f["귀책분류"].isin(["고객", "IBR"]))]

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**고객 긍정 → 상담사 귀책**: {len(pos_agent)}건")
            if not pos_agent.empty:
                st.dataframe(
                    pos_agent[["상담사", "채널", "최종점수", "Q3", "귀책분류", "문의불만사유", "상세분석"]].head(20),
                    use_container_width=True, hide_index=True, height=300
                )
        with mc2:
            st.markdown(f"**고객 부정 → 고객/IBR 귀책**: {len(neg_not_agent)}건")
            if not neg_not_agent.empty:
                st.dataframe(
                    neg_not_agent[["상담사", "채널", "최종점수", "Q3", "귀책분류", "문의불만사유", "상세분석"]].head(20),
                    use_container_width=True, hide_index=True, height=300
                )

    with tab_sentiment:
        cross_sent = df_f.groupby(["긍정부정", "귀책분류"]).size().reset_index(name="건수")
        fig_cs = px.bar(cross_sent, x="귀책분류", y="건수", color="긍정부정",
                        barmode="group",
                        color_discrete_map={"긍정": C_SUCCESS, "부정": C_DANGER, "기타": C_WARNING},
                        title="고객 긍정/부정 × 귀책 분류")
        fig_cs.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_cs, use_container_width=True)

        sent_score = df_f.groupby("긍정부정").agg(
            건수=("상담이력KEY", "count"),
            평균_CSAT=("최종점수", "mean"),
            평균_QA이행=("QA이행점수", "mean"),
        ).round(1).reset_index()
        st.dataframe(sent_score, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 6 — 상담사별 QA 상세 성과 + 종합 프로파일
    # ══════════════════════════════════════════════════════════
    section_header("상담사별 QA 상세 성과 & 종합 프로파일",
                   "성과 테이블 · 레이더 · 귀책 · 채널별 취약점 · 상세분석 종합 · 자동 코칭 포인트")

    agent_agg = (
        df_f.groupby("상담사")
        .agg(
            건수=("상담이력KEY", "count"),
            평균CSAT=("최종점수", "mean"),
            평균QA이행=("QA이행점수", "mean"),
            평균차감=("QA총차감", "mean"),
            정확성=("정확성(30)_차감", "mean"),
            숙련도=("숙련도(20)_차감", "mean"),
            친절도=("친절도(30)_차감", "mean"),
            약속이행=("약속이행(20)_차감", "mean"),
            상담사귀책=("귀책분류", lambda x: (x == "상담사").sum()),
            IBR귀책=("귀책분류", lambda x: (x == "IBR").sum()),
            고객귀책=("귀책분류", lambda x: (x == "고객").sum()),
        )
        .round(1)
        .sort_values("평균QA이행")
        .reset_index()
    )
    agent_agg["상담사귀책율(%)"] = (agent_agg["상담사귀책"] / agent_agg["건수"] * 100).round(1)
    st.dataframe(agent_agg, use_container_width=True, hide_index=True)

    # ── 상담사 상세 드릴다운 ──
    sel_agent = st.selectbox("🔎 상담사 상세 보기", agent_agg["상담사"].tolist(), key="u70_agent_sel")
    if sel_agent:
        df_agent = df_f[df_f["상담사"] == sel_agent]
        row = agent_agg[agent_agg["상담사"] == sel_agent].iloc[0]
        profile = build_agent_profile(df_agent, sel_agent)

        # ── 종합 프로파일 카드 ──
        tags_html = ""
        for cmt in profile["종합코멘트"]:
            cls = "warn" if "⚠️" in cmt or "심각" in cmt or "필요" in cmt else "ok"
            tags_html += f'<span class="profile-tag {cls}">{cmt}</span> '

        ch_badges = ""
        for ch, perf in profile["채널별성과"].items():
            bg = ch_color(ch)
            ch_badges += (
                f'<span class="ch-badge" style="background:{bg};">'
                f'{ch}: {perf["건수"]}건 · CSAT {perf["평균CSAT"]} · QA {perf["평균QA이행"]}'
                f'</span> '
            )

        st.markdown(f"""
        <div class="profile-card">
            <h4>👤 {sel_agent} — 종합 프로파일</h4>
            <p style="margin:4px 0;">
                건수 <b>{profile['건수']}</b> · 평균CSAT <b>{profile['평균CSAT']}</b>
                · 평균QA이행 <b>{profile['평균QA이행']}</b>
                · 주요귀책 <b>{profile['주요귀책']}</b>
                · 최다차감 <b>{profile['최다차감분류']}({profile['최다차감점수']}점)</b>
                · 반복지적 <b>{profile['최다차감항목']}({profile['최다차감항목건수']}건)</b>
            </p>
            <div style="margin:8px 0;">{ch_badges}</div>
            <div style="margin:8px 0;">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 레이더 + 귀책 파이 ──
        ac1, ac2 = st.columns([1, 1])
        with ac1:
            cats = ["정확성(30)", "숙련도(20)", "친절도(30)", "약속이행(20)"]
            deducts = [row["정확성"], row["숙련도"], row["친절도"], row["약속이행"]]
            max_vals = [30, 20, 30, 20]
            earned = [m - d for m, d in zip(max_vals, deducts)]

            fig6 = go.Figure()
            fig6.add_trace(go.Scatterpolar(
                r=earned + [earned[0]],
                theta=cats + [cats[0]],
                fill="toself", name=sel_agent, line_color=C_PRIMARY,
                fillcolor="rgba(99,102,241,0.15)",
            ))
            fig6.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
                title=f"{sel_agent} — 항목별 획득 점수", height=380,
                font=dict(family="Noto Sans KR"),
            )
            st.plotly_chart(fig6, use_container_width=True)

        with ac2:
            ab = df_agent["귀책분류"].value_counts().reset_index()
            ab.columns = ["귀책분류", "건수"]
            fig_ab = px.pie(ab, names="귀책분류", values="건수",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            title=f"{sel_agent} — 귀책 분포")
            fig_ab.update_layout(height=380, font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_ab, use_container_width=True)

        # ── 채널별 성과 비교 ──
        st.markdown(f"**{sel_agent} — 채널별 성과 비교**")
        ch_perf_data = profile["채널별성과"]
        if ch_perf_data:
            ch_rows = []
            for ch, perf in ch_perf_data.items():
                ch_rows.append({
                    "채널": ch, "건수": perf["건수"],
                    "평균CSAT": perf["평균CSAT"], "평균QA이행": perf["평균QA이행"],
                    "상담사귀책건수": perf["상담사귀책건수"],
                    "귀책율(%)": round(perf["상담사귀책건수"] / perf["건수"] * 100, 1) if perf["건수"] else 0,
                })
            df_ch_perf = pd.DataFrame(ch_rows)
            st.dataframe(df_ch_perf, use_container_width=True, hide_index=True)

            # 채널별 QA이행 바 차트
            fig_chp = px.bar(df_ch_perf, x="채널", y="평균QA이행",
                             color="채널",
                             color_discrete_map={ch: ch_color(ch) for ch in df_ch_perf["채널"]},
                             title=f"{sel_agent} — 채널별 QA이행 비교",
                             text="평균QA이행")
            fig_chp.update_layout(height=350, showlegend=False,
                                  font=dict(family="Noto Sans KR"))
            fig_chp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_chp, use_container_width=True)

        # ── 상세분석(AS열) 키워드 ──
        st.markdown(f"**{sel_agent} — 상세분석(AS열) 종합 키워드**")
        kw_agent = profile["상세분석키워드"]
        if not kw_agent.empty:
            fig_akw = px.bar(kw_agent, x="빈도", y="키워드", orientation="h",
                             color="빈도", color_continuous_scale="Purples",
                             title=f"{sel_agent} 상세분석 핵심 키워드")
            fig_akw.update_layout(height=350, yaxis=dict(autorange="reversed"),
                                  font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_akw, use_container_width=True)
        else:
            st.info("상세분석 텍스트가 없습니다.")

        # ── 상세분석 원문 모아보기 ──
        with st.expander(f"{sel_agent} — 상세분석 원문 전체"):
            analysis_texts = df_agent[df_agent["상세분석"] != ""][["회신월", "채널", "최종점수",
                                                                  "귀책분류", "상세분석"]].sort_values("회신월")
            if not analysis_texts.empty:
                st.dataframe(analysis_texts, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("상세분석 데이터 없음")

        # ── 차감 사유 상세 ──
        st.markdown(f"**{sel_agent}의 차감사유 상세**")
        sub_cols = list(SUB_LABELS.keys())
        agent_reasons = []
        for _, r in df_agent.iterrows():
            for col in sub_cols:
                val = str(r.get(col, "")).strip()
                if val:
                    agent_reasons.append({
                        "항목": SUB_LABELS.get(col, col),
                        "차감사유": val,
                        "귀책": r.get("귀책분류", ""),
                        "최종점수": r.get("최종점수", ""),
                    })
        if agent_reasons:
            df_ar = pd.DataFrame(agent_reasons)
            ar_freq = df_ar.groupby(["항목", "차감사유"]).size().reset_index(name="건수")
            ar_freq = ar_freq.sort_values("건수", ascending=False)
            st.dataframe(ar_freq, use_container_width=True, hide_index=True)
        else:
            st.info("차감 사유가 없습니다.")

        # ── 건별 상세 ──
        st.markdown(f"**{sel_agent} — 건별 상세**")
        detail_cols = ["회신월", "채널", "사업자", "브랜드", "최종점수", "QA이행점수",
                       "귀책분류", "문의불만사유", "긍정부정", "Q3", "상세분석", "피드백여부"]
        ex_cols = [c for c in detail_cols if c in df_agent.columns]
        st.dataframe(
            df_agent[ex_cols].sort_values("최종점수"),
            use_container_width=True, hide_index=True, height=400
        )

    # ══════════════════════════════════════════════════════════
    #  SECTION 7 — 귀책 × CSAT × QA이행 교차
    # ══════════════════════════════════════════════════════════
    section_header("귀책 × CSAT × QA이행 교차", "귀책 유형별 평균 점수 비교 · 산점도")

    cross = (
        df_f.groupby("귀책분류")
        .agg(건수=("상담이력KEY", "count"),
             평균CSAT=("최종점수", "mean"),
             평균QA이행=("QA이행점수", "mean"),
             평균차감=("QA총차감", "mean"))
        .round(1).sort_values("건수", ascending=False).reset_index()
    )
    st.dataframe(cross, use_container_width=True, hide_index=True)

    fig7 = px.scatter(
        df_f, x="최종점수", y="QA이행점수", size="QA총차감",
        color="귀책분류", hover_data=["상담사", "채널", "문의불만사유", "Q3"],
        title="CSAT vs QA이행 (버블=차감량, 색=귀책)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig7.update_layout(height=500, font=dict(family="Noto Sans KR"))
    st.plotly_chart(fig7, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 8 — 상세 데이터 테이블
    # ══════════════════════════════════════════════════════════
    section_header("전체 상세 데이터", "필터 적용 전체 데이터 (QA이행점수 오름차순)")

    show_cols = [
        "회신월", "발송일자", "상담사", "채널", "사업자", "브랜드",
        "상담유형대", "긍정부정", "최종점수", "친절점수", "만족점수",
        "귀책분류", "문의불만사유", "QA이행점수", "QA총차감",
        "정확성(30)_차감", "숙련도(20)_차감", "친절도(30)_차감", "약속이행(20)_차감",
        "Q3", "상세분석", "피드백여부",
    ]
    existing = [c for c in show_cols if c in df_f.columns]
    st.dataframe(
        df_f[existing].sort_values("QA이행점수"),
        use_container_width=True, hide_index=True, height=500,
    )


# ── csat.py 호환용 alias ──
page_under70_analysis = page_under70
