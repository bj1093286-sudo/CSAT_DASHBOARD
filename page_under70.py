# ──────────────────────────────────────────────────────────────
#  page_under70.py
#  70점 미만 QA 모니터링 교차분석 (v4 - 종합 프로파일 강화)
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
C_WARNING   = "#f59e0b"
C_DANGER    = "#ef4444"
C_BG_CARD   = "#ffffff"
C_BG_PAGE   = "#f8fafc"
C_BORDER    = "#e2e8f0"
C_TEXT      = "#1e293b"
C_TEXT_SUB  = "#64748b"

CHANNEL_COLORS = {"전화 IN": "#6366f1", "채팅": "#f59e0b"}

SHEET_ID = "1ujtxIKZJRR9vIC1TS5GWWEtM9luChlDJk4NwTqeYB2Q"
GID      = 2055211445

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  0. CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    html, body, [class*="css"], .stMarkdown, .stDataFrame,
    .stSelectbox, .stMultiSelect, .stTextInput, .stMetric,
    h1,h2,h3,h4,h5,h6,p,span,div,label,td,th {
        font-family: 'Noto Sans KR', sans-serif !important;
    }
    .kpi-row{display:flex;gap:14px;margin-bottom:18px;flex-wrap:wrap;}
    .kpi-bar{flex:1;min-width:160px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);transition:transform .15s;}
    .kpi-bar:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.10);}
    .kpi-bar-top{height:6px;}
    .kpi-bar-body{padding:16px 18px 14px;}
    .kpi-label{font-size:12px;font-weight:500;color:#64748b;margin-bottom:4px;}
    .kpi-value{font-size:26px;font-weight:900;line-height:1.15;}
    .kpi-sub{font-size:11px;color:#64748b;margin-top:2px;}
    .section-divider{border:none;border-top:2px solid #e2e8f0;margin:32px 0 24px;}
    .section-title{font-size:20px;font-weight:700;color:#1e293b;margin-bottom:4px;}
    .section-caption{font-size:13px;color:#64748b;margin-bottom:16px;}
    .profile-wrap{background:linear-gradient(135deg,#f8fafc,#eef2ff);border:1px solid #e2e8f0;border-radius:16px;padding:24px;margin-bottom:20px;}
    .profile-name{font-size:22px;font-weight:900;color:#1e293b;margin-bottom:6px;}
    .profile-stat{display:inline-block;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:6px 14px;margin:3px 4px 3px 0;font-size:13px;font-weight:500;}
    .profile-stat b{color:#6366f1;}
    .tag-danger{display:inline-block;background:#fef2f2;color:#dc2626;border:1px solid #fecaca;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;margin:3px 4px 3px 0;}
    .tag-warn{display:inline-block;background:#fffbeb;color:#d97706;border:1px solid #fde68a;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;margin:3px 4px 3px 0;}
    .tag-ok{display:inline-block;background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;margin:3px 4px 3px 0;}
    .tag-info{display:inline-block;background:#eef2ff;color:#4f46e5;border:1px solid #c7d2fe;border-radius:8px;padding:4px 12px;font-size:12px;font-weight:600;margin:3px 4px 3px 0;}
    .ch-pill{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;color:#fff;margin:3px 4px 3px 0;}
    .summary-box{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin-top:12px;font-size:14px;line-height:1.75;color:#334155;}
    .summary-box b{color:#6366f1;}
    .summary-box .red{color:#dc2626;font-weight:700;}
    .summary-box .orange{color:#d97706;font-weight:700;}
    </style>
    """, unsafe_allow_html=True)


def kpi_card_html(label, value, sub="", color=C_PRIMARY):
    return f'''<div class="kpi-bar"><div class="kpi-bar-top" style="background:{color};"></div>
    <div class="kpi-bar-body"><div class="kpi-label">{label}</div>
    <div class="kpi-value" style="color:{color};">{value}</div>
    <div class="kpi-sub">{sub}</div></div></div>'''


def render_kpi_row(cards):
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def section_header(title, caption=""):
    h = f'<hr class="section-divider"><div class="section-title">{title}</div>'
    if caption:
        h += f'<div class="section-caption">{caption}</div>'
    st.markdown(h, unsafe_allow_html=True)


def ch_color(ch):
    return CHANNEL_COLORS.get(ch, "#94a3b8")


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
        "회신월":0,"발송월":1,"회신주차":2,"발송주차":3,"발송일자":4,"회신일자":5,
        "사업자":6,"브랜드":7,"채널":8,"상담사":9,"입사일":10,"상담사근속":11,
        "상담유형대":12,"상담유형중":13,"상담유형소":14,"키워드":15,"긍정부정":16,
        "유형":17,"총합":18,"Q1":19,"Q2":20,"Q3":21,"친절점수":22,"만족점수":23,
        "최종점수":24,"만족율":25,"WK":26,"상담이력KEY":27,"문의유형":28,"귀책분류":29,
        "문의불만사유":30,"정확한안내":31,"프로세스":32,"전산처리":33,"맞춤설명":34,
        "문의파악":35,"숙련도_채널":36,"친절도_감정":37,"친절도_경청":38,"언어표현":39,
        "약속불이행":40,"약속지연이행":41,"약속시간누락":42,"이행점수":43,"상세분석":44,
        "피드백여부":45,"주문번호":46,
    }
    rename = {}
    for col_name, idx in COL_IDX.items():
        if idx < ncols:
            rename[data.columns[idx]] = col_name
    data.rename(columns=rename, inplace=True)
    for c in data.columns:
        data[c] = data[c].apply(lambda x: "" if str(x).strip().lower() in ("nan","none","") else str(x).strip())
    data = data[data["상담이력KEY"] != ""].copy()
    data = data[data["최종점수"] != ""].copy()
    if "회신월" in data.columns:
        data["회신월"] = data["회신월"].str.replace("월","").str.strip()
        data["회신월"] = pd.to_numeric(data["회신월"], errors="coerce")
    for c in ["최종점수","친절점수","만족점수","이행점수"]:
        if c in data.columns:
            data[c] = data[c].str.replace("%","").str.strip()
            data[c] = pd.to_numeric(data[c], errors="coerce")
    for c in ["발송월","회신주차","발송주차","WK"]:
        if c in data.columns:
            data[c] = data[c].str.replace("월","").str.replace("WK","").str.strip()
            data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. QA 차감 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QA_SUB_ITEMS = {
    "정확성(30)": {"정확한안내":10,"프로세스":10,"전산처리":10},
    "숙련도(20)": {"맞춤설명":10,"문의파악":5,"숙련도_채널":5},
    "친절도(30)": {"친절도_감정":10,"친절도_경청":15,"언어표현":5},
    "약속이행(20)": {"약속불이행":10,"약속지연이행":5,"약속시간누락":5},
}
SUB_LABELS = {
    "정확한안내":"정확한안내(10)","프로세스":"프로세스(10)","전산처리":"전산처리(10)",
    "맞춤설명":"맞춤설명(10)","문의파악":"문의파악(5)","숙련도_채널":"숙련도/채널(5)",
    "친절도_감정":"감정연출/양해(10)","친절도_경청":"경청/즉각호응(15)","언어표현":"언어표현(5)",
    "약속불이행":"약속불이행(10)","약속지연이행":"약속지연이행(5)","약속시간누락":"시간안내누락(5)",
}

def calc_deductions(df):
    df = df.copy()
    for category, items in QA_SUB_ITEMS.items():
        cat_d = f"{category}_차감"
        df[cat_d] = 0
        for col, score in items.items():
            flag = f"{col}_감점"
            df[flag] = df[col].apply(lambda x: 1 if str(x).strip() != "" else 0)
            df[cat_d] = df[cat_d] + df[flag] * score
    df["QA총차감"] = df["정확성(30)_차감"]+df["숙련도(20)_차감"]+df["친절도(30)_차감"]+df["약속이행(20)_차감"]
    df["QA이행점수"] = 100 - df["QA총차감"]
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. 상세분석 종합 프로파일 엔진
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 피드백 패턴 사전 — 상세분석 텍스트에서 반복 등장하는 코칭 패턴
FEEDBACK_PATTERNS = {
    "사과/양해 부족": re.compile(r"사과|양해.*(?:누락|부족|필요)"),
    "쿠션어 미사용": re.compile(r"쿠션어.*(?:필요|사용|누락)"),
    "비정중 표현": re.compile(r"비정중|불손|요조체|직설"),
    "보류버튼 미사용": re.compile(r"보류.*(?:버튼|필수)"),
    "설명 미흡": re.compile(r"설명.*(?:미흡|부족)|적극.*(?:설명|안내).*(?:없|부족)"),
    "케어 부족": re.compile(r"케어.*(?:부족|없|필요)|적극.*케어"),
    "프로세스 미준수": re.compile(r"프로세스.*(?:미준수|준수|위반)"),
    "선종결/선개입": re.compile(r"선종결|선개입|선종료"),
    "방어적 응대": re.compile(r"방어적|일방적.*안내"),
    "고객 불만 가중": re.compile(r"불만.*(?:가중|증가|확대)"),
    "컨시어지 미진행": re.compile(r"컨시어지.*(?:필요|진행|미)"),
    "끝인사/화답 누락": re.compile(r"끝인사|화답.*(?:누락|인사)"),
    "대기 지연": re.compile(r"대기.*(?:지연|발생|10초)"),
    "가독성 부족": re.compile(r"가독성|띄어쓰기"),
    "개인의견 언급": re.compile(r"개인.*(?:의견|선호|생각).*(?:언급|금지)"),
}


def extract_feedback_patterns(texts):
    """상세분석 텍스트에서 피드백 패턴 빈도 추출"""
    counts = Counter()
    for t in texts.dropna():
        t = str(t).strip()
        if not t:
            continue
        for label, pattern in FEEDBACK_PATTERNS.items():
            if pattern.search(t):
                counts[label] += 1
    return counts


def generate_agent_summary(agent_name, df_agent):
    """상담사 1인의 자연어 종합평가 생성"""
    n = len(df_agent)
    if n == 0:
        return "데이터가 없습니다."

    avg_csat = df_agent["최종점수"].mean()
    avg_qa = df_agent["QA이행점수"].mean()
    avg_ded = df_agent["QA총차감"].mean()

    # 귀책 분석
    blame_counts = df_agent["귀책분류"].value_counts().to_dict()
    agent_blame = blame_counts.get("상담사", 0)
    agent_blame_pct = round(agent_blame / n * 100)

    # 채널별
    ch_summary = []
    for ch in df_agent["채널"].unique():
        ch_df = df_agent[df_agent["채널"] == ch]
        ch_n = len(ch_df)
        ch_qa = ch_df["QA이행점수"].mean()
        ch_blame = int((ch_df["귀책분류"] == "상담사").sum())
        ch_summary.append({"채널": ch, "건수": ch_n, "QA": round(ch_qa, 1),
                           "귀책": ch_blame, "귀책율": round(ch_blame / ch_n * 100) if ch_n else 0})

    # 대분류별 차감
    cat_map = {"정확성(30)_차감":"정확성","숙련도(20)_차감":"숙련도",
               "친절도(30)_차감":"친절도","약속이행(20)_차감":"약속이행"}
    cat_avgs = {v: round(df_agent[k].mean(), 1) for k, v in cat_map.items() if k in df_agent.columns}
    worst_cat = max(cat_avgs, key=cat_avgs.get) if cat_avgs else "-"
    worst_cat_score = cat_avgs.get(worst_cat, 0)

    # 세부항목 최다 차감
    sub_flags = [c for c in df_agent.columns if c.endswith("_감점")]
    sub_sums = df_agent[sub_flags].sum().sort_values(ascending=False)
    top_subs = []
    for col_flag, cnt in sub_sums.head(3).items():
        if cnt > 0:
            col_name = col_flag.replace("_감점", "")
            top_subs.append((SUB_LABELS.get(col_name, col_name), int(cnt)))

    # 상세분석 피드백 패턴
    fb_patterns = extract_feedback_patterns(df_agent["상세분석"])
    top_fb = fb_patterns.most_common(5)

    # 긍정 고객인데 상담사 귀책
    pos_but_blame = len(df_agent[(df_agent["긍정부정"] == "긍정") & (df_agent["귀책분류"] == "상담사")])

    # ── 자연어 생성 ──
    lines = []
    lines.append(f"<b>{agent_name}</b>은(는) 해당 기간 총 <b>{n}건</b>의 모니터링 대상 상담을 수행했습니다.")
    lines.append(f"평균 CSAT <b>{avg_csat:.1f}점</b>, QA이행 <b>{avg_qa:.1f}점</b> (평균 차감 {avg_ded:.1f}점)입니다.")

    # 귀책
    if agent_blame_pct >= 60:
        lines.append(f'<span class="red">상담사 귀책 비율이 {agent_blame_pct}%({agent_blame}건)로 매우 높습니다.</span> 상담 품질에 대한 집중적인 코칭이 필요합니다.')
    elif agent_blame_pct >= 40:
        lines.append(f'<span class="orange">상담사 귀책 비율이 {agent_blame_pct}%({agent_blame}건)입니다.</span> 개선이 필요한 수준입니다.')
    else:
        lines.append(f"상담사 귀책 비율은 {agent_blame_pct}%({agent_blame}건)로, 대부분 외부 요인(IBR/고객)에 의한 저점입니다.")

    # 채널별
    for cs in ch_summary:
        ch_tag = f'[{cs["채널"]}]'
        if cs["귀책율"] >= 60:
            lines.append(f'<span class="red">{ch_tag} 채널에서 귀책율 {cs["귀책율"]}%로 특히 취약합니다.</span> ({cs["건수"]}건 중 {cs["귀책"]}건 상담사 귀책)')
        elif cs["QA"] < 80:
            lines.append(f'<span class="orange">{ch_tag} 채널 QA이행 {cs["QA"]}점으로 개선 필요합니다.</span> ({cs["건수"]}건)')
        else:
            lines.append(f'{ch_tag} 채널은 {cs["건수"]}건, QA이행 {cs["QA"]}점으로 양호합니다.')

    # 대분류 약점
    if worst_cat_score >= 10:
        lines.append(f'가장 취약한 영역은 <b>{worst_cat}</b>이며, 평균 <span class="red">{worst_cat_score}점 차감</span>이 발생하고 있습니다.')
    elif worst_cat_score >= 5:
        lines.append(f'<b>{worst_cat}</b> 영역에서 평균 <span class="orange">{worst_cat_score}점 차감</span>이 발생합니다.')

    # 세부항목
    if top_subs:
        sub_desc = ", ".join([f"<b>{s[0]}</b>({s[1]}건)" for s in top_subs])
        lines.append(f"반복 차감 항목: {sub_desc}.")

    # 상세분석 패턴
    if top_fb:
        fb_desc = ", ".join([f"<b>{f[0]}</b>({f[1]}회)" for f in top_fb])
        lines.append(f"상세분석에서 반복 지적되는 패턴: {fb_desc}.")

    # 불일치 케이스
    if pos_but_blame > 0:
        lines.append(f"고객이 긍정 응답했으나 상담사 귀책으로 분류된 건이 <b>{pos_but_blame}건</b> 있습니다. 해당 건은 재검토가 필요할 수 있습니다.")

    # 종합 판정
    if avg_qa < 70 and agent_blame_pct >= 50:
        lines.append(f'<br><span class="red">▶ 종합 판정: 즉각적인 1:1 코칭 및 집중 모니터링 대상입니다.</span>')
    elif avg_qa < 85 or agent_blame_pct >= 40:
        lines.append(f'<br><span class="orange">▶ 종합 판정: 주요 약점 중심으로 개선 코칭이 필요합니다.</span>')
    else:
        lines.append(f'<br>▶ 종합 판정: 현재 큰 이슈 없으나, 반복 차감 패턴에 대한 리마인드가 권장됩니다.')

    return "<br>".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. 페이지 렌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_under70():
    inject_custom_css()
    st.markdown("## 📋 70점 미만 QA 모니터링 교차분석")
    st.caption("CSAT 최종점수 70점 미만 · 모니터링 이행평가 · 귀책 · 차감사유 · 상담사 종합 프로파일")

    df_raw = load_monitoring_sheet()
    if df_raw.empty:
        st.warning("데이터를 불러올 수 없습니다.")
        return
    df = calc_deductions(df_raw)

    # ── 필터 ──
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

    mask = (df["회신월"].isin(sel_months) & df["상담사"].isin(sel_agents) &
            df["채널"].isin(sel_ch) & df["귀책분류"].isin(sel_blame))
    df_f = df[mask].copy()
    if df_f.empty:
        st.info("필터 조건에 맞는 데이터가 없습니다.")
        return

    # ══════════════════════════════════════════════════════════
    #  SECTION 1 — KPI
    # ══════════════════════════════════════════════════════════
    section_header("전체 현황", "필터 적용 기준 핵심 지표")
    total = len(df_f)
    avg_csat = df_f["최종점수"].mean()
    avg_qa = df_f["QA이행점수"].mean()
    avg_ded = df_f["QA총차감"].mean()
    ab_cnt = int((df_f["귀책분류"] == "상담사").sum())
    ab_pct = ab_cnt / total * 100 if total else 0

    render_kpi_row([
        kpi_card_html("모니터링 건수", f"{total}건", "전체 필터 적용", C_PRIMARY),
        kpi_card_html("평균 CSAT", f"{avg_csat:.1f}점",
                      "양호" if avg_csat >= 70 else "주의", C_SUCCESS if avg_csat >= 70 else C_DANGER),
        kpi_card_html("평균 QA이행", f"{avg_qa:.1f}점",
                      "양호" if avg_qa >= 85 else "개선필요", C_SUCCESS if avg_qa >= 85 else C_WARNING),
        kpi_card_html("평균 차감", f"{avg_ded:.1f}점", "낮을수록 양호", C_WARNING),
        kpi_card_html("상담사 귀책", f"{ab_pct:.1f}%", f"{ab_cnt}건 / {total}건", C_DANGER),
    ])

    # 채널별 KPI
    st.markdown("**채널별 KPI**")
    ch_cards = []
    for ch_name in sorted(df_f["채널"].unique()):
        ch_df = df_f[df_f["채널"] == ch_name]
        ch_n = len(ch_df)
        ch_csat = ch_df["최종점수"].mean()
        ch_qa = ch_df["QA이행점수"].mean()
        ch_bl = int((ch_df["귀책분류"] == "상담사").sum())
        ch_bl_p = ch_bl / ch_n * 100 if ch_n else 0
        ch_cards.append(kpi_card_html(
            f"{'📞' if '전화' in ch_name else '💬'} {ch_name}", f"{ch_n}건",
            f"CSAT {ch_csat:.1f} · QA {ch_qa:.1f} · 귀책 {ch_bl_p:.0f}%",
            ch_color(ch_name)))
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
        ch_piv = df_f.groupby(["채널", "귀책분류"]).size().unstack(fill_value=0)
        ch_piv["합계"] = ch_piv.sum(axis=1)
        for col in ch_piv.columns[:-1]:
            ch_piv[f"{col}(%)"] = (ch_piv[col] / ch_piv["합계"] * 100).round(1)
        st.dataframe(ch_piv, use_container_width=True)

    with tab_trend:
        mt = df_f.groupby(["회신월", "귀책분류"]).size().reset_index(name="건수")
        fig_t = px.line(mt, x="회신월", y="건수", color="귀책분류",
                        markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_t.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_t, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 3 — 불만사유 & 차감사유 & 상세분석
    # ══════════════════════════════════════════════════════════
    section_header("문의불만사유 & 차감사유 & 상세분석 패턴", "불만사유 Top15 · 차감사유 텍스트 · 상세분석 피드백 패턴")
    tab_reason, tab_deduct, tab_fb = st.tabs(["불만사유 Top 15", "차감사유 텍스트 빈도", "상세분석 피드백 패턴"])

    with tab_reason:
        reason = df_f[df_f["문의불만사유"] != ""]["문의불만사유"].value_counts().head(15).reset_index()
        reason.columns = ["불만사유", "건수"]
        fig3 = px.bar(reason, x="건수", y="불만사유", orientation="h",
                      color="건수", color_continuous_scale="Reds")
        fig3.update_layout(height=450, yaxis=dict(autorange="reversed"),
                           font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab_deduct:
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
            fig_sun = px.sunburst(reason_freq.head(50), path=["항목", "차감사유"], values="건수",
                                  color="건수", color_continuous_scale="YlOrRd",
                                  title="항목별 차감사유 분포")
            fig_sun.update_layout(height=500, font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("차감 사유 텍스트가 없습니다.")

    with tab_fb:
        fb_all = extract_feedback_patterns(df_f["상세분석"])
        if fb_all:
            fb_df = pd.DataFrame(fb_all.most_common(15), columns=["피드백 패턴", "빈도"])
            fig_fb = px.bar(fb_df, x="빈도", y="피드백 패턴", orientation="h",
                            color="빈도", color_continuous_scale="Purples",
                            title="상세분석 피드백 패턴 Top 15")
            fig_fb.update_layout(height=450, yaxis=dict(autorange="reversed"),
                                 font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_fb, use_container_width=True)

            # 채널별 비교
            st.markdown("**채널별 피드백 패턴 비교**")
            ch_fb_rows = []
            for ch in sorted(df_f["채널"].unique()):
                ch_fb = extract_feedback_patterns(df_f[df_f["채널"] == ch]["상세분석"])
                for pat, cnt in ch_fb.items():
                    ch_fb_rows.append({"채널": ch, "패턴": pat, "빈도": cnt})
            if ch_fb_rows:
                df_ch_fb = pd.DataFrame(ch_fb_rows)
                fig_cfb = px.bar(df_ch_fb, x="패턴", y="빈도", color="채널",
                                 barmode="group",
                                 color_discrete_map=CHANNEL_COLORS,
                                 title="채널별 피드백 패턴 비교")
                fig_cfb.update_layout(height=400, font=dict(family="Noto Sans KR"),
                                      xaxis_tickangle=-45)
                st.plotly_chart(fig_cfb, use_container_width=True)
        else:
            st.info("상세분석 데이터가 없습니다.")

    # ══════════════════════════════════════════════════════════
    #  SECTION 4 — QA 항목별 차감
    # ══════════════════════════════════════════════════════════
    section_header("QA 항목별 차감 현황", "대분류별 · 세부항목별 · 상담사×항목 히트맵 · 채널별 비교")
    tab_cat, tab_sub, tab_hm, tab_chqa = st.tabs(["대분류별", "세부항목별", "상담사×항목 히트맵", "채널별 항목 비교"])

    cat_cols = ["정확성(30)_차감","숙련도(20)_차감","친절도(30)_차감","약속이행(20)_차감"]
    with tab_cat:
        cat_means = df_f[cat_cols].mean().reset_index()
        cat_means.columns = ["항목", "평균차감"]
        cat_means["항목"] = cat_means["항목"].str.replace("_차감", "")
        fig4 = px.bar(cat_means, x="항목", y="평균차감", color="평균차감",
                      color_continuous_scale="OrRd", title="대분류별 평균 차감")
        fig4.update_layout(height=350, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig4, use_container_width=True)

    with tab_sub:
        sub_flags = [c for c in df_f.columns if c.endswith("_감점")]
        sub_counts = df_f[sub_flags].sum().reset_index()
        sub_counts.columns = ["항목", "차감건수"]
        sub_counts["항목"] = sub_counts["항목"].str.replace("_감점", "").map(lambda x: SUB_LABELS.get(x, x))
        sub_counts = sub_counts.sort_values("차감건수", ascending=False)
        fig5 = px.bar(sub_counts, x="차감건수", y="항목", orientation="h",
                      color="차감건수", color_continuous_scale="YlOrRd")
        fig5.update_layout(height=500, yaxis=dict(autorange="reversed"),
                           font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig5, use_container_width=True)

    with tab_hm:
        sub_flag_cols = [c for c in df_f.columns if c.endswith("_감점")]
        hm = df_f.groupby("상담사")[sub_flag_cols].mean().round(2) * 100
        hm.columns = [SUB_LABELS.get(c.replace("_감점",""), c.replace("_감점","")) for c in hm.columns]
        fig_hm = px.imshow(hm, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
                           title="상담사별 항목 감점율(%)",
                           labels=dict(x="QA항목", y="상담사", color="감점율(%)"))
        fig_hm.update_layout(height=max(350, len(hm)*35), font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab_chqa:
        ch_cat = df_f.groupby("채널")[cat_cols].mean().round(1)
        ch_cat.columns = [c.replace("_차감","") for c in ch_cat.columns]
        ch_long = ch_cat.reset_index().melt(id_vars="채널", var_name="항목", value_name="평균차감")
        fig_cq = px.bar(ch_long, x="채널", y="평균차감", color="항목", barmode="group",
                        color_discrete_sequence=[C_DANGER, C_WARNING, C_PRIMARY, C_SUCCESS],
                        title="채널별 대분류 평균 차감 비교")
        fig_cq.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_cq, use_container_width=True)

        ch_sub_hm = df_f.groupby("채널")[sub_flag_cols].mean().round(2) * 100
        ch_sub_hm.columns = [SUB_LABELS.get(c.replace("_감점",""), c.replace("_감점","")) for c in ch_sub_hm.columns]
        fig_csh = px.imshow(ch_sub_hm, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
                            title="채널별 세부항목 감점율(%)",
                            labels=dict(x="QA항목", y="채널", color="감점율(%)"))
        fig_csh.update_layout(height=max(250, len(ch_sub_hm)*50), font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_csh, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 5 — 고객 코멘트(Q3) vs 귀책
    # ══════════════════════════════════════════════════════════
    section_header("고객 코멘트(Q3) vs 실제 귀책 교차분석",
                   "고객 긍정 → 상담사 귀책? 부정 → 고객/IBR 귀책? 불일치 케이스")
    tab_mis, tab_sent = st.tabs(["불일치 케이스", "긍정/부정 × 귀책"])

    with tab_mis:
        pos_ag = df_f[(df_f["긍정부정"] == "긍정") & (df_f["귀책분류"] == "상담사")]
        neg_na = df_f[(df_f["긍정부정"] == "부정") & (df_f["귀책분류"].isin(["고객", "IBR"]))]
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**고객 긍정 → 상담사 귀책**: {len(pos_ag)}건")
            if not pos_ag.empty:
                st.dataframe(pos_ag[["상담사","채널","최종점수","Q3","귀책분류","문의불만사유","상세분석"]].head(20),
                             use_container_width=True, hide_index=True, height=300)
        with mc2:
            st.markdown(f"**고객 부정 → 고객/IBR 귀책**: {len(neg_na)}건")
            if not neg_na.empty:
                st.dataframe(neg_na[["상담사","채널","최종점수","Q3","귀책분류","문의불만사유","상세분석"]].head(20),
                             use_container_width=True, hide_index=True, height=300)

    with tab_sent:
        cs = df_f.groupby(["긍정부정", "귀책분류"]).size().reset_index(name="건수")
        fig_cs = px.bar(cs, x="귀책분류", y="건수", color="긍정부정", barmode="group",
                        color_discrete_map={"긍정":C_SUCCESS,"부정":C_DANGER,"기타":C_WARNING},
                        title="고객 긍정/부정 × 귀책 분류")
        fig_cs.update_layout(height=400, font=dict(family="Noto Sans KR"))
        st.plotly_chart(fig_cs, use_container_width=True)
        ss = df_f.groupby("긍정부정").agg(건수=("상담이력KEY","count"),
                                        평균_CSAT=("최종점수","mean"),
                                        평균_QA이행=("QA이행점수","mean")).round(1).reset_index()
        st.dataframe(ss, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 6 — 상담사별 종합 프로파일 ★핵심★
    # ══════════════════════════════════════════════════════════
    section_header("상담사별 종합 프로파일",
                   "성과 테이블 · 종합평가 리포트 · 레이더 · 채널별 취약점 · 피드백 패턴 · 코칭 포인트")

    agent_agg = (
        df_f.groupby("상담사")
        .agg(건수=("상담이력KEY","count"),
             평균CSAT=("최종점수","mean"), 평균QA이행=("QA이행점수","mean"),
             평균차감=("QA총차감","mean"),
             정확성=("정확성(30)_차감","mean"), 숙련도=("숙련도(20)_차감","mean"),
             친절도=("친절도(30)_차감","mean"), 약속이행=("약속이행(20)_차감","mean"),
             상담사귀책=("귀책분류", lambda x: (x=="상담사").sum()),
             IBR귀책=("귀책분류", lambda x: (x=="IBR").sum()),
             고객귀책=("귀책분류", lambda x: (x=="고객").sum()))
        .round(1).sort_values("평균QA이행").reset_index()
    )
    agent_agg["상담사귀책율(%)"] = (agent_agg["상담사귀책"]/agent_agg["건수"]*100).round(1)
    st.dataframe(agent_agg, use_container_width=True, hide_index=True)

    sel_agent = st.selectbox("🔎 상담사 상세 보기", agent_agg["상담사"].tolist(), key="u70_agent_sel")
    if sel_agent:
        df_agent = df_f[df_f["상담사"] == sel_agent]
        row = agent_agg[agent_agg["상담사"] == sel_agent].iloc[0]

        # ── 종합평가 리포트 (자연어) ──
        summary_html = generate_agent_summary(sel_agent, df_agent)
        st.markdown(f"""
        <div class="profile-wrap">
            <div class="profile-name">👤 {sel_agent} — 종합 평가 리포트</div>
            <div class="summary-box">{summary_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 레이더 + 귀책 파이 ──
        ac1, ac2 = st.columns([1, 1])
        with ac1:
            cats = ["정확성(30)","숙련도(20)","친절도(30)","약속이행(20)"]
            deducts = [row["정확성"],row["숙련도"],row["친절도"],row["약속이행"]]
            maxv = [30,20,30,20]
            earned = [m-d for m,d in zip(maxv, deducts)]
            fig6 = go.Figure()
            fig6.add_trace(go.Scatterpolar(
                r=earned+[earned[0]], theta=cats+[cats[0]],
                fill="toself", name=sel_agent, line_color=C_PRIMARY,
                fillcolor="rgba(99,102,241,0.15)"))
            fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,30])),
                               title=f"{sel_agent} — 항목별 획득 점수", height=380,
                               font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig6, use_container_width=True)

        with ac2:
            ab = df_agent["귀책분류"].value_counts().reset_index()
            ab.columns = ["귀책분류","건수"]
            fig_ab = px.pie(ab, names="귀책분류", values="건수",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            title=f"{sel_agent} — 귀책 분포")
            fig_ab.update_layout(height=380, font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_ab, use_container_width=True)

        # ── 채널별 성과 비교 ──
        st.markdown(f"**{sel_agent} — 채널별 성과 비교**")
        ch_rows = []
        for ch in df_agent["채널"].unique():
            ch_d = df_agent[df_agent["채널"] == ch]
            ch_n = len(ch_d)
            ch_rows.append({
                "채널": ch, "건수": ch_n,
                "평균CSAT": round(ch_d["최종점수"].mean(), 1),
                "평균QA이행": round(ch_d["QA이행점수"].mean(), 1),
                "상담사귀책": int((ch_d["귀책분류"]=="상담사").sum()),
                "귀책율(%)": round((ch_d["귀책분류"]=="상담사").sum()/ch_n*100, 1) if ch_n else 0,
            })
        df_chp = pd.DataFrame(ch_rows)
        st.dataframe(df_chp, use_container_width=True, hide_index=True)

        if len(df_chp) > 1:
            fig_chp = px.bar(df_chp, x="채널", y=["평균CSAT","평균QA이행"],
                             barmode="group", title=f"{sel_agent} — 채널별 CSAT vs QA이행",
                             color_discrete_map={"평균CSAT":C_DANGER,"평균QA이행":C_PRIMARY})
            fig_chp.update_layout(height=350, font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_chp, use_container_width=True)

        # ── 이 상담사의 피드백 패턴 ──
        st.markdown(f"**{sel_agent} — 상세분석 피드백 패턴**")
        agent_fb = extract_feedback_patterns(df_agent["상세분석"])
        if agent_fb:
            afb_df = pd.DataFrame(agent_fb.most_common(10), columns=["패턴","빈도"])
            fig_afb = px.bar(afb_df, x="빈도", y="패턴", orientation="h",
                             color="빈도", color_continuous_scale="Reds",
                             title=f"{sel_agent} 반복 지적 패턴")
            fig_afb.update_layout(height=350, yaxis=dict(autorange="reversed"),
                                  font=dict(family="Noto Sans KR"))
            st.plotly_chart(fig_afb, use_container_width=True)
        else:
            st.info("피드백 패턴 없음")

        # ── 차감사유 상세 ──
        st.markdown(f"**{sel_agent} — 차감사유 상세**")
        sub_cols = list(SUB_LABELS.keys())
        agent_reasons = []
        for _, r in df_agent.iterrows():
            for col in sub_cols:
                val = str(r.get(col, "")).strip()
                if val:
                    agent_reasons.append({"항목":SUB_LABELS.get(col,col),"차감사유":val,
                                          "귀책":r.get("귀책분류",""),"최종점수":r.get("최종점수","")})
        if agent_reasons:
            df_ar = pd.DataFrame(agent_reasons)
            ar_freq = df_ar.groupby(["항목","차감사유"]).size().reset_index(name="건수")
            st.dataframe(ar_freq.sort_values("건수",ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("차감 사유 없음")

        # ── 상세분석 원문 ──
        with st.expander(f"📝 {sel_agent} — 상세분석 원문 전체"):
            at = df_agent[df_agent["상세분석"]!=""][["회신월","채널","최종점수","귀책분류","상세분석"]].sort_values("회신월")
            if not at.empty:
                st.dataframe(at, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("상세분석 데이터 없음")

        # ── 건별 상세 ──
        st.markdown(f"**{sel_agent} — 건별 상세**")
        det_cols = ["회신월","채널","사업자","브랜드","최종점수","QA이행점수",
                    "귀책분류","문의불만사유","긍정부정","Q3","상세분석","피드백여부"]
        ex = [c for c in det_cols if c in df_agent.columns]
        st.dataframe(df_agent[ex].sort_values("최종점수"),
                     use_container_width=True, hide_index=True, height=400)

    # ══════════════════════════════════════════════════════════
    #  SECTION 7 — 귀책 × CSAT × QA이행 교차
    # ══════════════════════════════════════════════════════════
    section_header("귀책 × CSAT × QA이행 교차", "귀책 유형별 평균 점수 · 산점도")
    cross = (df_f.groupby("귀책분류")
             .agg(건수=("상담이력KEY","count"), 평균CSAT=("최종점수","mean"),
                  평균QA이행=("QA이행점수","mean"), 평균차감=("QA총차감","mean"))
             .round(1).sort_values("건수",ascending=False).reset_index())
    st.dataframe(cross, use_container_width=True, hide_index=True)

    fig7 = px.scatter(df_f, x="최종점수", y="QA이행점수", size="QA총차감",
                      color="귀책분류", hover_data=["상담사","채널","문의불만사유","Q3"],
                      title="CSAT vs QA이행 (버블=차감, 색=귀책)",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig7.update_layout(height=500, font=dict(family="Noto Sans KR"))
    st.plotly_chart(fig7, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 8 — 상세 데이터
    # ══════════════════════════════════════════════════════════
    section_header("전체 상세 데이터", "필터 적용 전체 데이터 (QA이행점수 오름차순)")
    show = ["회신월","발송일자","상담사","채널","사업자","브랜드","상담유형대","긍정부정",
            "최종점수","친절점수","만족점수","귀책분류","문의불만사유","QA이행점수","QA총차감",
            "정확성(30)_차감","숙련도(20)_차감","친절도(30)_차감","약속이행(20)_차감",
            "Q3","상세분석","피드백여부"]
    ex = [c for c in show if c in df_f.columns]
    st.dataframe(df_f[ex].sort_values("QA이행점수"),
                 use_container_width=True, hide_index=True, height=500)


# ── csat.py 호환 ──
page_under70_analysis = page_under70
