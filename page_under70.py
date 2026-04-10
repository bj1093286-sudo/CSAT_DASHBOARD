# ──────────────────────────────────────────────────────────────
#  page_under70.py
#  70점 미만 QA 모니터링 교차분석 페이지 (v2 - 디벨롭)
# ──────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# ── 색상 토큰 ─────────────────────────────────────────────────
C_PRIMARY = "#6366f1"
C_SUCCESS = "#22c55e"
C_WARNING = "#f59e0b"
C_DANGER  = "#ef4444"
C_BG_CARD = "#ffffff"
C_BORDER  = "#e2e8f0"
C_TEXT    = "#1e293b"
C_TEXT_SUB = "#64748b"

SHEET_ID = "1ujtxIKZJRR9vIC1TS5GWWEtM9luChlDJk4NwTqeYB2Q"
GID      = 2055211445

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

# 서브항목 한글 라벨
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
#  3. 페이지 렌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def page_under70():
    st.markdown("## 70점 미만 QA 모니터링 교차분석")
    st.caption("CSAT 최종점수 70점 미만 · 모니터링 이행평가 · 귀책 · 차감사유 · 고객코멘트 교차분석")

    df_raw = load_monitoring_sheet()
    if df_raw.empty:
        st.warning("데이터를 불러올 수 없습니다.")
        return

    df = calc_deductions(df_raw)

    # ── 필터 ──────────────────────────────────────────────────
    with st.expander("필터 설정", expanded=True):
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
    #  SECTION 1 — 전체 KPI
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 전체 현황")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("모니터링 건수", f"{len(df_f)}건")
    k2.metric("평균 CSAT", f"{df_f['최종점수'].mean():.1f}")
    k3.metric("평균 QA이행", f"{df_f['QA이행점수'].mean():.1f}")
    k4.metric("평균 차감", f"{df_f['QA총차감'].mean():.1f}")
    # 상담사귀책 비율
    agent_blame_cnt = len(df_f[df_f["귀책분류"] == "상담사"])
    agent_blame_pct = agent_blame_cnt / len(df_f) * 100 if len(df_f) > 0 else 0
    k5.metric("상담사귀책 비율", f"{agent_blame_pct:.1f}%", f"{agent_blame_cnt}건")

    # ══════════════════════════════════════════════════════════
    #  SECTION 2 — 귀책 분류
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 귀책 분류 현황")
    tab_all, tab_ch, tab_trend = st.tabs(["전체 비중", "채널별", "월별 추이"])

    with tab_all:
        blame = df_f["귀책분류"].value_counts().reset_index()
        blame.columns = ["귀책분류", "건수"]
        blame["비율(%)"] = (blame["건수"] / blame["건수"].sum() * 100).round(1)
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(blame, names="귀책분류", values="건수",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(blame, use_container_width=True, hide_index=True)

    with tab_ch:
        blame_ch = df_f.groupby(["채널", "귀책분류"]).size().reset_index(name="건수")
        fig2 = px.bar(blame_ch, x="채널", y="건수", color="귀책분류",
                      barmode="group", color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_trend:
        mt = df_f.groupby(["회신월", "귀책분류"]).size().reset_index(name="건수")
        fig_t = px.line(mt, x="회신월", y="건수", color="귀책분류",
                        markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_t.update_layout(height=400)
        st.plotly_chart(fig_t, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 3 — 불만사유 & 차감사유 텍스트 상세
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 문의불만사유 & 차감 사유 상세")
    tab_reason, tab_deduct_detail = st.tabs(["불만사유 Top 15", "차감 사유 텍스트 빈도"])

    with tab_reason:
        reason = df_f[df_f["문의불만사유"] != ""]["문의불만사유"].value_counts().head(15).reset_index()
        reason.columns = ["불만사유", "건수"]
        fig3 = px.bar(reason, x="건수", y="불만사유", orientation="h",
                      color="건수", color_continuous_scale="Reds")
        fig3.update_layout(height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab_deduct_detail:
        # 각 서브항목에 적힌 텍스트(차감 사유)를 모아서 빈도 분석
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

            # 항목별 차감사유 sunburst
            fig_sun = px.sunburst(
                reason_freq.head(50), path=["항목", "차감사유"], values="건수",
                color="건수", color_continuous_scale="YlOrRd",
                title="항목별 차감사유 분포"
            )
            fig_sun.update_layout(height=500)
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.info("차감 사유 텍스트가 없습니다.")

    # ══════════════════════════════════════════════════════════
    #  SECTION 4 — QA 항목별 차감
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### QA 항목별 차감 현황")
    tab_cat, tab_sub, tab_heatmap = st.tabs(["대분류별", "세부항목별", "상담사 × 항목 히트맵"])

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
        sub_counts["항목"] = sub_counts["항목"].str.replace("_감점", "").map(
            lambda x: SUB_LABELS.get(x, x)
        )
        sub_counts = sub_counts.sort_values("차감건수", ascending=False)
        fig5 = px.bar(sub_counts, x="차감건수", y="항목", orientation="h",
                      color="차감건수", color_continuous_scale="YlOrRd")
        fig5.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, use_container_width=True)

    with tab_heatmap:
        # 상담사 × 세부항목 감점율 히트맵
        sub_flag_cols = [c for c in df_f.columns if c.endswith("_감점")]
        hm = df_f.groupby("상담사")[sub_flag_cols].mean().round(2) * 100
        hm.columns = [SUB_LABELS.get(c.replace("_감점", ""), c.replace("_감점", "")) for c in hm.columns]
        fig_hm = px.imshow(
            hm, text_auto=".0f", aspect="auto",
            color_continuous_scale="YlOrRd",
            title="상담사별 항목 감점율 (%)",
            labels=dict(x="QA 항목", y="상담사", color="감점율(%)"),
        )
        fig_hm.update_layout(height=max(350, len(hm) * 35))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 5 — 고객 코멘트(Q3) vs 귀책 교차분석
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 고객 코멘트(Q3) vs 실제 귀책 교차분석")
    st.caption("고객이 긍정적으로 응답했는데 상담사 귀책? 부정인데 IBR/고객 귀책? 같은 불일치 케이스 분석")

    tab_mismatch, tab_sentiment = st.tabs(["불일치 케이스", "긍정/부정 × 귀책"])

    with tab_mismatch:
        # 긍정인데 상담사 귀책
        pos_agent = df_f[(df_f["긍정부정"] == "긍정") & (df_f["귀책분류"] == "상담사")]
        # 부정인데 고객/IBR 귀책
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
                        barmode="group", color_discrete_map={"긍정": C_SUCCESS, "부정": C_DANGER, "기타": C_WARNING},
                        title="고객 긍정/부정 × 귀책 분류")
        fig_cs.update_layout(height=400)
        st.plotly_chart(fig_cs, use_container_width=True)

        # 긍정/부정별 평균 점수 비교
        sent_score = df_f.groupby("긍정부정").agg(
            건수=("상담이력KEY", "count"),
            평균_CSAT=("최종점수", "mean"),
            평균_QA이행=("QA이행점수", "mean"),
        ).round(1).reset_index()
        st.dataframe(sent_score, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 6 — 상담사별 상세 성과
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 상담사별 QA 상세 성과")

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
    sel_agent = st.selectbox("상담사 상세 보기", agent_agg["상담사"].tolist(), key="u70_agent_sel")
    if sel_agent:
        df_agent = df_f[df_f["상담사"] == sel_agent]
        row = agent_agg[agent_agg["상담사"] == sel_agent].iloc[0]

        ac1, ac2 = st.columns([1, 1])
        with ac1:
            # 레이더 차트 — 획득 점수
            cats = ["정확성(30)", "숙련도(20)", "친절도(30)", "약속이행(20)"]
            deducts = [row["정확성"], row["숙련도"], row["친절도"], row["약속이행"]]
            max_vals = [30, 20, 30, 20]
            earned = [m - d for m, d in zip(max_vals, deducts)]

            fig6 = go.Figure()
            fig6.add_trace(go.Scatterpolar(
                r=earned + [earned[0]],
                theta=cats + [cats[0]],
                fill="toself", name=sel_agent, line_color=C_PRIMARY,
            ))
            fig6.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
                title=f"{sel_agent} — 항목별 획득 점수", height=380,
            )
            st.plotly_chart(fig6, use_container_width=True)

        with ac2:
            # 이 상담사의 귀책 분포
            ab = df_agent["귀책분류"].value_counts().reset_index()
            ab.columns = ["귀책분류", "건수"]
            fig_ab = px.pie(ab, names="귀책분류", values="건수",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            title=f"{sel_agent} — 귀책 분포")
            fig_ab.update_layout(height=380)
            st.plotly_chart(fig_ab, use_container_width=True)

        # 이 상담사의 차감 사유 상세
        st.markdown(f"**{sel_agent}의 차감 사유 상세**")
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

        # 이 상담사의 건별 상세
        st.markdown(f"**{sel_agent} — 건별 상세**")
        detail_cols = ["회신월", "채널", "사업자", "브랜드", "최종점수", "QA이행점수",
                       "귀책분류", "문의불만사유", "긍정부정", "Q3", "상세분석", "피드백여부"]
        ex_cols = [c for c in detail_cols if c in df_agent.columns]
        st.dataframe(
            df_agent[ex_cols].sort_values("최종점수"),
            use_container_width=True, hide_index=True, height=400
        )

    # ══════════════════════════════════════════════════════════
    #  SECTION 7 — 귀책 × CSAT 교차 산점도
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 귀책 × CSAT × QA이행 교차")
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
    fig7.update_layout(height=500)
    st.plotly_chart(fig7, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  SECTION 8 — 상세 데이터 테이블
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 전체 상세 데이터")
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
