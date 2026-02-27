import re
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_gsheets import GSheetsConnection


# ══════════════════════════════════════════════════════════════════
# 0. 전역 상수 & 색상 팔레트 (기존 유지)
# ══════════════════════════════════════════════════════════════════
FONT_NAME  = "Noto Sans KR"
C_NAVY     = "#1E3A5F"
C_BLUE     = "#2563EB"
C_BLUE_LT  = "#DBEAFE"
C_GREEN    = "#059669"
C_GREEN_LT = "#D1FAE5"
C_AMBER    = "#D97706"
C_AMBER_LT = "#FEF3C7"
C_RED      = "#DC2626"
C_RED_LT   = "#FEE2E2"
C_GRAY     = "#6B7280"
C_GRAY_LT  = "#F3F4F6"
C_WHITE    = "#FFFFFF"
C_BORDER   = "#CBD5E1"
C_CYAN     = "#0891B2"

CHART_COLORS = [
    "#2563EB","#059669","#D97706","#7C3AED",
    "#0891B2","#DB2777","#65A30D","#EA580C",
    "#0284C7","#DC2626","#16A34A","#9333EA",
]

SCORE_GOOD    = 90
SCORE_CAUTION = 70

EXCLUDED_AGENTS = {"엄소라","이은덕","한인경","양현정","이혜선","박성주"}


# ══════════════════════════════════════════════════════════════════
# 1. 날짜 파싱 (기존 유지)
# ══════════════════════════════════════════════════════════════════
def parse_date_series(series: pd.Series) -> pd.Series:
    def try_parse(val):
        if pd.isna(val):
            return pd.NaT
        if isinstance(val, (pd.Timestamp, datetime)):
            return pd.Timestamp(val)

        s = str(val).strip().replace(" ","")
        if s in ("","nan","NaT","None","NaN"):
            return pd.NaT

        try:
            num = float(s)
            if 40000 < num < 60000:
                return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(num))
            if 20000101 <= num <= 20991231:
                return pd.to_datetime(str(int(num)), format="%Y%m%d", errors="coerce")
            if 200001 <= num <= 209912:
                return pd.to_datetime(str(int(num)), format="%Y%m", errors="coerce")
        except (ValueError, TypeError):
            pass

        for fmt in ["%Y-%m-%d","%Y/%m/%d","%Y.%m.%d","%Y-%m","%Y/%m",
                    "%Y.%m","%Y%m%d","%Y%m","%m/%d/%Y","%d/%m/%Y",
                    "%Y년%m월%d일","%Y년%m월"]:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue

        try:
            return pd.to_datetime(s, infer_datetime_format=True, errors="coerce")
        except Exception:
            return pd.NaT

    return series.apply(try_parse)


def extract_ym(series: pd.Series) -> pd.Series:
    dt = parse_date_series(series)
    return dt.dt.strftime("%Y-%m").where(dt.notna(), other=pd.NA)


# ══════════════════════════════════════════════════════════════════
# 2. 감성 분류 (기존 유지)
# ══════════════════════════════════════════════════════════════════
POS_WORDS = [
    "감사합니다","고맙습니다","감사드립니다","정말감사","너무감사","진심감사",
    "친절하게","친절히","친절했","친절합니다","너무친절","정말친절","매우친절",
    "만족스럽","만족했","만족합니다","만족해요","대만족","완전만족","매우만족",
    "빠르게","신속하게","신속히","빠른처리","빠른답변","빠른해결","신속한처리",
    "해결됐","해결되었","해결해주셔서","처리완료","잘해결","깔끔하게해결",
    "훌륭한응대","좋은응대","정확한안내","상세한안내","친절한안내",
    "전문적으로","전문성있게","능숙하게","완벽하게","완벽해요",
    "너무좋아","정말좋아","정말좋은","너무좋은","최선을다해","열정적으로",
    "칭찬드려요","칭찬합니다","도움됐","도움이됐","많은도움","큰도움",
    "도움이되었","도움되었습니다","재이용할게요","또이용하겠","다시이용하겠",
    "추천합니다","적극추천","강추","감동받았","인상깊었","인상좋았",
    "정확하게","정확히","상세하게","자세하게","꼼꼼하게","꼼꼼히",
    "편리해요","편리합니다","간편해요","수월하게","정중하게","매너좋은",
    "예의바르게","친근하게","따뜻하게","따뜻한응대","믿을수있는","믿음직해요",
    "안심됐","안심이됩니다","깔끔하게","원활하게","문제없이","이상없이",
    "잘해주셔서","잘처리","잘안내","잘설명","수고하셨습니다","수고많으셨",
    "노력해주셔서","신경써주셔서","배려해주셔서","이해해주셔서","공감해주셔서",
    "잘들어주셔서","기분좋았","기분이좋았","덕분에해결","덕분에도움","덕분입니다",
    "잘처리됐","잘처리되었","정상처리","이해하기쉽게","알기쉽게",
    "좋았습니다","잘됐습니다","감사했습니다","편했습니다",
]
NEG_WORDS = [
    "불만","불쾌","불친절","불성실","불신",
    "화나","화났","짜증","짜증나","짜증났","짜증스럽","열받았","어이없","황당","기가막혀",
    "최악","형편없","엉망","엉터리","별로였","나빴","실망했","실망스럽","기대이하",
    "지연됐","지연되었","늦게","너무오래","한참기다렸","배송늦어","배송지연","아직안왔",
    "해결안됐","해결이안","미해결","처리안됐","안됐어요","안되네요","해결못","처리못",
    "방치됐","무시당했","나몰라라","오류발생","에러났","잘못처리","잘못안내",
    "잘못된정보","틀린정보","오안내","착오났","또전화","재문의","몇번이나","여러번",
    "몇번씩","계속문의","계속전화","또연락","무례하게","무례한","퉁명스럽","냉정",
    "차가운응대","딱딱한응대","고압적","강요했","잘모르는","전문성없어","비전문",
    "허술한","믿을수없어","의심스럽","사기같아","거짓말","허위안내","책임안져",
    "나몰라식","떠넘겼","전가했","연결안돼","설명이부족","답변부실","엉뚱한답변",
    "환불거부","환불안해","취소거부","환불지연","환불못받아","다시는이용안해",
    "절대추천안해","비추천","잠깐만요","잠깐만기다려","잠시만요","잠시만기다려",
    "조금만기다려","말이달라","말을바꿔","앞에서는","전에는된다고","다르게말했",
    "모순됐","일관성없","혼란스럽","헷갈려","혼동됐","설명이다달라","매번달라",
    "어제협의했는데","이미교환하기로","이미환불하기로","다시증빙","또증빙","재제출",
    "전체반품","전부보내","불필요하게","과도한요구",
]
NEGATION_PATTERNS = [
    r"불편함?\s*없",r"불만\s*없",r"문제\s*없",r"걱정\s*없",
    r"어렵지\s*않",r"나쁘지\s*않",r"부족하지\s*않",
    r"아쉽지\s*않",r"실망하지\s*않",r"불편하지\s*않",
    r"늦지\s*않",r"느리지\s*않",
    r"(?:전혀|하나도)\s*(?:불편|불만|문제)",
    r"전혀\s*없",r"하나도\s*없",
]
NEG_PATTERNS = [
    r"(?:잠깐만|잠시만|조금만).{0,10}(?:요|요\.)",
    r"(?:기다려|대기).{0,5}(?:달라|주세요|요).{0,20}(?:또|다시|반복|몇\s*번)",
    r"(?:또|다시|계속|반복).{0,10}(?:기다려|대기)",
    r"(?:처음|아까|어제|전날|이전|저번).{0,15}(?:다르|달라|바꿔|변경)",
    r"(?:말|설명|안내).{0,10}(?:달라|다르|바뀌|바꿔|모순|틀려)",
    r"(?:상담사|직원).{0,10}(?:마다|마다.{0,5})(?:달라|다르|다른)",
    r"(?:어제|전날|이미|아까).{0,10}(?:합의|협의|약속|확인).{0,15}(?:또|다시|추가)",
    r"(?:이미|전에).{0,10}(?:보냈|제출|드렸).{0,15}(?:또|다시|재|추가)",
    r"(?:드로퍼|일부|부분).{0,10}(?:인데|인데도).{0,15}(?:전체|전부|본체|다)",
    r"(?:전체|전부|본체).{0,10}(?:보내|반품|가져).{0,10}(?:달라|요|하라)",
]


def classify_sentiment(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return None

    t = re.sub(r"\s+","",text.strip())
    for pat in NEG_PATTERNS:
        if re.search(pat, text.strip()):
            return "부정"

    negated   = any(re.search(p, t) for p in NEGATION_PATTERNS)
    pos_count = sum(1 for w in POS_WORDS if w in t)
    neg_count = 0 if negated else sum(1 for w in NEG_WORDS if w in t)

    if neg_count == 0 and pos_count == 0:
        short_pos = ["감사합니다","고맙습니다","잘됐어요","잘됐습니다","해결됐어요",
                     "완료됐습니다","수고하셨습니다","잘부탁드립니다"]
        return "긍정" if any(w in t for w in short_pos) else "중립"

    return "부정" if neg_count > pos_count else "긍정"


def classify_with_score(text: str, score) -> str:
    base = classify_sentiment(text)
    if base is None:
        return None
    try:
        if float(score) <= 90 and base == "긍정":
            return "중립"
    except (TypeError, ValueError):
        pass
    return base


def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    col       = next((c for c in ["주관식","verbatim","Q3","의견"] if c in df.columns), None)
    score_col = "최종점수" if "최종점수" in df.columns else None
    if col:
        if score_col:
            df["긍정부정"] = df.apply(lambda r: classify_with_score(r[col], r[score_col]), axis=1)
        else:
            df["긍정부정"] = df[col].apply(classify_sentiment)
    else:
        df["긍정부정"] = None
    return df


# ══════════════════════════════════════════════════════════════════
# 3. 컬럼 매핑 (기존 유지)
# ══════════════════════════════════════════════════════════════════
COL_MAP = {
    "회신월":"회신월","발송월":"발송월",
    "회신일자":"회신일자","발송일자":"발송일자",
    "사업자":"사업자","브랜드":"브랜드",
    "채널":"채널","상담사":"상담사",
    "입사일":"입사일","상담사근속":"근속",
    "상담유형(대)":"상담유형대","상담유형(중)":"상담유형중",
    "상담유형(소)":"상담유형소",
    "긍정/부정":"감성_원본","유형":"유형",
    "총합":"총합","Q1":"Q1","Q2":"Q2","Q3":"주관식",
    "친절점수":"친절점수","만족점수":"만족점수",
    "최종점수":"최종점수","만족율(건)":"만족율",
    "회신주차":"회신주차","발송주차":"발송주차",
    "WK":"WK","상담KEY":"상담KEY",
    "문의유형":"문의유형","키워드":"키워드",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True).copy()
    df.columns = [
        str(c).strip().replace("\n","").replace("\r","")
                     .replace(" ","").replace('"',"").replace("'","")
        for c in df.columns
    ]
    rename = {}
    for orig, std in COL_MAP.items():
        orig_c = orig.replace(" ","").replace("\n","")
        for col in df.columns:
            if col == orig_c or orig_c in col:
                if col not in rename:
                    rename[col] = std
                break
    df = df.rename(columns=rename)

    fallback = {
        "친절점수":  ["친절","Q1점수"],
        "만족점수":  ["만족","Q2점수"],
        "최종점수":  ["최종","final"],
        "주관식":    ["Q3","verbatim","의견"],
        "상담사":    ["agent","담당자"],
        "브랜드":    ["brand"],
        "채널":      ["channel","매체"],
        "근속":      ["근속","tenure"],
        "입사일":    ["입사"],
        "상담유형대":["대분류","유형대"],
        "문의유형":  ["문의","inquiry"],
        "회신주차":  ["회신주차","reply_week","replyweek"],
        "발송주차":  ["발송주차","send_week","sendweek"],
    }
    for std, kws in fallback.items():
        if std not in df.columns:
            for col in df.columns:
                if any(kw in col for kw in kws):
                    df = df.rename(columns={col: std})
                    break

    return df


# ══════════════════════════════════════════════════════════════════
# 4. 데이터 정제 (기존 + 근속 fallback 추가)
# ══════════════════════════════════════════════════════════════════
def build_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 숫자 변환
    for col in ["친절점수","만족점수","최종점수","총합","Q1","Q2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 날짜 파싱
    for dcol in ["회신일자","발송일자","입사일"]:
        if dcol in df.columns:
            df[dcol] = parse_date_series(df[dcol])

    # 월 생성
    def make_month_col(col_name):
        if col_name in df.columns:
            parsed = extract_ym(df[col_name])
            if parsed.notna().sum() > 0:
                return parsed
        date_col = "회신일자" if "회신" in col_name else "발송일자"
        if date_col in df.columns and df[date_col].notna().sum() > 0:
            return df[date_col].dt.strftime("%Y-%m").where(df[date_col].notna(), pd.NA)
        return pd.Series(["미확인"] * len(df))

    df["발송월_정제"] = make_month_col("발송월")
    df["회신월_정제"] = make_month_col("회신월")

    # 회신일 / 발송일
    if "회신일자" in df.columns:
        df["회신일자_정제"] = parse_date_series(df["회신일자"])
        df["회신일"] = df["회신일자_정제"].dt.strftime("%Y-%m-%d").where(df["회신일자_정제"].notna(), pd.NA)
    else:
        df["회신일자_정제"] = pd.NaT
        df["회신일"] = pd.NA

    if "발송일자" in df.columns:
        df["발송일자_정제"] = parse_date_series(df["발송일자"])
        df["발송일"] = df["발송일자_정제"].dt.strftime("%Y-%m-%d").where(df["발송일자_정제"].notna(), pd.NA)
    else:
        df["발송일자_정제"] = pd.NaT
        df["발송일"] = pd.NA

    # 주차
    def make_week_str(ts):
        if pd.isna(ts):
            return pd.NA
        iso = ts.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    if "회신주차" in df.columns:
        df["회신주차_정제"] = df["회신주차"].astype(str).str.strip().replace(
            {"nan": pd.NA, "None": pd.NA, "NaT": pd.NA, "": pd.NA, "NaN": pd.NA}
        )
    else:
        df["회신주차_정제"] = df["회신일자_정제"].apply(make_week_str)

    if "발송주차" in df.columns:
        df["발송주차_정제"] = df["발송주차"].astype(str).str.strip().replace(
            {"nan": pd.NA, "None": pd.NA, "NaT": pd.NA, "": pd.NA, "NaN": pd.NA}
        )
    else:
        df["발송주차_정제"] = df["발송일자_정제"].apply(make_week_str)

    # 채널 구분
    if "채널" in df.columns:
        def norm_ch(v):
            v = str(v).strip()
            if any(x in v for x in ["채팅","chat","Chat","CHAT"]):
                return "채팅"
            if any(x in v for x in ["전화","phone","Phone","IB","인바운드","콜"]):
                return "전화IN"
            return v
        df["채널_구분"] = df["채널"].apply(norm_ch)
    else:
        df["채널_구분"] = "전체"

    # ── NEW: 근속 fallback 생성 ───────────────────────────────
    # 우선순위: (1) 이미 '근속' 컬럼 있으면 그대로 사용
    #         (2) 없고 '입사일' 있으면 근속개월 + 근속(구간) 생성
    if "근속" not in df.columns:
        if "입사일" in df.columns:
            ref = df["회신일자_정제"]
            # 회신일이 비어있으면 오늘 기준
            today = pd.Timestamp(datetime.now().date())
            ref = ref.where(ref.notna(), other=today)

            join = df["입사일"]
            # 입사일이 NaT면 근속 계산 불가
            months = np.where(
                join.notna() & ref.notna(),
                (ref.dt.year - join.dt.year) * 12 + (ref.dt.month - join.dt.month),
                np.nan
            )
            df["근속개월"] = pd.to_numeric(months, errors="coerce")

            def bucket(m):
                if pd.isna(m):
                    return pd.NA
                m = float(m)
                if m < 3:
                    return "0~3개월"
                if m < 6:
                    return "3~6개월"
                if m < 12:
                    return "6~12개월"
                if m < 24:
                    return "1~2년"
                return "2년+"

            df["근속"] = pd.Series(df["근속개월"]).apply(bucket)
        else:
            df["근속"] = pd.NA
    else:
        # 기존 데이터에 근속이 있으면 숫자도 만들어두고 싶다면(선택)
        # df["근속개월"] = pd.NA
        pass
    # ─────────────────────────────────────────────────────────

    return df


def split_active_and_scored(df: pd.DataFrame):
    MISSING = {"#N/A","nan","NaT","None","NA","N/A","","#REF!","NaN","<NA>"}
    df_all = df.copy()

    retired_agents = set()
    if "입사일" in df.columns:
        is_missing = df["입사일"].astype(str).str.strip().isin(MISSING)
        if "상담사" in df.columns:
            retired_agents = set(df[is_missing]["상담사"].dropna().astype(str).unique())
        df_active = df[~is_missing].copy()
    else:
        df_active = df.copy()

    if "최종점수" in df_active.columns:
        df_scored = df_active[df_active["최종점수"].notna() & (df_active["최종점수"] > 0)].copy()
    else:
        df_scored = df_active.copy()
    df_scored = add_sentiment_column(df_scored)

    if "최종점수" in df_all.columns:
        df_scored_all = df_all[df_all["최종점수"].notna() & (df_all["최종점수"] > 0)].copy()
    else:
        df_scored_all = df_all.copy()
    df_scored_all = add_sentiment_column(df_scored_all)

    available_months = sorted([
        m for m in df_scored["회신월_정제"].dropna().unique().tolist()
        if m != "미확인"
    ])

    return df_all, df_active, df_scored, df_scored_all, available_months, retired_agents


# ══════════════════════════════════════════════════════════════════
# 5. 분석 함수(기존 유지)
# ══════════════════════════════════════════════════════════════════
def _agent_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "상담사" not in df.columns:
        return df
    return df[~df["상담사"].astype(str).isin(EXCLUDED_AGENTS)].copy()


def calc_response_rate(df_all, df_scored, group_col=None):
    if group_col and group_col in df_all.columns:
        rows = []
        for g in sorted(df_all[group_col].dropna().unique()):
            total  = len(df_all[df_all[group_col]==g])
            scored = len(df_scored[df_scored[group_col]==g]) if group_col in df_scored.columns else 0
            rows.append({"구분":g,"발송건수":total,"응답건수":scored,
                         "응답률(%)":round(scored/total*100,1) if total>0 else 0})
        return pd.DataFrame(rows)
    total  = len(df_all)
    scored = len(df_scored)
    return pd.DataFrame([{"구분":"전체","발송건수":total,"응답건수":scored,
                          "응답률(%)":round(scored/total*100,1) if total>0 else 0}])


def pivot_avg(df, group_col, agent_filter=False):
    score_cols = [c for c in ["친절점수","만족점수","최종점수"] if c in df.columns]
    if group_col not in df.columns or not score_cols:
        return pd.DataFrame()
    src = _agent_filter(df) if agent_filter else df
    res = src.groupby(group_col)[score_cols].mean().round(1).reset_index()
    if "최종점수" in src.columns:
        cnt = src.groupby(group_col)["최종점수"].count().reset_index(name="응답건수")
        res = res.merge(cnt, on=group_col, how="left")
    if "최종점수" in res.columns:
        res["상태"] = res["최종점수"].apply(
            lambda x: "🔴 주의" if x < SCORE_CAUTION
                      else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
        res = res.sort_values("최종점수", ascending=False)
    return res


def detect_gaps(df, threshold=20):
    if "친절점수" not in df.columns or "만족점수" not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["갭(친절-만족)"] = df2["친절점수"] - df2["만족점수"]
    return df2[df2["갭(친절-만족)"].abs() >= threshold].copy()


def sentiment_summary(df):
    if "긍정부정" not in df.columns:
        return pd.DataFrame()
    counts = df["긍정부정"].value_counts()
    total  = counts.sum()
    return pd.DataFrame([
        {"긍정/부정":l,
         "건수":counts.get(l,0),
         "비율(%)":round(counts.get(l,0)/total*100,1) if total>0 else 0}
        for l in ["긍정","중립","부정"]
    ])


def extract_keywords(df, top_n=20):
    col = next((c for c in ["주관식","verbatim","Q3"] if c in df.columns), None)
    if not col:
        return []
    stopwords = {
        "이","가","은","는","을","를","의","에","도","로","으로","와","과",
        "이다","있다","하다","했다","합니다","습니다","이에요","예요","해요",
        "네요","요","죠","그리고","그런데","하지만","그래서","때문에",
        "것","수","등","더","잘","좀","매우","너무","정말","진짜",
        "많이","항상","늘","제","제가","저","저는","저도","됐습니다","했습니다",
    }
    words = []
    for t in df[col].dropna().astype(str):
        words.extend([w for w in re.findall(r"[가-힣]{2,}", t) if w not in stopwords])
    return Counter(words).most_common(top_n)


def calc_mom(now_val, prev_val, is_pp=False):
    if prev_val is None or prev_val == 0 or now_val is None:
        return None, "-"
    try:
        n, p = float(now_val), float(prev_val)
        if is_pp:
            d = round(n - p, 1)
            s = f"{'▲' if d>=0 else '▼'}{abs(d)}%p"
        else:
            d = round((n - p) / abs(p) * 100, 1)
            s = f"{'▲' if d>=0 else '▼'}{abs(d)}%"
        return d, s
    except Exception:
        return None, "-"


def action_needed(df, df_all):
    actions = []
    df_viz = _agent_filter(df)

    if "상담사" in df_viz.columns and "최종점수" in df_viz.columns:
        for agent, score in df_viz.groupby("상담사")["최종점수"].mean().items():
            if score < SCORE_CAUTION:
                actions.append({"구분":"상담사","항목":agent,
                    "내용":f"최종점수 {score:.1f}점 (기준 70점 미만) – 즉시 코칭 필요",
                    "우선순위":"🔴 긴급"})
            elif score < SCORE_GOOD:
                actions.append({"구분":"상담사","항목":agent,
                    "내용":f"최종점수 {score:.1f}점 – 지속 모니터링 필요",
                    "우선순위":"🟡 주의"})

    if "채널_구분" in df_all.columns:
        for _, row in calc_response_rate(df_all, df, "채널_구분").iterrows():
            if row["응답률(%)"] < 20:
                actions.append({"구분":"채널","항목":row["구분"],
                    "내용":f"응답률 {row['응답률(%)']}% – 발송 프로세스 점검 필요",
                    "우선순위":"🟠 개선"})

    if "긍정부정" in df.columns:
        neg = df[df["긍정부정"]=="부정"]
        if len(neg) > 0:
            actions.append({"구분":"VOC","항목":"부정 응답",
                "내용":f"{len(neg)}건 부정 의견 감지 – 즉시 VOC 검토 필요",
                "우선순위":"🔴 긴급"})

    gap_df = detect_gaps(df)
    if len(gap_df) > 0:
        actions.append({"구분":"품질","항목":"친절↔만족 점수 괴리",
            "내용":f"{len(gap_df)}건에서 친절↔만족 20점↑ 갭 발생",
            "우선순위":"🟠 개선"})

    return pd.DataFrame(actions) if actions else pd.DataFrame(columns=["구분","항목","내용","우선순위"])


# ══════════════════════════════════════════════════════════════════
# 6. 주차/일자 필터 (기존 유지)
# ══════════════════════════════════════════════════════════════════
def _normalize_week_str(w):
    if not w:
        return w
    w = str(w).strip()
    if re.match(r"^\d{4}-W\d{2}$", w):
        return w
    m = re.match(r"^(\d{4})-W(\d{1})$", w)
    if m:
        return f"{m.group(1)}-W{int(m.group(2)):02d}"
    return w


def get_week_range(week_str: str):
    try:
        week_str = _normalize_week_str(week_str)
        parts = week_str.split("-W")
        if len(parts) != 2:
            return None, None
        year, wnum = int(parts[0]), int(parts[1])
        start = pd.Timestamp.fromisocalendar(year, wnum, 1)
        return start, start + pd.Timedelta(days=6)
    except Exception:
        return None, None


def filter_by_week(df: pd.DataFrame, week_str: str, week_col: str = "회신주차_정제") -> pd.DataFrame:
    if not week_str:
        return pd.DataFrame(columns=df.columns)
    week_str_norm = _normalize_week_str(week_str)

    for try_col in [week_col, "회신주차_정제", "회신주차"]:
        if try_col in df.columns:
            mask = df[try_col].astype(str).str.strip() == week_str_norm
            result = df[mask].copy()
            if not result.empty:
                return result
            mask2 = df[try_col].astype(str).str.strip() == week_str
            result2 = df[mask2].copy()
            if not result2.empty:
                return result2

    start, end = get_week_range(week_str_norm)
    if start is None or "회신일자_정제" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    mask = (df["회신일자_정제"] >= start) & (df["회신일자_정제"] <= end)
    return df[mask].copy()


def filter_by_week_sent(df: pd.DataFrame, week_str: str) -> pd.DataFrame:
    if not week_str:
        return pd.DataFrame(columns=df.columns)
    week_str_norm = _normalize_week_str(week_str)

    for try_col in ["발송주차_정제", "발송주차"]:
        if try_col in df.columns:
            mask = df[try_col].astype(str).str.strip() == week_str_norm
            result = df[mask].copy()
            if not result.empty:
                return result
            mask2 = df[try_col].astype(str).str.strip() == week_str
            result2 = df[mask2].copy()
            if not result2.empty:
                return result2

    start, end = get_week_range(week_str_norm)
    if start is None or "발송일자_정제" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    mask = (df["발송일자_정제"] >= start) & (df["발송일자_정제"] <= end)
    return df[mask].copy()


def filter_by_date(df: pd.DataFrame, date_str: str, date_col: str = "회신일") -> pd.DataFrame:
    if not date_str:
        return pd.DataFrame(columns=df.columns)
    for try_col in [date_col, "회신일"]:
        if try_col in df.columns:
            result = df[df[try_col].astype(str) == date_str].copy()
            if not result.empty:
                return result
    if "회신일자_정제" in df.columns:
        try:
            target = pd.Timestamp(date_str)
            return df[df["회신일자_정제"].dt.date == target.date()].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=df.columns)


def filter_by_date_sent(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    if not date_str:
        return pd.DataFrame(columns=df.columns)
    for try_col in ["발송일", "발송일자"]:
        if try_col in df.columns:
            result = df[df[try_col].astype(str) == date_str].copy()
            if not result.empty:
                return result
    if "발송일자_정제" in df.columns:
        try:
            target = pd.Timestamp(date_str)
            return df[df["발송일자_정제"].dt.date == target.date()].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=df.columns)


# ══════════════════════════════════════════════════════════════════
# 7. Google Sheets 로드 (첫 번째 시트 기본)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def load_from_gsheets() -> pd.DataFrame:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(ttl="10m")  # 첫 번째 시트(기본) [Source: public-gsheet]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return df


def prepare_data(df_raw: pd.DataFrame):
    df = normalize_columns(df_raw)
    df = build_time_columns(df)
    df_all, df_active, df_scored, df_scored_all, available_months, retired_agents = split_active_and_scored(df)
    return df_all, df_active, df_scored, df_scored_all, available_months, retired_agents


# ══════════════════════════════════════════════════════════════════
# 8. UI (CSS)
# ══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px; }}
          section[data-testid="stSidebar"] {{
            background: {C_GRAY_LT};
            border-right: 1px solid {C_BORDER};
          }}
          .kpi {{
            border: 1px solid {C_BORDER};
            border-radius: 14px;
            padding: 12px 14px;
            background: white;
          }}
          .kpi-label {{
            color: {C_GRAY};
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 6px;
          }}
          .kpi-value {{
            font-size: 22px;
            font-weight: 900;
            color: {C_NAVY};
            line-height: 1.15;
          }}
          .kpi-delta {{
            font-size: 12px;
            font-weight: 800;
            margin-top: 6px;
          }}
          .section-title {{
            font-size: 16px;
            font-weight: 900;
            color: {C_NAVY};
            margin: 10px 0 10px 0;
          }}
        </style>
        """,
        unsafe_allow_html=True
    )


def kpi_card(label, value, delta="-", delta_color=C_GRAY):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta" style="color:{delta_color}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def dcol(s):
    if str(s).startswith("▲"):
        return C_GREEN
    if str(s).startswith("▼"):
        return C_RED
    return C_GRAY


def safe_mean(df, col):
    if col in df.columns and df[col].notna().any():
        return round(df[col].mean(), 1)
    return None


# ══════════════════════════════════════════════════════════════════
# 9. 월/주/일 데이터 생성 (대시보드용)
# ══════════════════════════════════════════════════════════════════
def fm(df, month, col="회신월_정제"):
    return df[df[col].astype(str) == month].copy() if (month and col in df.columns) else df.copy()


def fm_sent(df, month):
    if not month:
        return df.copy()
    for c in ["발송월_정제","회신월_정제"]:
        if c in df.columns:
            r = df[df[c].astype(str) == month].copy()
            if len(r) > 0:
                return r
    return df.copy()


def compute_monthly_trends(df_all, df_scored_all):
    if "회신월_정제" not in df_scored_all.columns:
        return pd.DataFrame(), pd.DataFrame()

    all_months = sorted([m for m in df_scored_all["회신월_정제"].dropna().unique() if m != "미확인"])

    monthly_rate_rows = []
    for m in all_months:
        ms = fm_sent(df_all, m)
        mr = fm(df_scored_all, m)
        t  = len(ms); r = len(mr)
        monthly_rate_rows.append({"월": m, "발송건수": t, "응답건수": r, "응답률(%)": round(r/t*100,1) if t>0 else 0})
    monthly_rate_df = pd.DataFrame(monthly_rate_rows)

    score_cols = [c for c in ["친절점수","만족점수","최종점수"] if c in df_scored_all.columns]
    monthly_score_rows = []
    for m in all_months:
        mr = fm(df_scored_all, m)
        row = {"월": m}
        for sc in score_cols:
            row[sc] = round(mr[sc].mean(), 1) if mr[sc].notna().any() else None
        monthly_score_rows.append(row)
    monthly_score_df = pd.DataFrame(monthly_score_rows)

    return monthly_rate_df, monthly_score_df


# ══════════════════════════════════════════════════════════════════
# 10. 페이지들
# ══════════════════════════════════════════════════════════════════
def page_overview(df_all, df_scored, df_scored_all, available_months, target_month, selected_date, selected_week):
    st.markdown('<div class="section-title">개요</div>', unsafe_allow_html=True)

    df_m        = fm(df_scored,     target_month)
    df_m_kpi    = fm(df_scored_all, target_month)
    df_m_all    = fm_sent(df_all,   target_month)

    sorted_m = sorted([m for m in available_months if m <= target_month]) if target_month else []
    prev_m   = sorted_m[-2] if len(sorted_m) >= 2 else None
    df_prev_all = fm_sent(df_all, prev_m)   if prev_m else pd.DataFrame()
    df_prev_kpi = fm(df_scored_all, prev_m) if prev_m else pd.DataFrame()

    total_sent   = len(df_m_all)
    total_scored = len(df_m_kpi)
    resp_rate    = round(total_scored/total_sent*100,1) if total_sent>0 else 0

    prev_sent   = len(df_prev_all) if not df_prev_all.empty else None
    prev_scored = len(df_prev_kpi) if not df_prev_kpi.empty else None
    prev_rate   = round(prev_scored/prev_sent*100,1) if prev_sent and prev_scored else None

    avg_final = safe_mean(df_m_kpi, "최종점수")
    avg_kind  = safe_mean(df_m_kpi, "친절점수")
    avg_satis = safe_mean(df_m_kpi, "만족점수")
    prev_final = safe_mean(df_prev_kpi, "최종점수") if not df_prev_kpi.empty else None

    _, d_sent_str  = calc_mom(total_sent,   prev_sent,   is_pp=False)
    _, d_resp_str  = calc_mom(total_scored, prev_scored, is_pp=False)
    _, d_rate_str  = calc_mom(resp_rate,    prev_rate,   is_pp=True)
    _, d_score_str = calc_mom(avg_final,    prev_final,  is_pp=False)

    neg_cnt = len(df_m_kpi[df_m_kpi["긍정부정"]=="부정"]) if "긍정부정" in df_m_kpi.columns else 0
    gap_cnt = len(detect_gaps(df_m_kpi))

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("발송건수", f"{total_sent:,}건", d_sent_str, dcol(d_sent_str))
    with c2: kpi_card("응답건수", f"{total_scored:,}건", d_resp_str, dcol(d_resp_str))
    with c3: kpi_card("응답률", f"{resp_rate}%", d_rate_str, dcol(d_rate_str))
    with c4: kpi_card("최종점수", "-" if avg_final is None else f"{avg_final}점", d_score_str, dcol(d_score_str))

    c5,c6,c7,c8 = st.columns(4)
    with c5: kpi_card("친절점수", "-" if avg_kind is None else f"{avg_kind}점")
    with c6: kpi_card("만족점수", "-" if avg_satis is None else f"{avg_satis}점")
    with c7: kpi_card("부정응답", f"{neg_cnt:,}건", "-", C_RED)
    with c8: kpi_card("점수갭(20점↑)", f"{gap_cnt:,}건", "-", C_AMBER)

    st.markdown("")

    # 트렌드
    monthly_rate_df, monthly_score_df = compute_monthly_trends(df_all, df_scored_all)
    t1, t2 = st.columns([1,1])
    with t1:
        st.markdown('<div class="section-title">응답률 변화 트렌드</div>', unsafe_allow_html=True)
        if monthly_rate_df.empty:
            st.caption("데이터 없음")
        else:
            fig = px.line(monthly_rate_df, x="월", y="응답률(%)", markers=True)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown('<div class="section-title">월별 친절·만족·최종 점수 추이</div>', unsafe_allow_html=True)
        if monthly_score_df.empty:
            st.caption("데이터 없음")
        else:
            mdf = monthly_score_df.melt(id_vars=["월"], value_vars=[c for c in ["친절점수","만족점수","최종점수"] if c in monthly_score_df.columns],
                                        var_name="지표", value_name="점수")
            fig = px.line(mdf, x="월", y="점수", color="지표", markers=True)
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("")

    # 채널 + 감성 + Action
    b1, b2, b3 = st.columns([1,1,1.2])

    with b1:
        st.markdown('<div class="section-title">채널별 응답률</div>', unsafe_allow_html=True)
        if "채널_구분" in df_m_all.columns:
            ch_rate = calc_response_rate(df_m_all, df_m_kpi, "채널_구분")
            st.dataframe(ch_rate, use_container_width=True, hide_index=True)
        else:
            st.caption("채널 컬럼 없음")

    with b2:
        st.markdown('<div class="section-title">긍정/중립/부정 분포</div>', unsafe_allow_html=True)
        if "긍정부정" in df_m_kpi.columns:
            s = df_m_kpi["긍정부정"].value_counts().reset_index()
            s.columns = ["감성","건수"]
            fig = px.pie(s, names="감성", values="건수", hole=0.55,
                         color="감성", color_discrete_map={"긍정":C_GREEN,"중립":C_AMBER,"부정":C_RED})
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("감성 컬럼 없음")

    with b3:
        st.markdown('<div class="section-title">즉시 조치 필요</div>', unsafe_allow_html=True)
        act = action_needed(df_m, df_m_all)
        if act.empty:
            st.success("특이사항 없음")
        else:
            st.dataframe(act, use_container_width=True, hide_index=True)


def page_day_week(df_all, df_scored, df_scored_all, selected_date, selected_week):
    st.markdown('<div class="section-title">일자 / 주차 리포트</div>', unsafe_allow_html=True)

    # Daily
    st.subheader("DAILY")
    if selected_date:
        df_day_all = filter_by_date_sent(df_all, selected_date)
        df_day_kpi = filter_by_date(df_scored_all, selected_date, date_col="회신일")
        if df_day_kpi.empty:
            st.warning("선택 일자에 응답 데이터가 없습니다.")
        else:
            st.write("채널별 응답률")
            st.dataframe(calc_response_rate(df_day_all, df_day_kpi, "채널_구분"), use_container_width=True, hide_index=True)

            st.write("감성 분포")
            st.dataframe(sentiment_summary(df_day_kpi), use_container_width=True, hide_index=True)

            neg = df_day_kpi[df_day_kpi["긍정부정"]=="부정"] if "긍정부정" in df_day_kpi.columns else pd.DataFrame()
            st.write("부정 VOC")
            if neg.empty:
                st.caption("없음")
            else:
                cols = [c for c in ["회신일","상담사","브랜드","채널_구분","최종점수","주관식","긍정부정"] if c in neg.columns]
                st.dataframe(neg[cols], use_container_width=True, hide_index=True)
    else:
        st.caption("사이드바에서 일자를 선택하세요.")

    st.divider()

    # Weekly
    st.subheader("WEEKLY")
    if selected_week:
        df_week_all = filter_by_week_sent(df_all, selected_week)
        df_week_kpi = filter_by_week(df_scored_all, selected_week, week_col="회신주차_정제")
        if df_week_kpi.empty:
            st.warning("선택 주차에 응답 데이터가 없습니다.")
        else:
            st.write("채널별 응답률")
            st.dataframe(calc_response_rate(df_week_all, df_week_kpi, "채널_구분"), use_container_width=True, hide_index=True)

            st.write("키워드 TOP 20")
            kws = extract_keywords(df_week_kpi, 20)
            if kws:
                st.dataframe(pd.DataFrame(kws, columns=["키워드","빈도"]), use_container_width=True, hide_index=True)
            else:
                st.caption("키워드 없음")
    else:
        st.caption("사이드바에서 주차를 선택하세요.")


def page_scores(df_m):
    st.markdown('<div class="section-title">점수 분석</div>', unsafe_allow_html=True)

    for gcol, title, do_af in [
        ("상담사",     "상담사별(재직/제외필터)", True),
        ("브랜드",     "브랜드별",              False),
        ("상담유형대", "상담유형(대)별",        False),
        ("채널_구분",  "채널별",                False),
        ("근속",       "근속별",                False),
    ]:
        if gcol not in df_m.columns:
            continue

        st.subheader(title)
        piv = pivot_avg(df_m, gcol, agent_filter=do_af)
        st.dataframe(piv, use_container_width=True, hide_index=True)

        if "최종점수" in df_m.columns:
            src = _agent_filter(df_m) if do_af else df_m
            grp = src.groupby(gcol)["최종점수"].mean().round(1).sort_values(ascending=False).head(30).reset_index()
            grp.columns = [gcol, "최종점수(평균)"]
            fig = px.bar(grp, x=gcol, y="최종점수(평균)")
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ 친절↔만족 점수 갭(20점↑)")
    gap = detect_gaps(df_m)
    if gap.empty:
        st.caption("없음")
    else:
        cols = [c for c in ["회신일","상담사","브랜드","채널_구분","친절점수","만족점수","최종점수","갭(친절-만족)","주관식"] if c in gap.columns]
        st.dataframe(gap[cols], use_container_width=True, hide_index=True)


def page_verbatim(df_m):
    st.markdown('<div class="section-title">주관식 분석</div>', unsafe_allow_html=True)

    st.subheader("감성 분포")
    sent = sentiment_summary(df_m)
    st.dataframe(sent, use_container_width=True, hide_index=True)

    if "긍정부정" in df_m.columns:
        s = df_m["긍정부정"].value_counts().reset_index()
        s.columns = ["감성","건수"]
        fig = px.pie(s, names="감성", values="건수", hole=0.55,
                     color="감성", color_discrete_map={"긍정":C_GREEN,"중립":C_AMBER,"부정":C_RED})
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("키워드 TOP 20")
    kws = extract_keywords(df_m, 20)
    if kws:
        kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
        fig = px.bar(kdf, x="빈도", y="키워드", orientation="h")
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("키워드 없음(주관식 컬럼 확인)")

    st.subheader("부정 응답 상세")
    if "긍정부정" in df_m.columns:
        neg = df_m[df_m["긍정부정"]=="부정"]
        if neg.empty:
            st.caption("없음")
        else:
            cols = [c for c in ["회신일","상담사","브랜드","채널_구분","최종점수","주관식"] if c in neg.columns]
            st.dataframe(neg[cols], use_container_width=True, hide_index=True)

    st.subheader("전체 주관식 응답")
    col = "주관식" if "주관식" in df_m.columns else None
    if col:
        vbt = df_m[df_m[col].notna() & (df_m[col].astype(str).str.strip()!="")]
        cols = [c for c in ["회신일","상담사","브랜드","채널_구분","최종점수","주관식","긍정부정"] if c in vbt.columns]
        st.dataframe(vbt[cols], use_container_width=True, hide_index=True)
    else:
        st.caption("주관식 컬럼 없음")


def page_integrated(df_m):
    st.markdown('<div class="section-title">통합 분석(히트맵)</div>', unsafe_allow_html=True)

    def heatmap(df, idx, col, val, title, agent_filter=False):
        src = _agent_filter(df) if agent_filter else df
        if src.empty or idx not in src.columns or col not in src.columns or val not in src.columns:
            st.caption(f"{title} : 데이터 없음")
            return
        p = src.pivot_table(values=val, index=idx, columns=col, aggfunc="mean").round(1)
        if p.empty:
            st.caption(f"{title} : 피벗 없음")
            return
        fig = px.imshow(p, text_auto=True, aspect="auto", color_continuous_scale=["#DC2626","#FEF3C7","#059669"])
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), title=title)
        st.plotly_chart(fig, use_container_width=True)

    # 상담유형대 x 채널
    if all(c in df_m.columns for c in ["상담유형대","채널_구분","최종점수"]):
        heatmap(df_m, "상담유형대", "채널_구분", "최종점수", "상담유형(대) × 채널 만족도", agent_filter=False)

    # 상담사 x 채널(필터)
    if all(c in df_m.columns for c in ["상담사","채널_구분","최종점수"]):
        heatmap(df_m, "상담사", "채널_구분", "최종점수", "상담사 × 채널 만족도(재직/제외필터)", agent_filter=True)

    # 근속 x 채널
    if all(c in df_m.columns for c in ["근속","채널_구분","최종점수"]):
        heatmap(df_m, "근속", "채널_구분", "최종점수", "근속 × 채널 만족도", agent_filter=False)


def page_action(df_m, df_m_all):
    st.markdown('<div class="section-title">Action 필요</div>', unsafe_allow_html=True)

    act = action_needed(df_m, df_m_all)
    if act.empty:
        st.success("특이사항 없음")
    else:
        st.dataframe(act, use_container_width=True, hide_index=True)

    st.subheader("부정 응답 상세")
    if "긍정부정" in df_m.columns:
        neg = df_m[df_m["긍정부정"]=="부정"]
        if neg.empty:
            st.caption("없음")
        else:
            cols = [c for c in ["회신일","상담사","브랜드","채널_구분","최종점수","주관식"] if c in neg.columns]
            st.dataframe(neg[cols], use_container_width=True, hide_index=True)


def page_daily_agent(df_m):
    st.markdown('<div class="section-title">일별 상담사 성과</div>', unsafe_allow_html=True)

    df_v = _agent_filter(df_m)
    if df_v.empty or "회신일" not in df_v.columns or "상담사" not in df_v.columns or "최종점수" not in df_v.columns:
        st.warning("일별 상담사 성과 데이터가 부족합니다.")
        return

    pivot = (df_v.groupby(["회신일","상담사"])["최종점수"].mean().round(1).unstack(fill_value=None))
    if pivot.empty:
        st.caption("피벗 데이터 없음")
        return

    st.subheader("일자별 상담사 최종점수 트렌드")
    p_long = pivot.reset_index().melt(id_vars=["회신일"], var_name="상담사", value_name="최종점수")
    p_long = p_long.dropna()
    fig = px.line(p_long, x="회신일", y="최종점수", color="상담사", markers=True)
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("상담사별 일평균 요약")
    daily_avg = (df_v.groupby("상담사")["최종점수"]
                 .agg(["mean", "min", "max", "count"]).round(1).reset_index())
    daily_avg.columns = ["상담사","평균점수","최저점수","최고점수","응답건수"]
    daily_avg["상태"] = daily_avg["평균점수"].apply(
        lambda x: "🔴 주의" if x < SCORE_CAUTION else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
    daily_avg = daily_avg.sort_values("평균점수", ascending=False)
    st.dataframe(daily_avg, use_container_width=True, hide_index=True)


def page_low_scores(df_scored_all, target_month=None):
    st.markdown('<div class="section-title">70점 미만 전체</div>', unsafe_allow_html=True)

    src = fm(df_scored_all, target_month) if target_month else df_scored_all.copy()
    if "최종점수" not in src.columns:
        st.warning("최종점수 컬럼 없음")
        return

    low = src[src["최종점수"] < SCORE_CAUTION].copy().sort_values("최종점수", ascending=True)
    if low.empty:
        st.success("70점 미만 데이터 없음")
        return

    cols = [c for c in ["회신월_정제","회신일","상담사","브랜드","채널_구분","상담유형대","근속","친절점수","만족점수","최종점수","주관식","긍정부정"] if c in low.columns]
    st.dataframe(low[cols], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# 11. Streamlit App
# ══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="CSAT Dashboard", layout="wide")
    inject_css()

    st.sidebar.markdown("## CSAT Dashboard")
    menu = st.sidebar.radio(
        "메뉴",
        ["개요","일자·주차","점수분석","주관식분석","통합분석(히트맵)","Action필요","일별상담사성과","70점미만_전체"]
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("구글시트 새로고침"):
        load_from_gsheets.clear()

    # 데이터 로드
    try:
        df_raw = load_from_gsheets()
    except Exception as e:
        st.error("구글시트 연결 실패: 공개 설정(링크 공개)과 Secrets를 확인하세요.")
        st.exception(e)
        st.info("공개 구글시트 연결 가이드: https://docs.streamlit.io/develop/tutorials/databases/public-gsheet")
        return

    df_all, df_active, df_scored, df_scored_all, available_months, retired_agents = prepare_data(df_raw)

    # 기간 선택
    st.sidebar.markdown("### 기간 선택")
    if available_months:
        target_month = st.sidebar.selectbox("월(필수)", available_months, index=len(available_months)-1)
    else:
        target_month = st.sidebar.text_input("월(예: 2026-01)", value="")

    # 일자/주차 목록
    MISSING_SET = {"","nan","NaT","None","NaN","<NA>","NA"}
    available_dates = sorted([str(d) for d in df_scored_all.get("회신일", pd.Series([])).dropna().unique() if str(d) not in MISSING_SET]) \
                      if "회신일" in df_scored_all.columns else []
    week_col = "회신주차_정제" if "회신주차_정제" in df_scored_all.columns else "회신주차"
    available_weeks = sorted([str(w) for w in df_scored_all.get(week_col, pd.Series([])).dropna().unique() if str(w) not in MISSING_SET]) \
                      if week_col in df_scored_all.columns else []

    selected_date = st.sidebar.selectbox("일자(선택)", [""] + available_dates, index=0)
    selected_date = selected_date or None
    selected_week = st.sidebar.selectbox("주차(선택)", [""] + available_weeks, index=0)
    selected_week = selected_week or None

    st.sidebar.markdown("---")
    st.sidebar.caption(f"주의 < {SCORE_CAUTION} / 양호 ≥ {SCORE_GOOD}")
    st.sidebar.caption(f"제외 상담사: {', '.join(sorted(EXCLUDED_AGENTS))}")

    # 상단
    st.markdown("## 고객 만족도 대시보드 (센터장)")
    st.caption(f"월: {target_month} | 일자: {selected_date or '-'} | 주차: {selected_week or '-'}")

    # 월 기준 df_m (재직자 scored)
    df_m     = fm(df_scored, target_month)
    df_m_all = fm_sent(df_all, target_month)

    # 라우팅
    if menu == "개요":
        page_overview(df_all, df_scored, df_scored_all, available_months, target_month, selected_date, selected_week)
    elif menu == "일자·주차":
        page_day_week(df_all, df_scored, df_scored_all, selected_date, selected_week)
    elif menu == "점수분석":
        page_scores(df_m)
    elif menu == "주관식분석":
        page_verbatim(df_m)
    elif menu == "통합분석(히트맵)":
        page_integrated(df_m)
    elif menu == "Action필요":
        page_action(df_m, df_m_all)
    elif menu == "일별상담사성과":
        page_daily_agent(df_m)
    elif menu == "70점미만_전체":
        page_low_scores(df_scored_all, target_month=target_month)


if __name__ == "__main__":
    main()
