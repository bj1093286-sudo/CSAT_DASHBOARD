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
# 0. 전역 상수 & 색상 팔레트 (변경 없음)
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

# ✅ shadcn/ui 개선 - 색상 토큰 추가
C_PRIMARY     = "#6366f1"
C_PRIMARY_LT  = "rgba(99,102,241,0.1)"
C_MUTED       = "#f8fafc"
C_MUTED_FG    = "#64748b"
C_FOREGROUND  = "#0f172a"
C_BORDER_SH   = "rgba(226,232,240,0.8)"
C_BG          = "#F0F2F5"
C_SUCCESS     = "#22c55e"
C_SUCCESS_FG  = "#16a34a"
C_DANGER      = "#ef4444"
C_DANGER_FG   = "#dc2626"
C_WARNING     = "#f59e0b"
C_INFO        = "#3b82f6"

CHART_COLORS = [
    "#6366f1","#22c55e","#f59e0b","#7C3AED",
    "#0891B2","#DB2777","#65A30D","#EA580C",
    "#0284C7","#DC2626","#16A34A","#9333EA",
]

SCORE_GOOD    = 90
SCORE_CAUTION = 70
EXCLUDED_AGENTS = {"엄소라","이은덕","한인경","양현정","이혜선","박성주"}


# ══════════════════════════════════════════════════════════════════
# 1. 날짜 파싱 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def parse_date_series(series: pd.Series) -> pd.Series:
    def try_parse(val):
        if pd.isna(val):
            return pd.NaT
        if isinstance(val, (pd.Timestamp, datetime)):
            return pd.Timestamp(val)
        s = str(val).strip().replace(" ", "")
        if s in ("", "nan", "NaT", "None", "NaN"):
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
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m", "%Y/%m",
                    "%Y.%m", "%Y%m%d", "%Y%m", "%m/%d/%Y", "%d/%m/%Y",
                    "%Y년%m월%d일", "%Y년%m월"]:
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
# 2. 감성 분류 (변경 없음)
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
    r"불편함?\s*없", r"불만\s*없", r"문제\s*없", r"걱정\s*없",
    r"어렵지\s*않", r"나쁘지\s*않", r"부족하지\s*않",
    r"아쉽지\s*않", r"실망하지\s*않", r"불편하지\s*않",
    r"늦지\s*않", r"느리지\s*않",
    r"(?:전혀|하나도)\s*(?:불편|불만|문제)",
    r"전혀\s*없", r"하나도\s*없",
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
    t = re.sub(r"\s+", "", text.strip())
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
# 3. 컬럼 매핑 (변경 없음)
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
        "친절점수":   ["친절","Q1점수"],
        "만족점수":   ["만족","Q2점수"],
        "최종점수":   ["최종","final"],
        "주관식":     ["Q3","verbatim","의견"],
        "상담사":     ["agent","담당자"],
        "브랜드":     ["brand"],
        "채널":       ["channel","매체"],
        "근속":       ["근속","tenure"],
        "입사일":     ["입사"],
        "상담유형대": ["대분류","유형대"],
        "문의유형":   ["문의","inquiry"],
        "회신주차":   ["회신주차","reply_week","replyweek"],
        "발송주차":   ["발송주차","send_week","sendweek"],
        "상담KEY":    ["상담key","상담Key","상담키","consultkey"],
    }
    for std, kws in fallback.items():
        if std not in df.columns:
            for col in df.columns:
                if any(kw.lower() in col.lower() for kw in kws):
                    df = df.rename(columns={col: std})
                    break
    return df


# ══════════════════════════════════════════════════════════════════
# 4. 데이터 정제 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def build_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["친절점수","만족점수","최종점수","총합","Q1","Q2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for dcol in ["회신일자","발송일자","입사일"]:
        if dcol in df.columns:
            df[dcol] = parse_date_series(df[dcol])

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

    def make_week_str(ts):
        if pd.isna(ts):
            return pd.NA
        iso = ts.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    if "회신주차" in df.columns:
        df["회신주차_정제"] = df["회신주차"].astype(str).str.strip().replace(
            {"nan": pd.NA, "None": pd.NA, "NaT": pd.NA, "": pd.NA, "NaN": pd.NA})
    else:
        df["회신주차_정제"] = df["회신일자_정제"].apply(make_week_str)

    if "발송주차" in df.columns:
        df["발송주차_정제"] = df["발송주차"].astype(str).str.strip().replace(
            {"nan": pd.NA, "None": pd.NA, "NaT": pd.NA, "": pd.NA, "NaN": pd.NA})
    else:
        df["발송주차_정제"] = df["발송일자_정제"].apply(make_week_str)

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

    if "근속" not in df.columns:
        if "입사일" in df.columns:
            ref   = df["회신일자_정제"]
            today = pd.Timestamp(datetime.now().date())
            ref   = ref.where(ref.notna(), other=today)
            join  = df["입사일"]
            months = np.where(
                join.notna() & ref.notna(),
                (ref.dt.year - join.dt.year) * 12 + (ref.dt.month - join.dt.month),
                np.nan
            )
            df["근속개월"] = pd.to_numeric(months, errors="coerce")

            def bucket(m):
                if pd.isna(m): return pd.NA
                m = float(m)
                if m < 3:   return "0~3개월"
                if m < 6:   return "3~6개월"
                if m < 12:  return "6~12개월"
                if m < 24:  return "1~2년"
                return "2년+"

            df["근속"] = pd.Series(df["근속개월"]).apply(bucket)
        else:
            df["근속"] = pd.NA

    return df


def split_active_and_scored(df: pd.DataFrame):
    MISSING = {"#N/A","nan","NaT","None","NA","N/A","","#REF!","NaN","<NA>"}
    df_all  = df.copy()

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
# 5. 분석 함수 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def _agent_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "상담사" not in df.columns:
        return df
    return df[~df["상담사"].astype(str).isin(EXCLUDED_AGENTS)].copy()


def calc_response_rate(df_all, df_scored, group_col=None):
    if group_col and group_col in df_all.columns:
        rows = []
        for g in sorted(df_all[group_col].dropna().unique()):
            total  = len(df_all[df_all[group_col] == g])
            scored = len(df_scored[df_scored[group_col] == g]) if group_col in df_scored.columns else 0
            rows.append({"구분": g, "발송건수": total, "응답건수": scored,
                         "응답률(%)": round(scored / total * 100, 1) if total > 0 else 0})
        return pd.DataFrame(rows)
    total  = len(df_all)
    scored = len(df_scored)
    return pd.DataFrame([{"구분": "전체", "발송건수": total, "응답건수": scored,
                          "응답률(%)": round(scored / total * 100, 1) if total > 0 else 0}])


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
            lambda x: "🔴 주의" if x < SCORE_CAUTION else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
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
        {"긍정/부정": l,
         "건수": counts.get(l, 0),
         "비율(%)": round(counts.get(l, 0) / total * 100, 1) if total > 0 else 0}
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
            s = f"{'▲' if d >= 0 else '▼'}{abs(d)}%p"
        else:
            d = round((n - p) / abs(p) * 100, 1)
            s = f"{'▲' if d >= 0 else '▼'}{abs(d)}%"
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
        neg = df[df["긍정부정"] == "부정"]
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
# 6. 주차/일자 필터 (변경 없음)
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


def get_prev_week_str(week_str: str, offset: int = 1) -> str:
    try:
        week_str = _normalize_week_str(week_str)
        start, _ = get_week_range(week_str)
        if start is None:
            return ""
        prev = start - pd.Timedelta(weeks=offset)
        iso  = prev.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ""


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
# 7. RAW 테이블 컬럼 정렬 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def get_display_cols(df: pd.DataFrame, preferred: list) -> list:
    result = []
    if "상담KEY" in df.columns:
        result.append("상담KEY")
    for c in preferred:
        if c in df.columns and c not in result:
            result.append(c)
    for c in df.columns:
        if c not in result and not c.startswith("_") and c not in [
            "발송월_정제","회신월_정제","회신일자_정제","발송일자_정제",
            "회신주차_정제","발송주차_정제","근속개월"
        ]:
            result.append(c)
    return result


# ══════════════════════════════════════════════════════════════════
# 8. Google Sheets 로드 (변경 없음)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def load_from_gsheets() -> pd.DataFrame:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df   = conn.read(ttl="10m")
    df   = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return df


@st.cache_data(ttl=600)
def prepare_data(df_raw_hash):
    pass


def prepare_data_from_df(df_raw: pd.DataFrame):
    df = normalize_columns(df_raw)
    df = build_time_columns(df)
    return split_active_and_scored(df)


# ══════════════════════════════════════════════════════════════════
# 9. 채널별 월별 추이 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def build_channel_monthly_trend(df_all, df_scored_all, all_months):
    result = {}
    if "채널_구분" not in df_all.columns:
        return result
    channels = sorted(df_all["채널_구분"].dropna().unique())

    def fm_s(df, month):
        for c in ["발송월_정제","회신월_정제"]:
            if c in df.columns:
                r = df[df[c].astype(str) == month].copy()
                if len(r) > 0:
                    return r
        return df.copy()

    def fm(df, month):
        if "회신월_정제" in df.columns:
            return df[df["회신월_정제"].astype(str) == month].copy()
        return df.copy()

    for ch in channels:
        rows = []
        for m in all_months:
            ms    = fm_s(df_all, m)
            mr    = fm(df_scored_all, m)
            ms_ch = ms[ms["채널_구분"] == ch] if "채널_구분" in ms.columns else ms
            mr_ch = mr[mr["채널_구분"] == ch] if "채널_구분" in mr.columns else mr
            t = len(ms_ch); r = len(mr_ch)
            row_d = {"월": m, "발송건수": t, "응답건수": r,
                     "응답률(%)": round(r / t * 100, 1) if t > 0 else 0}
            for sc in ["친절점수","만족점수","최종점수"]:
                if sc in mr_ch.columns and mr_ch[sc].notna().any():
                    row_d[sc] = round(mr_ch[sc].mean(), 1)
                else:
                    row_d[sc] = None
            rows.append(row_d)
        result[ch] = pd.DataFrame(rows)
    return result


# ══════════════════════════════════════════════════════════════════
# 10. 3주 트렌드 데이터 생성 (변경 없음)
# ══════════════════════════════════════════════════════════════════
def build_week_trend_data(df_all, df_scored_all, selected_week):
    MISSING_SET_W = {"","nan","NaT","None","NaN","<NA>","NA"}
    w_col = "회신주차_정제" if "회신주차_정제" in df_scored_all.columns else "회신주차"
    all_weeks_sorted = []
    if w_col in df_scored_all.columns:
        all_weeks_sorted = sorted([
            str(w) for w in df_scored_all[w_col].dropna().unique()
            if str(w) not in MISSING_SET_W
        ])

    selected_week_norm = _normalize_week_str(selected_week)

    def _get_prev(wk_norm, offset):
        if wk_norm in all_weeks_sorted:
            idx = all_weeks_sorted.index(wk_norm)
            if idx - offset >= 0:
                return all_weeks_sorted[idx - offset]
        return get_prev_week_str(wk_norm, offset)

    w_prev2 = _get_prev(selected_week_norm, 2)
    w_prev1 = _get_prev(selected_week_norm, 1)
    week_list   = [w_prev2, w_prev1, selected_week_norm]
    week_labels = [
        f"W-2 ({w_prev2})"            if w_prev2 else "W-2",
        f"W-1 ({w_prev1})"            if w_prev1 else "W-1",
        f"선택 ({selected_week_norm})",
    ]

    rate_vals  = []
    score_rows = {"친절점수":[], "만족점수":[], "최종점수":[]}

    for wk in week_list:
        if not wk:
            rate_vals.append(None)
            for k in score_rows:
                score_rows[k].append(None)
            continue
        wk_all = filter_by_week_sent(df_all, wk)
        wk_kpi = filter_by_week(df_scored_all, wk, week_col="회신주차_정제")
        t = len(wk_all); r = len(wk_kpi)
        rate_vals.append(round(r / t * 100, 1) if t > 0 else 0.0)
        for k in score_rows:
            if k in wk_kpi.columns and wk_kpi[k].notna().any():
                score_rows[k].append(round(wk_kpi[k].mean(), 1))
            else:
                score_rows[k].append(None)

    return {"labels": week_labels, "rate_vals": rate_vals, "score_rows": score_rows}


# ══════════════════════════════════════════════════════════════════
# 11. UI (CSS + 사이드바)
# ══════════════════════════════════════════════════════════════════

# ✅ shadcn/ui 개선 - inject_css 전체 재작성
def inject_css():
    st.markdown(f"""
    <style>
      /* ── 폰트 import ── */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

      /* ── CSS 변수 (shadcn/ui 토큰) ── */
      :root {{
        --background:        #ffffff;
        --foreground:        #0f172a;
        --muted:             #f8fafc;
        --muted-foreground:  #64748b;
        --border:            rgba(226, 232, 240, 0.8);
        --primary:           #6366f1;
        --primary-hover:     #4f46e5;
        --primary-lt:        rgba(99, 102, 241, 0.1);
        --success:           #22c55e;
        --success-fg:        #16a34a;
        --success-lt:        rgba(34, 197, 94, 0.1);
        --danger:            #ef4444;
        --danger-fg:         #dc2626;
        --danger-lt:         rgba(239, 68, 68, 0.1);
        --warning:           #f59e0b;
        --warning-fg:        #d97706;
        --warning-lt:        rgba(245, 158, 11, 0.1);
        --info:              #3b82f6;
        --info-lt:           rgba(59, 130, 246, 0.1);
        --card-shadow:       0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
        --card-radius:       12px;
        --btn-radius:        8px;
        --badge-radius:      9999px;
        --transition:        all 150ms cubic-bezier(0.4, 0, 0.2, 1);
      }}

      /* ── 전역 기본 ── */
      html, body, [class*="css"] {{
        font-family: 'Inter', 'Noto Sans KR', sans-serif;
        color: var(--foreground);
        background-color: #F0F2F5;
      }}

      .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
        background-color: #F0F2F5;
      }}

      /* ════════════════════════════════════
         사이드바 - Compact 개선 (메뉴 상단 집중)
         ════════════════════════════════════ */
      section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
        min-width: 200px !important;
        max-width: 215px !important;
      }}
      section[data-testid="stSidebar"] > div {{
        padding: 0 !important;
      }}
      /* 사이드바 내 selectbox 라벨 소형화 */
      section[data-testid="stSidebar"] label {{
        font-size: 10px !important;
        color: rgba(148,163,184,0.7) !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
      }}
      /* 사이드바 내 selectbox 본체 소형화 */
      section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {{
        font-size: 12px !important;
        min-height: 30px !important;
        padding: 2px 8px !important;
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
      }}
      section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
        position: absolute !important;
        top: -30px !important;
        left: 0 !important;
        opacity: 0 !important;
        height: 30px !important;
        width: 100% !important;
        z-index: 999 !important;
        cursor: pointer !important;
        border: none !important;
        background: transparent !important;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - KPI 카드
         ════════════════════════════════════ */
      .kpi {{
        background: var(--background);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 20px 22px;
        box-shadow: var(--card-shadow);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
      }}
      .kpi::before {{
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: var(--primary);
        border-radius: 3px 0 0 3px;
      }}
      .kpi:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-1px);
      }}
      .kpi-label {{
        font-size: 11px;
        font-weight: 700;
        color: var(--muted-foreground);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
      }}
      .kpi-value {{
        font-size: 26px;
        font-weight: 800;
        color: var(--foreground);
        line-height: 1.1;
        letter-spacing: -0.025em;
      }}
      .kpi-delta {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 8px;
        padding: 2px 8px;
        border-radius: var(--badge-radius);
        background: rgba(100, 116, 139, 0.08);
        color: var(--muted-foreground);
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 섹션 타이틀
         ════════════════════════════════════ */
      .section-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 700;
        color: var(--foreground);
        letter-spacing: -0.01em;
        margin: 20px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
      }}
      .section-title-icon {{
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--primary-lt);
        border-radius: 6px;
        font-size: 13px;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 페이지 헤더
         ════════════════════════════════════ */
      .page-header {{
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #2563eb 100%);
        border-radius: var(--card-radius);
        padding: 24px 28px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 16px rgba(99,102,241,0.18);
        position: relative;
        overflow: hidden;
      }}
      .page-header::after {{
        content: '';
        position: absolute;
        right: -40px;
        top: -40px;
        width: 160px;
        height: 160px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
      }}
      .page-header-title {{
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.3;
      }}
      .page-header-sub {{
        font-size: 12px;
        opacity: 0.65;
        margin-top: 6px;
        font-weight: 400;
        letter-spacing: 0.01em;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 배지 (Badge)
         ════════════════════════════════════ */
      .badge {{
        display: inline-flex;
        align-items: center;
        border-radius: var(--badge-radius);
        font-size: 11px;
        font-weight: 700;
        padding: 2px 10px;
        letter-spacing: 0.02em;
      }}
      .badge-default {{
        background: var(--primary-lt);
        color: var(--primary);
        border: 1px solid rgba(99,102,241,0.2);
      }}
      .badge-green {{
        background: var(--success-lt);
        color: var(--success-fg);
        border: 1px solid rgba(34,197,94,0.2);
      }}
      .badge-red {{
        background: var(--danger-lt);
        color: var(--danger-fg);
        border: 1px solid rgba(239,68,68,0.2);
      }}
      .badge-amber {{
        background: var(--warning-lt);
        color: var(--warning-fg);
        border: 1px solid rgba(245,158,11,0.2);
      }}
      .badge-info {{
        background: var(--info-lt);
        color: var(--info);
        border: 1px solid rgba(59,130,246,0.2);
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 카드 컨테이너
         ════════════════════════════════════ */
      .sh-card {{
        background: var(--background);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 20px 24px;
        box-shadow: var(--card-shadow);
        margin-bottom: 12px;
      }}
      .sh-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }}
      .sh-card-title {{
        font-size: 14px;
        font-weight: 700;
        color: var(--foreground);
        letter-spacing: -0.01em;
      }}
      .sh-card-desc {{
        font-size: 12px;
        color: var(--muted-foreground);
        margin-top: 2px;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 버튼
         ════════════════════════════════════ */
      div.stButton > button {{
        border-radius: var(--btn-radius) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        height: 36px !important;
        padding: 0 16px !important;
        transition: var(--transition) !important;
        letter-spacing: 0.01em;
      }}
      div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(99,102,241,0.2) !important;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 데이터프레임
         ════════════════════════════════════ */
      div[data-testid="stDataFrame"] {{
        border-radius: 10px !important;
        overflow: hidden;
        border: 1px solid var(--border);
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 셀렉트박스/인풋
         ════════════════════════════════════ */
      div[data-testid="stSelectbox"] > div > div,
      div[data-testid="stTextInput"] > div > div > input {{
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        font-size: 13px !important;
        transition: var(--transition) !important;
      }}
      div[data-testid="stSelectbox"] label,
      div[data-testid="stRadio"] label,
      div[data-testid="stTextInput"] label {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--muted-foreground) !important;
        letter-spacing: 0.04em !important;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 탭
         ════════════════════════════════════ */
      div[data-testid="stTabs"] > div:first-child {{
        background: #f1f5f9;
        border-radius: 8px;
        padding: 4px;
        gap: 2px;
        border: none !important;
      }}
      div[data-testid="stTabs"] button[role="tab"] {{
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: var(--muted-foreground) !important;
        transition: var(--transition) !important;
        border: none !important;
        padding: 6px 14px !important;
      }}
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        background: var(--background) !important;
        color: var(--foreground) !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
      }}
      div[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {{
        background: rgba(99,102,241,0.06) !important;
        color: var(--primary) !important;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - Alert / Info
         ════════════════════════════════════ */
      div[data-testid="stAlert"] {{
        border-radius: 10px !important;
        border-left-width: 3px !important;
        font-size: 13px !important;
      }}

      /* ════════════════════════════════════
         ✅ shadcn/ui - 주차 트렌드 테이블
         ════════════════════════════════════ */
      .trend-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
      }}
      .trend-table th {{
        background: var(--foreground);
        color: white;
        padding: 10px 14px;
        text-align: center;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.04em;
      }}
      .trend-table td {{
        padding: 9px 14px;
        text-align: center;
        border-bottom: 1px solid var(--border);
        font-size: 13px;
        color: var(--foreground);
      }}
      .trend-table tr:last-child td {{
        border-bottom: none;
      }}
      .trend-table tr:nth-child(even) td {{
        background: var(--muted);
      }}
      .trend-table tr:hover td {{
        background: var(--primary-lt);
        transition: var(--transition);
      }}

      /* ════════════════════════════════════
         ✅ 반응형 여백
         ════════════════════════════════════ */
      .spacer-sm  {{ height: 8px; }}
      .spacer-md  {{ height: 16px; }}
      .spacer-lg  {{ height: 24px; }}
    </style>
    """, unsafe_allow_html=True)


# ✅ shadcn/ui 개선 - kpi_card HTML 재작성
def kpi_card(label, value, delta="-", delta_color=C_GRAY):
    # delta 색상 → badge 클래스 매핑
    if "▲" in str(delta):
        badge_style = "background:rgba(34,197,94,0.1);color:#16a34a;"
    elif "▼" in str(delta):
        badge_style = "background:rgba(239,68,68,0.1);color:#dc2626;"
    else:
        badge_style = "background:rgba(100,116,139,0.08);color:#64748b;"

    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta" style="{badge_style}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def dcol(s):
    if str(s).startswith("▲"): return C_GREEN
    if str(s).startswith("▼"): return C_RED
    return C_GRAY


def safe_mean(df, col):
    if col in df.columns and df[col].notna().any():
        return round(df[col].mean(), 1)
    return None


# ✅ shadcn/ui 개선 - section_title HTML 재작성
def section_title(text, icon=""):
    icon_html = f'<div class="section-title-icon">{icon}</div>' if icon else ""
    st.markdown(
        f'<div class="section-title">{icon_html}<span>{text}</span></div>',
        unsafe_allow_html=True
    )


# ✅ shadcn/ui 개선 - page_header HTML 재작성
def page_header(title, sub=""):
    sub_html = f"<div class='page-header-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-title">{title}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 12. 월/발송 필터 헬퍼 (변경 없음)
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
    rate_rows  = []
    for m in all_months:
        ms = fm_sent(df_all, m);  mr = fm(df_scored_all, m)
        t  = len(ms);             r  = len(mr)
        rate_rows.append({"월": m, "발송건수": t, "응답건수": r,
                          "응답률(%)": round(r / t * 100, 1) if t > 0 else 0})
    monthly_rate_df = pd.DataFrame(rate_rows)

    score_cols  = [c for c in ["친절점수","만족점수","최종점수"] if c in df_scored_all.columns]
    score_rows_ = []
    for m in all_months:
        mr    = fm(df_scored_all, m)
        row_d = {"월": m}
        for sc in score_cols:
            row_d[sc] = round(mr[sc].mean(), 1) if mr[sc].notna().any() else None
        score_rows_.append(row_d)
    monthly_score_df = pd.DataFrame(score_rows_)
    return monthly_rate_df, monthly_score_df


# ══════════════════════════════════════════════════════════════════
# 13. 페이지 구현
# ══════════════════════════════════════════════════════════════════

# ── 13-1. 개요 (그래프 개선) ───────────────────────────────────
def page_overview(df_all, df_scored, df_scored_all, available_months,
                  target_month, selected_date, selected_week):

    page_header("대시보드 개요",
                f"월: {target_month}  |  일자: {selected_date or '-'}  |  주차: {selected_week or '-'}")

    df_m     = fm(df_scored,     target_month)
    df_m_kpi = fm(df_scored_all, target_month)
    df_m_all = fm_sent(df_all,   target_month)

    sorted_m    = sorted([m for m in available_months if m <= target_month]) if target_month else []
    prev_m      = sorted_m[-2] if len(sorted_m) >= 2 else None
    df_prev_all = fm_sent(df_all, prev_m)   if prev_m else pd.DataFrame()
    df_prev_kpi = fm(df_scored_all, prev_m) if prev_m else pd.DataFrame()

    total_sent   = len(df_m_all)
    total_scored = len(df_m_kpi)
    resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0

    prev_sent   = len(df_prev_all)  if not df_prev_all.empty  else None
    prev_scored = len(df_prev_kpi)  if not df_prev_kpi.empty  else None
    prev_rate   = round(prev_scored / prev_sent * 100, 1) if prev_sent and prev_scored else None

    avg_final  = safe_mean(df_m_kpi, "최종점수")
    avg_kind   = safe_mean(df_m_kpi, "친절점수")
    avg_satis  = safe_mean(df_m_kpi, "만족점수")
    prev_final = safe_mean(df_prev_kpi, "최종점수") if not df_prev_kpi.empty else None

    _, d_sent_str  = calc_mom(total_sent,   prev_sent,   is_pp=False)
    _, d_resp_str  = calc_mom(total_scored, prev_scored, is_pp=False)
    _, d_rate_str  = calc_mom(resp_rate,    prev_rate,   is_pp=True)
    _, d_score_str = calc_mom(avg_final,    prev_final,  is_pp=False)

    neg_cnt = len(df_m_kpi[df_m_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_m_kpi.columns else 0
    gap_cnt = len(detect_gaps(df_m_kpi))

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("발송건수",  f"{total_sent:,}건",              d_sent_str,  dcol(d_sent_str))
    with c2: kpi_card("응답건수",  f"{total_scored:,}건",            d_resp_str,  dcol(d_resp_str))
    with c3: kpi_card("응답률",    f"{resp_rate}%",                  d_rate_str,  dcol(d_rate_str))
    with c4: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점", d_score_str, dcol(d_score_str))

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5: kpi_card("친절점수",       "-" if avg_kind  is None else f"{avg_kind}점")
    with c6: kpi_card("만족점수",       "-" if avg_satis is None else f"{avg_satis}점")
    with c7: kpi_card("부정응답",       f"{neg_cnt:,}건", "-", C_RED)
    with c8: kpi_card("점수갭(20점↑)", f"{gap_cnt:,}건", "-", C_AMBER)

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # ✅ shadcn/ui 개선 - 트렌드 차트 (Area chart + 개선된 스타일)
    monthly_rate_df, monthly_score_df = compute_monthly_trends(df_all, df_scored_all)
    t1, t2 = st.columns(2)

    with t1:
        section_title("응답률 변화 트렌드", "📈")
        if monthly_rate_df.empty:
            st.caption("데이터 없음")
        else:
            # ✅ Line + Area 차트로 개선
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_rate_df["월"],
                y=monthly_rate_df["응답률(%)"],
                mode="lines+markers",
                name="응답률(%)",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=8, color="#6366f1",
                            line=dict(color="white", width=2)),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12, color="#0f172a"),
                showlegend=False,
                xaxis=dict(showgrid=False, linecolor="rgba(226,232,240,0.8)",
                           tickfont=dict(size=11, color="#64748b")),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                           linecolor="rgba(226,232,240,0.8)",
                           tickfont=dict(size=11, color="#64748b")),
            )
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        section_title("월별 친절·만족·최종 점수 추이", "📊")
        if monthly_score_df.empty:
            st.caption("데이터 없음")
        else:
            # ✅ 멀티라인 + 마커 개선
            fig = go.Figure()
            color_map = {
                "친절점수": ("#f59e0b", "rgba(245,158,11,0.08)"),
                "만족점수": ("#22c55e", "rgba(34,197,94,0.08)"),
                "최종점수": ("#6366f1", "rgba(99,102,241,0.08)"),
            }
            for sc, (color, fill) in color_map.items():
                if sc in monthly_score_df.columns:
                    fig.add_trace(go.Scatter(
                        x=monthly_score_df["월"],
                        y=monthly_score_df[sc],
                        mode="lines+markers",
                        name=sc,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, color=color,
                                    line=dict(color="white", width=2)),
                    ))
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=30),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12, color="#0f172a"),
                legend=dict(
                    orientation="h", y=-0.25, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"
                ),
                xaxis=dict(showgrid=False, linecolor="rgba(226,232,240,0.8)",
                           tickfont=dict(size=11, color="#64748b")),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                           linecolor="rgba(226,232,240,0.8)",
                           tickfont=dict(size=11, color="#64748b")),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # 채널별 + 긍정부정 + Action
    b1, b2, b3 = st.columns([1, 1, 1.3])

    with b1:
        section_title("채널별 응답률", "📡")
        if "채널_구분" in df_m_all.columns:
            ch_rate = calc_response_rate(df_m_all, df_m_kpi, "채널_구분")
            st.dataframe(ch_rate, use_container_width=True, hide_index=True)
        else:
            st.caption("채널 컬럼 없음")

    with b2:
        section_title("긍정/부정 분포", "😊")
        if "긍정부정" in df_m_kpi.columns:
            s = df_m_kpi["긍정부정"].value_counts().reset_index()
            s.columns = ["긍정/부정","건수"]
            # ✅ shadcn/ui 개선 - Donut 차트 스타일 개선
            fig = go.Figure(go.Pie(
                labels=s["긍정/부정"],
                values=s["건수"],
                hole=0.62,
                marker=dict(
                    colors=["#22c55e","#f59e0b","#ef4444"],
                    line=dict(color="white", width=2)
                ),
                textfont=dict(size=12, family="Inter"),
                hovertemplate="%{label}<br>%{value}건 (%{percent})<extra></extra>",
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR"),
                legend=dict(
                    orientation="v", x=1.02, y=0.5,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("긍정/부정 컬럼 없음")

    with b3:
        section_title("즉시 조치 필요", "⚠️")
        act = action_needed(df_m, df_m_all)
        if act.empty:
            st.success("특이사항 없음")
        else:
            st.dataframe(act, use_container_width=True, hide_index=True)

    # 채널별 월별 추이
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    section_title("채널별 월별 추이 (발송/응답/점수)", "📅")
    all_months_list = sorted([m for m in df_scored_all["회신월_정제"].dropna().unique()
                               if m != "미확인"]) if "회신월_정제" in df_scored_all.columns else []
    ch_trend = build_channel_monthly_trend(df_all, df_scored_all, all_months_list)
    if ch_trend:
        tabs = st.tabs(list(ch_trend.keys()))
        for tab, (ch_name, trend_df) in zip(tabs, ch_trend.items()):
            with tab:
                if not trend_df.empty:
                    st.dataframe(trend_df, use_container_width=True, hide_index=True)


# ── 13-2. 일자·주차 (그래프 개선) ─────────────────────────────
def page_day_week(df_all, df_scored, df_scored_all, selected_date, selected_week):
    try:
        page_header(
            "일자 / 주차 리포트",
            f"날짜: {selected_date or '미선택'}  |  주차: {selected_week or '미선택'}"
        )
    except Exception:
        st.title("일자 / 주차 리포트")

    # ── 미선택 시 전체 일별 요약 표시 ──
    if not selected_date and not selected_week:
        st.info("💡 사이드바 상단에서 **일자** 또는 **주차**를 선택하면 해당 기간 상세 분석이 표시됩니다. 아래는 전체 기간 일별 요약입니다.")
        try:
            if "회신일" in df_scored_all.columns and "최종점수" in df_scored_all.columns:
                daily_sum = (
                    df_scored_all[df_scored_all["회신일"].notna()]
                    .groupby("회신일")
                    .agg(응답건수=("최종점수", "count"), 평균점수=("최종점수", "mean"))
                    .round(1)
                    .reset_index()
                    .sort_values("회신일", ascending=True)
                )
                # 최근 30일만 차트용
                chart_data = daily_sum.tail(30).copy()

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    section_title("일별 응답건수 (최근 30일)", "")
                    fig_ds = go.Figure(go.Bar(
                        x=list(chart_data["회신일"]),
                        y=list(chart_data["응답건수"]),
                        marker=dict(color="#6366f1", opacity=0.8,
                                    line=dict(color="white", width=0.5)),
                    ))
                    fig_ds.update_layout(
                        height=300, margin=dict(l=10, r=10, t=10, b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=11),
                        xaxis=dict(showgrid=False, tickangle=-45,
                                   tickfont=dict(size=9, color="#64748b")),
                        yaxis=dict(showgrid=True,
                                   gridcolor="rgba(226,232,240,0.6)"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ds, use_container_width=True)

                with col_s2:
                    section_title("일별 평균점수 (최근 30일)", "")
                    fig_dp = go.Figure(go.Scatter(
                        x=list(chart_data["회신일"]),
                        y=list(chart_data["평균점수"]),
                        mode="lines+markers",
                        line=dict(color="#6366f1", width=2.5),
                        marker=dict(size=6, color="#6366f1",
                                    line=dict(color="white", width=2)),
                        fill="tozeroy",
                        fillcolor="rgba(99,102,241,0.07)",
                    ))
                    fig_dp.add_hline(y=SCORE_GOOD, line_dash="dash",
                                     line_color="#22c55e", line_width=1.5,
                                     annotation_text="양호(90)")
                    fig_dp.add_hline(y=SCORE_CAUTION, line_dash="dash",
                                     line_color="#ef4444", line_width=1.5,
                                     annotation_text="주의(70)")
                    fig_dp.update_layout(
                        height=300, margin=dict(l=10, r=80, t=10, b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=11),
                        xaxis=dict(showgrid=False, tickangle=-45,
                                   tickfont=dict(size=9, color="#64748b")),
                        yaxis=dict(showgrid=True,
                                   gridcolor="rgba(226,232,240,0.6)",
                                   range=[50, 105],
                                   tickfont=dict(size=10, color="#64748b")),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dp, use_container_width=True)

                section_title("일별 응답 현황 전체 목록", "")
                st.dataframe(
                    daily_sum.sort_values("회신일", ascending=False),
                    use_container_width=True, hide_index=True
                )
        except Exception as e:
            st.warning(f"일별 요약 표시 중 오류: {e}")

    tab_daily, tab_weekly = st.tabs(["📅 DAILY 상세", "📆 WEEKLY 상세"])

    # ── DAILY ──
    with tab_daily:
        if not selected_date:
            st.info("👆 사이드바 상단 **일자(선택)** 에서 날짜를 선택하세요.")
        else:
            df_day_all = filter_by_date_sent(df_all, selected_date)
            df_day_kpi = filter_by_date(df_scored_all, selected_date, date_col="회신일")
            if "긍정부정" not in df_day_kpi.columns and not df_day_kpi.empty:
                df_day_kpi = add_sentiment_column(df_day_kpi)

            if df_day_kpi.empty:
                st.warning(f"'{selected_date}' 에 해당하는 응답 데이터가 없습니다.")
            else:
                total_sent   = len(df_day_all)
                total_scored = len(df_day_kpi)
                resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0
                avg_final    = safe_mean(df_day_kpi, "최종점수")
                avg_kind     = safe_mean(df_day_kpi, "친절점수")
                avg_satis    = safe_mean(df_day_kpi, "만족점수")
                neg_cnt      = len(df_day_kpi[df_day_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_day_kpi.columns else 0

                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: kpi_card("발송건수",  f"{total_sent:,}건")
                with c2: kpi_card("응답건수",  f"{total_scored:,}건")
                with c3: kpi_card("응답률",    f"{resp_rate}%",
                                  delta_color=C_GREEN if resp_rate >= 20 else C_RED)
                with c4: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점")
                with c5: kpi_card("친절/만족",
                                  f"{avg_kind or '-'} / {avg_satis or '-'}")
                with c6: kpi_card("부정응답",  f"{neg_cnt}건", delta_color=C_RED)

                st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
                d1, d2 = st.columns(2)

                with d1:
                    section_title("채널별 응답률 및 점수", "📡")
                    ch_rate_d  = calc_response_rate(df_day_all, df_day_kpi, "채널_구분")
                    ch_score_d = pivot_avg(df_day_kpi, "채널_구분")
                    if not ch_score_d.empty and "채널_구분" in ch_score_d.columns:
                        ch_d = ch_rate_d.merge(ch_score_d.rename(columns={"채널_구분":"구분"}),
                                               on="구분", how="left")
                        ch_d = ch_d.loc[:, ~ch_d.columns.duplicated()]
                    else:
                        ch_d = ch_rate_d
                    st.dataframe(ch_d, use_container_width=True, hide_index=True)

                    section_title("긍정/부정 분포", "😊")
                    # ✅ 개선 - 가로 bar chart
                    if "긍정부정" in df_day_kpi.columns:
                        sent_d = sentiment_summary(df_day_kpi)
                        if not sent_d.empty:
                            fig = go.Figure(go.Bar(
                                x=sent_d["건수"],
                                y=sent_d["긍정/부정"],
                                orientation="h",
                                marker=dict(
                                    color=["#22c55e","#f59e0b","#ef4444"],
                                    line=dict(color="white", width=1)
                                ),
                                text=sent_d["비율(%)"].astype(str) + "%",
                                textposition="outside",
                                textfont=dict(size=12),
                            ))
                            fig.update_layout(
                                height=180,
                                margin=dict(l=10, r=40, t=10, b=10),
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                                yaxis=dict(showgrid=False),
                                font=dict(family="Inter, Noto Sans KR", size=12),
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(sentiment_summary(df_day_kpi), use_container_width=True, hide_index=True)

                with d2:
                    section_title("브랜드별 평균 점수", "🏷️")
                    if "브랜드" in df_day_kpi.columns:
                        st.dataframe(pivot_avg(df_day_kpi, "브랜드"),
                                     use_container_width=True, hide_index=True)

                    section_title("상담사별 점수 (재직자)", "👤")
                    if "상담사" in df_day_kpi.columns:
                        st.dataframe(pivot_avg(df_day_kpi, "상담사", agent_filter=True),
                                     use_container_width=True, hide_index=True)

                if "긍정부정" in df_day_kpi.columns:
                    neg_d = df_day_kpi[df_day_kpi["긍정부정"] == "부정"]
                    if not neg_d.empty:
                        section_title(f"부정 응답 상세 ({len(neg_d)}건)", "⚠️")
                        cols = get_display_cols(neg_d, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식","긍정부정"])
                        st.dataframe(neg_d[cols], use_container_width=True, hide_index=True)

                section_title("전체 응답 목록 (일자)", "📋")
                preferred = ["회신일","상담사","브랜드","채널_구분","상담유형대",
                             "친절점수","만족점수","최종점수","주관식","긍정부정"]
                cols = get_display_cols(df_day_kpi, preferred)
                st.dataframe(df_day_kpi[cols].reset_index(drop=True),
                             use_container_width=True, hide_index=True)

    # ── WEEKLY ──
    with tab_weekly:
        if not selected_week:
            st.info("👆 사이드바 상단 **주차(선택)** 에서 주차를 선택하세요.")
        else:
            df_week_all = filter_by_week_sent(df_all, selected_week)
            df_week_kpi = filter_by_week(df_scored_all, selected_week, week_col="회신주차_정제")
            if "긍정부정" not in df_week_kpi.columns and not df_week_kpi.empty:
                df_week_kpi = add_sentiment_column(df_week_kpi)

            if df_week_kpi.empty:
                st.warning(f"'{selected_week}' 에 해당하는 응답 데이터가 없습니다.")
            else:
                total_sent   = len(df_week_all)
                total_scored = len(df_week_kpi)
                resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0
                avg_final    = safe_mean(df_week_kpi, "최종점수")
                avg_kind     = safe_mean(df_week_kpi, "친절점수")
                avg_satis    = safe_mean(df_week_kpi, "만족점수")
                neg_cnt      = len(df_week_kpi[df_week_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_week_kpi.columns else 0
                gap_cnt      = len(detect_gaps(df_week_kpi))

                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: kpi_card("발송건수",   f"{total_sent:,}건")
                with c2: kpi_card("응답건수",   f"{total_scored:,}건")
                with c3: kpi_card("응답률",     f"{resp_rate}%",
                                  delta_color=C_GREEN if resp_rate >= 20 else C_RED)
                with c4: kpi_card("최종점수",   "-" if avg_final is None else f"{avg_final}점")
                with c5: kpi_card("부정응답",   f"{neg_cnt}건", delta_color=C_RED)
                with c6: kpi_card("점수갭",     f"{gap_cnt}건", delta_color=C_AMBER)

                st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

                section_title("3주 트렌드 (선택주차 + 이전 2주)", "📈")
                week_trend = build_week_trend_data(df_all, df_scored_all, selected_week)
                labels     = week_trend["labels"]
                rate_vals  = week_trend["rate_vals"]
                sc_rows    = week_trend["score_rows"]
                sc_keys    = [k for k in ["친절점수","만족점수","최종점수"] if k in sc_rows]

                trend_rows = []
                for i, lbl in enumerate(labels):
                    r = {"주차": lbl, "응답률(%)": rate_vals[i] if rate_vals[i] is not None else "-"}
                    for k in sc_keys:
                        r[k] = sc_rows[k][i] if i < len(sc_rows[k]) and sc_rows[k][i] is not None else "-"
                    trend_rows.append(r)
                st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

                # ✅ shadcn/ui 개선 - 트렌드 차트
                wt1, wt2 = st.columns(2)
                with wt1:
                    rr_df = pd.DataFrame({"주차": labels, "응답률(%)": rate_vals}).dropna()
                    if not rr_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=rr_df["주차"],
                            y=rr_df["응답률(%)"],
                            marker=dict(
                                color=["rgba(99,102,241,0.4)","rgba(99,102,241,0.7)","#6366f1"],
                                line=dict(color="white", width=1)
                            ),
                            text=rr_df["응답률(%)"].astype(str) + "%",
                            textposition="outside",
                            textfont=dict(size=12),
                        ))
                        fig.update_layout(
                            title=dict(text="주차별 응답률", font=dict(size=13, color="#0f172a"), x=0.02),
                            height=280,
                            margin=dict(l=10, r=10, t=40, b=10),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=12),
                            showlegend=False,
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with wt2:
                    sc_plot = [{"주차": labels[i], "지표": k, "점수": sc_rows[k][i]}
                               for k in sc_keys for i in range(len(labels))
                               if sc_rows[k][i] is not None]
                    if sc_plot:
                        sc_df = pd.DataFrame(sc_plot)
                        fig = go.Figure()
                        color_map_w = {
                            "친절점수": "#f59e0b",
                            "만족점수": "#22c55e",
                            "최종점수": "#6366f1",
                        }
                        for metric in sc_keys:
                            sub = sc_df[sc_df["지표"] == metric]
                            fig.add_trace(go.Scatter(
                                x=sub["주차"], y=sub["점수"],
                                mode="lines+markers",
                                name=metric,
                                line=dict(color=color_map_w.get(metric, "#6366f1"), width=2.5),
                                marker=dict(size=10, line=dict(color="white", width=2)),
                            ))
                        fig.update_layout(
                            title=dict(text="주차별 점수 트렌드", font=dict(size=13, color="#0f172a"), x=0.02),
                            height=280,
                            margin=dict(l=10, r=10, t=40, b=30),
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=12),
                            legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center",
                                        font=dict(size=11)),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
                w1, w2 = st.columns(2)

                with w1:
                    section_title("채널별 응답률 및 점수", "📡")
                    ch_rate_w  = calc_response_rate(df_week_all, df_week_kpi, "채널_구분")
                    ch_score_w = pivot_avg(df_week_kpi, "채널_구분")
                    if not ch_score_w.empty and "채널_구분" in ch_score_w.columns:
                        ch_w = ch_rate_w.merge(ch_score_w.rename(columns={"채널_구분":"구분"}),
                                               on="구분", how="left")
                        ch_w = ch_w.loc[:, ~ch_w.columns.duplicated()]
                    else:
                        ch_w = ch_rate_w
                    st.dataframe(ch_w, use_container_width=True, hide_index=True)

                    section_title("긍정/부정 분포", "😊")
                    st.dataframe(sentiment_summary(df_week_kpi), use_container_width=True, hide_index=True)

                with w2:
                    section_title("브랜드별 평균 점수", "🏷️")
                    if "브랜드" in df_week_kpi.columns:
                        br_w = pivot_avg(df_week_kpi, "브랜드")
                        st.dataframe(br_w, use_container_width=True, hide_index=True)

                    section_title("상담사별 점수 (재직자)", "👤")
                    if "상담사" in df_week_kpi.columns:
                        ag_w = pivot_avg(df_week_kpi, "상담사", agent_filter=True)
                        st.dataframe(ag_w, use_container_width=True, hide_index=True)

                # ✅ shadcn/ui 개선 - 키워드 가로 막대 차트
                section_title("키워드 TOP 20 (주차)", "🔑")
                kws = extract_keywords(df_week_kpi, 20)
                if kws:
                    kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
                    fig = go.Figure(go.Bar(
                        x=kdf["빈도"],
                        y=kdf["키워드"],
                        orientation="h",
                        marker=dict(
                            color=kdf["빈도"],
                            colorscale=[[0,"rgba(99,102,241,0.3)"],[1,"#6366f1"]],
                            showscale=False,
                            line=dict(color="white", width=0.5),
                        ),
                        text=kdf["빈도"],
                        textposition="outside",
                        textfont=dict(size=11),
                    ))
                    fig.update_layout(
                        height=500,
                        margin=dict(l=10, r=40, t=10, b=10),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=12),
                        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                gap_w = detect_gaps(df_week_kpi)
                if not gap_w.empty:
                    section_title(f"친절↔만족 점수 갭 분석 ({len(gap_w)}건)", "⚠️")
                    cols = get_display_cols(gap_w, ["상담사","브랜드","채널_구분",
                                                    "친절점수","만족점수","최종점수","갭(친절-만족)","주관식"])
                    st.dataframe(gap_w[cols], use_container_width=True, hide_index=True)

                if "긍정부정" in df_week_kpi.columns:
                    neg_w = df_week_kpi[df_week_kpi["긍정부정"] == "부정"]
                    if not neg_w.empty:
                        section_title(f"부정 응답 상세 ({len(neg_w)}건)", "⚠️")
                        cols = get_display_cols(neg_w, ["회신일","상담사","브랜드","채널_구분",
                                                        "최종점수","주관식","긍정부정"])
                        st.dataframe(neg_w[cols], use_container_width=True, hide_index=True)

                section_title("전체 응답 목록 (주차)", "📋")
                preferred_w = ["회신주차_정제","회신일","상담사","브랜드","채널_구분",
                               "상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"]
                cols = get_display_cols(df_week_kpi, preferred_w)
                st.dataframe(df_week_kpi[cols].reset_index(drop=True),
                             use_container_width=True, hide_index=True)


# ── 13-3. 점수분석 (그래프 개선) ─────────────────────────────────
def page_scores(df_m):
    page_header("점수 분석")

    for gcol, title, do_af in [
        ("상담사",     "상담사별 (재직/제외필터)", True),
        ("브랜드",     "브랜드별",               False),
        ("상담유형대", "상담유형(대)별",          False),
        ("채널_구분",  "채널별",                 False),
        ("근속",       "근속별",                 False),
    ]:
        if gcol not in df_m.columns:
            continue
        section_title(title, "📊")
        piv = pivot_avg(df_m, gcol, agent_filter=do_af)
        st.dataframe(piv, use_container_width=True, hide_index=True)

        if "최종점수" in df_m.columns:
            src  = _agent_filter(df_m) if do_af else df_m
            grp  = (src.groupby(gcol)["최종점수"]
                       .mean().round(1)
                       .sort_values(ascending=False)
                       .head(30)
                       .reset_index())
            grp.columns = [gcol, "최종점수(평균)"]

            # ✅ shadcn/ui 개선 - 수평 막대 + 색상 그라데이션
            fig = go.Figure(go.Bar(
                x=grp["최종점수(평균)"],
                y=grp[gcol],
                orientation="h",
                marker=dict(
                    color=grp["최종점수(평균)"],
                    colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                    cmin=50, cmax=100,
                    showscale=True,
                    colorbar=dict(
                        thickness=12,
                        len=0.6,
                        tickfont=dict(size=10),
                    ),
                    line=dict(color="white", width=0.5),
                ),
                text=grp["최종점수(평균)"].astype(str) + "점",
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                height=max(360, len(grp) * 32 + 60),
                margin=dict(l=10, r=60, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                           range=[0, 110]),
                yaxis=dict(showgrid=False, autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    section_title("친절↔만족 점수 갭 분석 (20점↑)", "⚠️")
    gap = detect_gaps(df_m)
    if gap.empty:
        st.caption("없음")
    else:
        cols = get_display_cols(gap, ["회신일","상담사","브랜드","채널_구분",
                                      "친절점수","만족점수","최종점수","갭(친절-만족)","주관식"])
        st.dataframe(gap[cols], use_container_width=True, hide_index=True)


# ── 13-4. 주관식분석 (그래프 개선) ──────────────────────────────
def page_verbatim(df_m):
    page_header("주관식 분석")

    section_title("긍정/부정 분포", "😊")
    v1, v2 = st.columns([1, 1.5])

    with v1:
        sent = sentiment_summary(df_m)
        st.dataframe(sent, use_container_width=True, hide_index=True)

    with v2:
        if "긍정부정" in df_m.columns:
            s = df_m["긍정부정"].value_counts().reset_index()
            s.columns = ["긍정/부정","건수"]
            # ✅ shadcn/ui 개선 - Donut + 중앙 텍스트
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=s["긍정/부정"],
                values=s["건수"],
                hole=0.65,
                marker=dict(
                    colors=["#22c55e","#f59e0b","#ef4444"],
                    line=dict(color="white", width=3)
                ),
                textfont=dict(size=12, family="Inter"),
                hovertemplate="%{label}<br>%{value}건 (%{percent})<extra></extra>",
            ))
            total_v = s["건수"].sum()
            fig.add_annotation(
                text=f"<b>{total_v}</b><br><span style='font-size:11px'>총 응답</span>",
                x=0.5, y=0.5,
                font=dict(size=18, color="#0f172a", family="Inter"),
                showarrow=False,
                align="center",
            )
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR"),
                legend=dict(
                    orientation="v", x=1.02, y=0.5,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"
                ),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ✅ shadcn/ui 개선 - 키워드 가로 막대 (그라데이션 색상)
    section_title("키워드 TOP 20", "🔑")
    kws = extract_keywords(df_m, 20)
    if kws:
        kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
        fig = go.Figure(go.Bar(
            x=kdf["빈도"],
            y=kdf["키워드"],
            orientation="h",
            marker=dict(
                color=kdf["빈도"],
                colorscale=[[0,"rgba(99,102,241,0.3)"],[1,"#6366f1"]],
                showscale=False,
                line=dict(color="white", width=0.5),
            ),
            text=kdf["빈도"],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            height=540,
            margin=dict(l=10, r=40, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, Noto Sans KR", size=12),
            xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("키워드 없음 (주관식 컬럼 확인)")

    section_title("부정 응답 상세", "⚠️")
    if "긍정부정" in df_m.columns:
        neg = df_m[df_m["긍정부정"] == "부정"]
        if neg.empty:
            st.caption("없음")
        else:
            cols = get_display_cols(neg, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식"])
            st.dataframe(neg[cols], use_container_width=True, hide_index=True)

    section_title("전체 주관식 응답", "📋")
    col = "주관식" if "주관식" in df_m.columns else None
    if col:
        vbt  = df_m[df_m[col].notna() & (df_m[col].astype(str).str.strip() != "")]
        cols = get_display_cols(vbt, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식","긍정부정"])
        st.dataframe(vbt[cols], use_container_width=True, hide_index=True)
    else:
        st.caption("주관식 컬럼 없음")


# ── 13-5. 통합분석(히트맵) (그래프 개선) ─────────────────────────
def page_integrated(df_m):
    page_header("통합 분석 (히트맵)")

    # ✅ shadcn/ui 개선 - 히트맵 스타일 강화
    def heatmap_section(df, idx, col_name, val, title, agent_filter=False, icon="🗺️"):
        src = _agent_filter(df) if agent_filter else df
        if src.empty or idx not in src.columns or col_name not in src.columns or val not in src.columns:
            return
        section_title(title, icon)
        p = src.pivot_table(values=val, index=idx, columns=col_name, aggfunc="mean").round(1)
        if p.empty:
            st.caption("데이터 없음")
            return

        # ✅ 개선된 히트맵 - 어노테이션 + 개선된 컬러스케일
        fig = go.Figure(go.Heatmap(
            z=p.values,
            x=p.columns.tolist(),
            y=p.index.tolist(),
            text=p.values,
            texttemplate="%{text:.1f}",
            textfont=dict(size=12, color="white"),
            colorscale=[
                [0.0,  "#ef4444"],
                [0.3,  "#f97316"],
                [0.55, "#f59e0b"],
                [0.75, "#22c55e"],
                [1.0,  "#16a34a"],
            ],
            zmin=60, zmax=100,
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="점수", side="right"),
                thickness=14,
                len=0.8,
                tickfont=dict(size=11),
            ),
        ))
        fig.update_layout(
            height=max(380, len(p) * 38 + 80),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, Noto Sans KR", size=12, color="#0f172a"),
            xaxis=dict(side="bottom", tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    heatmap_section(df_m, "상담유형대", "채널_구분", "최종점수",
                    "상담유형(대) × 채널 만족도", agent_filter=False, icon="🗺️")

    heatmap_section(df_m, "상담사", "채널_구분", "최종점수",
                    "상담사 × 채널 만족도 (재직자 기준)", agent_filter=True, icon="👤")

    inq_col = next((c for c in ["문의유형","상담유형대","상담유형중"] if c in df_m.columns), None)
    if inq_col and "상담사" in df_m.columns and "최종점수" in df_m.columns:
        heatmap_section(df_m, "상담사", inq_col, "최종점수",
                        f"상담사 × {inq_col} 만족도 (재직자 기준)", agent_filter=True, icon="📋")

    heatmap_section(df_m, "브랜드", "상담유형대", "최종점수",
                    "브랜드 × 상담유형(대) 만족도", agent_filter=False, icon="🏷️")

    if "근속" in df_m.columns:
        heatmap_section(df_m, "근속", "채널_구분", "최종점수",
                        "근속 × 채널 만족도", agent_filter=False, icon="📅")

        section_title("근속별 평균 점수", "📊")
        piv = pivot_avg(df_m, "근속")
        st.dataframe(piv, use_container_width=True, hide_index=True)


# ── 13-6. Action 필요 (변경 없음) ──────────────────────────────
def page_action(df_m, df_m_all):
    page_header("Action 필요")

    act = action_needed(df_m, df_m_all)
    if act.empty:
        st.success("✅ 특이사항 없음")
    else:
        st.dataframe(act, use_container_width=True, hide_index=True)

    section_title("부정 응답 상세", "⚠️")
    if "긍정부정" in df_m.columns:
        neg = df_m[df_m["긍정부정"] == "부정"]
        if neg.empty:
            st.caption("없음")
        else:
            cols = get_display_cols(neg, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식"])
            st.dataframe(neg[cols], use_container_width=True, hide_index=True)


# ── 13-7. 일별 상담사 성과 (그래프 개선) ─────────────────────────
def page_daily_agent(df_m):
    page_header("일별 상담사 성과")

    df_v = _agent_filter(df_m)
    if df_v.empty or "회신일" not in df_v.columns or "상담사" not in df_v.columns or "최종점수" not in df_v.columns:
        st.warning("일별 상담사 성과 데이터가 부족합니다.")
        return

    pivot_df = (df_v.groupby(["회신일","상담사"])["최종점수"]
                    .mean().round(1).unstack(fill_value=None))
    if pivot_df.empty:
        st.caption("피벗 데이터 없음")
        return

    # ✅ shadcn/ui 개선 - 상담사 트렌드 라인 차트
    section_title("일자별 상담사 최종점수 트렌드", "📈")
    p_long = pivot_df.reset_index().melt(id_vars=["회신일"], var_name="상담사", value_name="최종점수")
    p_long = p_long.dropna()

    fig = go.Figure()
    agents = p_long["상담사"].unique()
    for i, agent in enumerate(agents):
        sub = p_long[p_long["상담사"] == agent]
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(
            x=sub["회신일"],
            y=sub["최종점수"],
            mode="lines+markers",
            name=agent,
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color, line=dict(color="white", width=1.5)),
        ))

    # SCORE_GOOD, SCORE_CAUTION 기준선
    fig.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e",
                  annotation_text="양호(90)", annotation_position="right",
                  line_width=1.5)
    fig.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444",
                  annotation_text="주의(70)", annotation_position="right",
                  line_width=1.5)

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=120, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Noto Sans KR", size=12, color="#0f172a"),
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor="rgba(226,232,240,0.8)",
                   tickfont=dict(size=11, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                   linecolor="rgba(226,232,240,0.8)",
                   tickfont=dict(size=11, color="#64748b"),
                   range=[0, 105]),
    )
    st.plotly_chart(fig, use_container_width=True)

    section_title("일자별 상담사 피벗 테이블", "📋")
    pivot_disp = pivot_df.reset_index()
    pivot_disp.columns.name = None
    pivot_disp = pivot_disp.where(pd.notnull(pivot_disp), other="")
    st.dataframe(pivot_disp, use_container_width=True, hide_index=True)

    # ✅ shadcn/ui 개선 - 성과 요약 + 점수 분포 차트
    section_title("상담사별 성과 요약", "👤")
    daily_avg = (df_v.groupby("상담사")["최종점수"]
                     .agg(["mean","min","max","count"]).round(1).reset_index())
    daily_avg.columns = ["상담사","평균점수","최저점수","최고점수","응답건수"]
    daily_avg["상태"] = daily_avg["평균점수"].apply(
        lambda x: "🔴 주의" if x < SCORE_CAUTION else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
    daily_avg = daily_avg.sort_values("평균점수", ascending=False)
    st.dataframe(daily_avg, use_container_width=True, hide_index=True)

    # ✅ 상담사별 평균점수 수평 막대 차트
    fig2 = go.Figure(go.Bar(
        x=daily_avg["평균점수"],
        y=daily_avg["상담사"],
        orientation="h",
        marker=dict(
            color=daily_avg["평균점수"],
            colorscale=[[0,"#ef4444"],[0.4,"#f59e0b"],[1,"#22c55e"]],
            cmin=50, cmax=100,
            showscale=False,
            line=dict(color="white", width=0.5),
        ),
        text=daily_avg["평균점수"].astype(str) + "점",
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig2.add_vline(x=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
    fig2.add_vline(x=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
    fig2.update_layout(
        height=max(300, len(daily_avg) * 34 + 40),
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Noto Sans KR", size=12),
        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", range=[0, 110]),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "채널_구분" in df_v.columns:
        section_title("일자별 채널별 응답 건수", "📡")
        daily_ch = (df_v.groupby(["회신일", "채널_구분"])["최종점수"]
                        .count().unstack(fill_value=0).reset_index())
        daily_ch.columns.name = None
        st.dataframe(daily_ch, use_container_width=True, hide_index=True)


# ── 13-8. 70점 미만 전체 (그래프 개선) ───────────────────────────
def page_low_scores(df_scored_all, target_month=None):
    page_header("70점 미만 전체 목록")

    src = fm(df_scored_all, target_month) if target_month else df_scored_all.copy()
    if "최종점수" not in src.columns:
        st.warning("최종점수 컬럼 없음")
        return

    low = src[src["최종점수"] < SCORE_CAUTION].copy().sort_values("최종점수", ascending=True)
    if low.empty:
        st.success("70점 미만 데이터 없음")
        return

    # ✅ shadcn/ui 개선 - 요약 KPI 카드
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("70점 미만 건수", f"{len(low):,}건", delta_color=C_RED)
    with k2:
        kpi_card("평균 점수", f"{low['최종점수'].mean():.1f}점", delta_color=C_AMBER)
    with k3:
        if "상담사" in low.columns:
            agent_cnt = low["상담사"].nunique()
            kpi_card("해당 상담사 수", f"{agent_cnt:,}명", delta_color=C_AMBER)

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # ✅ shadcn/ui 개선 - 점수 분포 히스토그램 + 상담사별 건수 차트
    ch1, ch2 = st.columns(2)

    with ch1:
        section_title("점수 구간별 분포", "📊")
        fig = go.Figure(go.Histogram(
            x=low["최종점수"],
            nbinsx=14,
            marker=dict(
                color="#ef4444",
                opacity=0.85,
                line=dict(color="white", width=1),
            ),
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, Noto Sans KR", size=12),
            xaxis=dict(showgrid=False, title="최종점수",
                       tickfont=dict(size=11, color="#64748b")),
            yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                       title="건수", tickfont=dict(size=11, color="#64748b")),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        section_title("상담사별 70점 미만 건수 TOP 15", "👤")
        if "상담사" in low.columns:
            agent_low = (low.groupby("상담사")["최종점수"]
                            .count()
                            .sort_values(ascending=False)
                            .head(15)
                            .reset_index())
            agent_low.columns = ["상담사", "건수"]
            fig2 = go.Figure(go.Bar(
                x=agent_low["건수"],
                y=agent_low["상담사"],
                orientation="h",
                marker=dict(
                    color=agent_low["건수"],
                    colorscale=[[0, "rgba(239,68,68,0.4)"], [1, "#ef4444"]],
                    showscale=False,
                    line=dict(color="white", width=0.5),
                ),
                text=agent_low["건수"],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig2.update_layout(
                height=280,
                margin=dict(l=10, r=40, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                yaxis=dict(showgrid=False, autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ✅ 브랜드/채널별 분포 차트
    ch3, ch4 = st.columns(2)

    with ch3:
        if "브랜드" in low.columns:
            section_title("브랜드별 70점 미만 건수", "🏷️")
            brand_low = (low.groupby("브랜드")["최종점수"]
                            .count()
                            .sort_values(ascending=False)
                            .reset_index())
            brand_low.columns = ["브랜드", "건수"]
            fig3 = go.Figure(go.Bar(
                x=brand_low["브랜드"],
                y=brand_low["건수"],
                marker=dict(
                    color=brand_low["건수"],
                    colorscale=[[0, "rgba(245,158,11,0.4)"], [1, "#f59e0b"]],
                    showscale=False,
                    line=dict(color="white", width=1),
                ),
                text=brand_low["건수"],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig3.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        if "채널_구분" in low.columns:
            section_title("채널별 70점 미만 건수", "📡")
            ch_low = (low.groupby("채널_구분")["최종점수"]
                         .count()
                         .reset_index())
            ch_low.columns = ["채널", "건수"]
            fig4 = go.Figure(go.Pie(
                labels=ch_low["채널"],
                values=ch_low["건수"],
                hole=0.55,
                marker=dict(
                    colors=["#6366f1", "#f59e0b", "#ef4444", "#22c55e"],
                    line=dict(color="white", width=3),
                ),
                textfont=dict(size=12, family="Inter"),
                hovertemplate="%{label}<br>%{value}건 (%{percent})<extra></extra>",
            ))
            fig4.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR"),
                legend=dict(
                    orientation="h", y=-0.15, x=0.5, xanchor="center",
                    font=dict(size=11),
                ),
                showlegend=True,
            )
            st.plotly_chart(fig4, use_container_width=True)

    # 전체 목록 테이블
    section_title(f"70점 미만 전체 목록 ({len(low):,}건)", "📋")
    preferred = ["회신월_정제","회신일","상담사","브랜드","채널_구분",
                 "상담유형대","근속","친절점수","만족점수","최종점수","주관식","긍정부정"]
    cols = get_display_cols(low, preferred)
    st.dataframe(low[cols].reset_index(drop=True), use_container_width=True, hide_index=True)


# ── 13-9. 검색 (그래프 개선) ────────────────────────────────────
def page_search(df_scored_all, df_all):
    page_header("검색", "상담KEY 또는 상담사 이름으로 원천 데이터를 검색합니다")

    s1, s2 = st.columns([2, 1])
    with s1:
        search_query = st.text_input(
            "🔍 검색어 입력",
            placeholder="상담KEY 또는 상담사 이름을 입력하세요",
            label_visibility="collapsed",
        )
    with s2:
        search_type = st.selectbox(
            "검색 유형",
            ["상담KEY + 상담사 이름 (전체)", "상담KEY만", "상담사 이름만"],
            label_visibility="collapsed",
        )

    if not search_query.strip():
        st.info("검색어를 입력하면 해당 데이터를 바로 보여드립니다.")

        section_title("전체 데이터 미리보기 (최근 50건)", "📋")
        preferred = ["회신일","상담사","브랜드","채널_구분","상담유형대",
                     "친절점수","만족점수","최종점수","주관식","긍정부정"]
        cols = get_display_cols(df_scored_all, preferred)
        st.dataframe(
            df_scored_all[cols].tail(50).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
        return

    q = search_query.strip()

    mask = pd.Series([False] * len(df_scored_all), index=df_scored_all.index)

    if search_type in ["상담KEY + 상담사 이름 (전체)", "상담KEY만"]:
        if "상담KEY" in df_scored_all.columns:
            mask = mask | df_scored_all["상담KEY"].astype(str).str.contains(
                q, case=False, na=False)

    if search_type in ["상담KEY + 상담사 이름 (전체)", "상담사 이름만"]:
        if "상담사" in df_scored_all.columns:
            mask = mask | df_scored_all["상담사"].astype(str).str.contains(
                q, case=False, na=False)

    result = df_scored_all[mask].copy()

    if result.empty:
        st.warning(f"'{q}' 에 해당하는 데이터가 없습니다.")
        return

    st.success(f"✅ '{q}' 검색 결과: **{len(result):,}건**")

    # ✅ shadcn/ui 개선 - 검색 결과 KPI
    if len(result) > 0:
        k1, k2, k3, k4, k5 = st.columns(5)
        avg_f = safe_mean(result, "최종점수")
        avg_k = safe_mean(result, "친절점수")
        avg_s = safe_mean(result, "만족점수")
        neg_c = (len(result[result["긍정부정"] == "부정"])
                 if "긍정부정" in result.columns else 0)
        with k1: kpi_card("검색 결과",  f"{len(result):,}건")
        with k2: kpi_card("최종점수",   "-" if avg_f is None else f"{avg_f}점")
        with k3: kpi_card("친절점수",   "-" if avg_k is None else f"{avg_k}점")
        with k4: kpi_card("만족점수",   "-" if avg_s is None else f"{avg_s}점")
        with k5: kpi_card("부정응답",   f"{neg_c}건", delta_color=C_RED)

        st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # 검색 결과 테이블
    preferred = ["회신월_정제","회신일","상담사","브랜드","채널_구분",
                 "상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"]
    cols = get_display_cols(result, preferred)
    st.dataframe(
        result[cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    # ✅ shadcn/ui 개선 - 긍정/부정 분포 + 점수 분포 차트
    if "긍정부정" in result.columns and result["긍정부정"].notna().any():
        sr1, sr2 = st.columns(2)

        with sr1:
            section_title("긍정/부정 분포", "😊")
            sent_r = sentiment_summary(result)
            st.dataframe(sent_r, use_container_width=True, hide_index=True)

            if not sent_r.empty:
                fig_s = go.Figure(go.Bar(
                    x=sent_r["건수"],
                    y=sent_r["긍정/부정"],
                    orientation="h",
                    marker=dict(
                        color=["#22c55e", "#f59e0b", "#ef4444"],
                        line=dict(color="white", width=1),
                    ),
                    text=sent_r["비율(%)"].astype(str) + "%",
                    textposition="outside",
                    textfont=dict(size=12),
                ))
                fig_s.update_layout(
                    height=200,
                    margin=dict(l=10, r=50, t=10, b=10),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Inter, Noto Sans KR", size=12),
                    xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True)

        with sr2:
            section_title("최종점수 분포", "📊")
            if "최종점수" in result.columns and result["최종점수"].notna().any():
                # ✅ 히스토그램 + 기준선 표시
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=result["최종점수"],
                    nbinsx=20,
                    marker=dict(
                        color="#6366f1",
                        opacity=0.8,
                        line=dict(color="white", width=1),
                    ),
                    name="점수 분포",
                ))
                fig_h.add_vline(
                    x=SCORE_GOOD,
                    line_dash="dash",
                    line_color="#22c55e",
                    line_width=1.5,
                    annotation_text=f"양호({SCORE_GOOD})",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color="#22c55e"),
                )
                fig_h.add_vline(
                    x=SCORE_CAUTION,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=1.5,
                    annotation_text=f"주의({SCORE_CAUTION})",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color="#ef4444"),
                )
                fig_h.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=20, b=10),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Inter, Noto Sans KR", size=12),
                    xaxis=dict(
                        showgrid=False,
                        title="최종점수",
                        tickfont=dict(size=11, color="#64748b"),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(226,232,240,0.6)",
                        title="건수",
                        tickfont=dict(size=11, color="#64748b"),
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 14. 사이드바 네비게이션 (Compact 개선 + 기간선택 상단 + 신규 메뉴)
# ══════════════════════════════════════════════════════════════════

# ── 기존 9개 메뉴 + 신규 2개(인사이트, 교육자료) ──
MENU_ITEMS = [
    {"key": "개요",         "icon": "🏠", "label": "개요"},
    {"key": "일자주차",     "icon": "📅", "label": "일자·주차"},
    {"key": "점수분석",     "icon": "📊", "label": "점수분석"},
    {"key": "주관식분석",   "icon": "💬", "label": "주관식"},
    {"key": "히트맵",       "icon": "🗺️", "label": "히트맵"},
    {"key": "Action필요",   "icon": "⚠️", "label": "Action"},
    {"key": "상담사성과",   "icon": "👤", "label": "상담사성과"},
    {"key": "70점미만",     "icon": "🔴", "label": "70점미만"},
    {"key": "검색",         "icon": "🔍", "label": "검색"},
    {"key": "인사이트",     "icon": "💡", "label": "인사이트"},
    {"key": "교육자료",     "icon": "🎓", "label": "교육자료"},
]

# ── 사이드바: 기간선택(상단) + compact 메뉴 ──
def render_sidebar_nav(current_menu: str, available_months=None,
                       df_scored_all=None, df_all=None):
    MISSING_SET = {"", "nan", "NaT", "None", "NaN", "<NA>", "NA"}

    with st.sidebar:
        # ── 헤더 ──
        st.markdown("""
        <div style="padding:10px 12px 8px;
                    border-bottom:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:13px;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">
                📊 CSAT Dashboard
            </div>
            <div style="font-size:10px;color:rgba(148,163,184,0.6);margin-top:1px;">
                고객만족도 분석 시스템
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ════════════════════════════════════
        # ① 기간 선택 — 사이드바 최상단
        # ════════════════════════════════════
        st.markdown("""
        <div style="padding:8px 12px 4px;font-size:9px;font-weight:800;
                    color:rgba(148,163,184,0.55);letter-spacing:1px;
                    text-transform:uppercase;">
            📆 기간 선택
        </div>
        """, unsafe_allow_html=True)

        # 월 선택
        if available_months:
            target_month = st.selectbox(
                "월(필수)", available_months,
                index=len(available_months) - 1,
                key="sel_month",
            )
        else:
            target_month = st.text_input(
                "월(예: 2026-01)", value="", key="sel_month_txt")

        # 일자 선택
        available_dates = []
        if df_scored_all is not None and "회신일" in df_scored_all.columns:
            available_dates = sorted([
                str(d) for d in df_scored_all["회신일"].dropna().unique()
                if str(d) not in MISSING_SET
            ])
        selected_date = st.selectbox(
            "일자(선택)", [""] + available_dates, index=0, key="sel_date")
        selected_date = selected_date or None

        # 주차 선택
        available_weeks = []
        if df_scored_all is not None:
            week_col = ("회신주차_정제" if "회신주차_정제" in df_scored_all.columns else "회신주차")
            if week_col in df_scored_all.columns:
                available_weeks = sorted([
                    str(w) for w in df_scored_all[week_col].dropna().unique()
                    if str(w) not in MISSING_SET
                ])
        selected_week = st.selectbox(
            "주차(선택)", [""] + available_weeks, index=0, key="sel_week")
        selected_week = selected_week or None

        st.markdown(
            "<hr style='margin:6px 10px 4px;border-color:rgba(255,255,255,0.08);'>",
            unsafe_allow_html=True,
        )

        # ════════════════════════════════════
        # ② 메뉴 — Compact (아이콘 + 짧은 레이블)
        # ════════════════════════════════════
        st.markdown("""
        <div style="padding:4px 12px 3px;font-size:9px;font-weight:800;
                    color:rgba(148,163,184,0.55);letter-spacing:1px;
                    text-transform:uppercase;">
            메뉴
        </div>
        """, unsafe_allow_html=True)

        if "menu" not in st.session_state:
            st.session_state["menu"] = MENU_ITEMS[0]["key"]

        for item in MENU_ITEMS:
            key      = item["key"]
            icon     = item["icon"]
            label    = item["label"]
            is_active = st.session_state["menu"] == key
            display  = f"{icon} {label}"

            if is_active:
                st.markdown(f"""
                <div style="margin:1px 6px;padding:6px 10px;
                            border-radius:6px;
                            background:rgba(99,102,241,0.22);
                            border-left:2px solid #6366f1;position:relative;">
                    <span style="font-size:12px;font-weight:600;
                                 color:#a5b4fc;">{display}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="margin:1px 6px;padding:6px 10px;
                            border-radius:6px;
                            border-left:2px solid transparent;">
                    <span style="font-size:12px;font-weight:400;
                                 color:rgba(203,213,225,0.8);">{display}</span>
                </div>
                """, unsafe_allow_html=True)

            if st.button(display, key=f"nav_{key}", use_container_width=True):
                st.session_state["menu"] = key
                st.rerun()

        st.markdown(
            "<hr style='margin:6px 10px 4px;border-color:rgba(255,255,255,0.08);'>",
            unsafe_allow_html=True,
        )

        # ── 기준 안내 (compact) ──
        st.markdown(f"""
        <div style="padding:2px 12px 6px;font-size:9px;
                    color:rgba(148,163,184,0.5);line-height:1.6;">
          🟢 양호 ≥{SCORE_GOOD} &nbsp;|&nbsp; 🔴 주의 &lt;{SCORE_CAUTION}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 새로고침", use_container_width=True, key="refresh_btn"):
            load_from_gsheets.clear()
            st.rerun()

    return st.session_state["menu"], target_month, selected_date, selected_week


# ══════════════════════════════════════════════════════════════════
# 13-10. 인사이트 (QA강사/센터장 회의용 종합 인사이트)
# ══════════════════════════════════════════════════════════════════
def page_insight(df_all, df_scored, df_scored_all, available_months, target_month):
    page_header("💡 QA 인사이트 리포트",
                f"센터장·QA팀 회의용 종합 분석  |  기준월: {target_month}")

    df_m     = fm(df_scored,     target_month)
    df_m_kpi = fm(df_scored_all, target_month)
    df_m_all = fm_sent(df_all,   target_month)

    # ── 이전월 비교 ──
    sorted_m  = sorted([m for m in available_months if m <= target_month]) if target_month else []
    prev_m    = sorted_m[-2] if len(sorted_m) >= 2 else None
    df_prev   = fm(df_scored_all, prev_m) if prev_m else pd.DataFrame()

    # ── 핵심 KPI 요약 ──
    section_title("이달의 핵심 지표 요약", "📌")
    total_sent   = len(df_m_all)
    total_scored = len(df_m_kpi)
    resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0
    avg_final    = safe_mean(df_m_kpi, "최종점수")
    avg_kind     = safe_mean(df_m_kpi, "친절점수")
    avg_satis    = safe_mean(df_m_kpi, "만족점수")
    prev_final   = safe_mean(df_prev, "최종점수") if not df_prev.empty else None
    _, d_score_str = calc_mom(avg_final, prev_final, is_pp=False)
    neg_cnt = len(df_m_kpi[df_m_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_m_kpi.columns else 0
    pos_cnt = len(df_m_kpi[df_m_kpi["긍정부정"] == "긍정"]) if "긍정부정" in df_m_kpi.columns else 0
    gap_cnt = len(detect_gaps(df_m_kpi))

    i1, i2, i3, i4, i5 = st.columns(5)
    with i1: kpi_card("응답건수",  f"{total_scored:,}건")
    with i2: kpi_card("응답률",    f"{resp_rate}%")
    with i3: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점", d_score_str, dcol(d_score_str))
    with i4: kpi_card("긍정 응답", f"{pos_cnt:,}건")
    with i5: kpi_card("부정 응답", f"{neg_cnt:,}건")

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # ── 월별 흐름 & 채널별 비교 ──
    col_a, col_b = st.columns(2)

    with col_a:
        section_title("월별 최종점수 추이 (전체)", "📈")
        _, monthly_score_df = compute_monthly_trends(df_all, df_scored_all)
        if not monthly_score_df.empty and "최종점수" in monthly_score_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_score_df["월"], y=monthly_score_df["최종점수"],
                mode="lines+markers+text",
                text=monthly_score_df["최종점수"].astype(str),
                textposition="top center",
                textfont=dict(size=11, color="#6366f1"),
                line=dict(color="#6366f1", width=3),
                marker=dict(size=9, color="#6366f1", line=dict(color="white", width=2)),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
            ))
            fig.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e",
                          annotation_text="양호(90)", line_width=1.5)
            fig.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444",
                          annotation_text="주의(70)", line_width=1.5)
            fig.update_layout(
                height=300, margin=dict(l=10,r=80,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#64748b")),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                           range=[50, 105], tickfont=dict(size=11, color="#64748b")),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("데이터 없음")

    with col_b:
        section_title("채널별 점수 비교 (이번 달)", "📡")
        if "채널_구분" in df_m_kpi.columns:
            ch_avg = pivot_avg(df_m_kpi, "채널_구분")
            if not ch_avg.empty and "최종점수" in ch_avg.columns:
                fig2 = go.Figure(go.Bar(
                    x=ch_avg["채널_구분"], y=ch_avg["최종점수"],
                    marker=dict(
                        color=ch_avg["최종점수"],
                        colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                        cmin=60, cmax=100, showscale=False,
                        line=dict(color="white", width=1),
                    ),
                    text=ch_avg["최종점수"].astype(str) + "점",
                    textposition="outside", textfont=dict(size=12),
                ))
                fig2.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
                fig2.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
                fig2.update_layout(
                    height=300, margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter, Noto Sans KR", size=12),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", range=[0,110]),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("채널 데이터 없음")

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # ── 브랜드별 성과 비교 ──
    section_title("브랜드별 성과 비교 (이번 달)", "🏷️")
    if "브랜드" in df_m_kpi.columns:
        br_avg = pivot_avg(df_m_kpi, "브랜드")
        if not br_avg.empty:
            col_br1, col_br2 = st.columns([1.2, 2])
            with col_br1:
                st.dataframe(br_avg, use_container_width=True, hide_index=True)
            with col_br2:
                if "최종점수" in br_avg.columns:
                    fig_br = go.Figure(go.Bar(
                        x=br_avg["최종점수"], y=br_avg["브랜드"],
                        orientation="h",
                        marker=dict(
                            color=br_avg["최종점수"],
                            colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                            cmin=60, cmax=100, showscale=False,
                            line=dict(color="white", width=0.5),
                        ),
                        text=br_avg["최종점수"].astype(str) + "점",
                        textposition="outside", textfont=dict(size=11),
                    ))
                    fig_br.update_layout(
                        height=max(250, len(br_avg)*34+40),
                        margin=dict(l=10,r=60,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=12),
                        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", range=[0,110]),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                    )
                    st.plotly_chart(fig_br, use_container_width=True)

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # ── 긍정/부정 트렌드 (월별) ──
    section_title("월별 긍정/부정 응답 추이", "😊")
    if "회신월_정제" in df_scored_all.columns and "긍정부정" in df_scored_all.columns:
        all_months_sorted = sorted([m for m in df_scored_all["회신월_정제"].dropna().unique() if m != "미확인"])
        sent_rows = []
        for mo in all_months_sorted:
            sub = fm(df_scored_all, mo)
            t = len(sub)
            if t == 0:
                continue
            pos = len(sub[sub["긍정부정"] == "긍정"])
            neg = len(sub[sub["긍정부정"] == "부정"])
            neu = len(sub[sub["긍정부정"] == "중립"])
            sent_rows.append({"월": mo,
                              "긍정(%)": round(pos/t*100, 1),
                              "중립(%)": round(neu/t*100, 1),
                              "부정(%)": round(neg/t*100, 1)})
        if sent_rows:
            s_df = pd.DataFrame(sent_rows)
            fig_s = go.Figure()
            for col_name, color in [("긍정(%)", "#22c55e"), ("중립(%)", "#f59e0b"), ("부정(%)", "#ef4444")]:
                if col_name in s_df.columns:
                    fig_s.add_trace(go.Scatter(
                        x=s_df["월"], y=s_df[col_name],
                        mode="lines+markers", name=col_name,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, color=color, line=dict(color="white", width=2)),
                    ))
            fig_s.update_layout(
                height=280, margin=dict(l=10,r=10,t=10,b=30),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center", font=dict(size=11)),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#64748b")),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                           tickfont=dict(size=11, color="#64748b"), title="%"),
            )
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # ── 이달의 주요 이슈 / 시사점 ──
    section_title("이달의 주요 이슈 & 시사점", "🚨")
    act = action_needed(df_m, df_m_all)
    if act.empty:
        st.success("✅ 이번 달 특이 이슈 없음 — 전반적으로 양호한 상태입니다.")
    else:
        # 우선순위별 색상 구분
        priority_color = {
            "🔴 긴급": "background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;",
            "🟡 주의": "background:rgba(245,158,11,0.08);border-left:3px solid #f59e0b;",
            "🟠 개선": "background:rgba(249,115,22,0.08);border-left:3px solid #f97316;",
        }
        for _, row in act.iterrows():
            style = priority_color.get(row.get("우선순위",""), "background:#f8fafc;border-left:3px solid #64748b;")
            st.markdown(f"""
            <div style="{style} padding:10px 16px;border-radius:0 8px 8px 0;margin-bottom:6px;">
                <span style="font-size:12px;font-weight:700;color:#0f172a;">[{row.get('구분','')}] {row.get('항목','')}</span>
                <span style="font-size:11px;color:#64748b;margin-left:10px;">{row.get('내용','')}</span>
                <span style="font-size:11px;font-weight:700;float:right;">{row.get('우선순위','')}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # ── 상담사 상태 분포 (신호등) ──
    section_title("상담사 상태 분포 (신호등)", "🚦")
    if "상담사" in df_m.columns and "최종점수" in df_m.columns:
        ag_perf = _agent_filter(df_m).groupby("상담사")["최종점수"].mean().round(1).reset_index()
        ag_perf.columns = ["상담사", "평균점수"]
        red_agents   = ag_perf[ag_perf["평균점수"] < SCORE_CAUTION]
        amber_agents = ag_perf[(ag_perf["평균점수"] >= SCORE_CAUTION) & (ag_perf["평균점수"] < SCORE_GOOD)]
        green_agents = ag_perf[ag_perf["평균점수"] >= SCORE_GOOD]

        sg1, sg2, sg3 = st.columns(3)
        with sg1:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);
                        border-radius:10px;padding:14px 16px;">
                <div style="font-size:11px;font-weight:700;color:#dc2626;margin-bottom:8px;">
                    🔴 즉시 코칭 필요 ({len(red_agents)}명) — 70점 미만
                </div>
                {"".join([f'<div style="font-size:12px;color:#0f172a;padding:2px 0;">{r["상담사"]} <b>{r["평균점수"]}점</b></div>' for _,r in red_agents.iterrows()]) if not red_agents.empty else '<div style="font-size:12px;color:#64748b;">해당 없음</div>'}
            </div>
            """, unsafe_allow_html=True)
        with sg2:
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2);
                        border-radius:10px;padding:14px 16px;">
                <div style="font-size:11px;font-weight:700;color:#d97706;margin-bottom:8px;">
                    🟡 지속 모니터링 ({len(amber_agents)}명) — 70~89점
                </div>
                {"".join([f'<div style="font-size:12px;color:#0f172a;padding:2px 0;">{r["상담사"]} <b>{r["평균점수"]}점</b></div>' for _,r in amber_agents.iterrows()]) if not amber_agents.empty else '<div style="font-size:12px;color:#64748b;">해당 없음</div>'}
            </div>
            """, unsafe_allow_html=True)
        with sg3:
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);
                        border-radius:10px;padding:14px 16px;">
                <div style="font-size:11px;font-weight:700;color:#16a34a;margin-bottom:8px;">
                    🟢 양호 ({len(green_agents)}명) — 90점 이상
                </div>
                {"".join([f'<div style="font-size:12px;color:#0f172a;padding:2px 0;">{r["상담사"]} <b>{r["평균점수"]}점</b></div>' for _,r in green_agents.iterrows()]) if not green_agents.empty else '<div style="font-size:12px;color:#64748b;">해당 없음</div>'}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("상담사/점수 데이터 없음")


# ══════════════════════════════════════════════════════════════════
# 13-11. 교육자료 (QA강사용 코칭 리포트)
# ══════════════════════════════════════════════════════════════════
def page_education(df_m, df_scored_all, target_month):
    page_header("🎓 QA 교육·코칭 자료",
                f"QA강사 교육 준비용 분석  |  기준월: {target_month}")

    tab_coach, tab_keyword, tab_case, tab_best = st.tabs([
        "🧑‍🏫 코칭 대상 분석",
        "🔑 키워드·VOC 패턴",
        "⚠️ 개선 사례 모음",
        "🌟 우수 사례 모음",
    ])

    # ────────────────────────────────────
    # TAB 1: 코칭 대상 분석
    # ────────────────────────────────────
    with tab_coach:
        section_title("코칭 우선순위 매트릭스", "🎯")
        st.markdown("""
        <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
                    border-radius:8px;padding:10px 16px;font-size:12px;color:#0f172a;margin-bottom:12px;">
            💡 <b>활용법</b>: 점수가 낮고 응답건수가 많은 상담사를 우선 코칭 대상으로 삼으세요.
            부정응답 건수와 친절↔만족 갭이 큰 경우 VOC 기반 맞춤 교육이 효과적입니다.
        </div>
        """, unsafe_allow_html=True)

        if "상담사" in df_m.columns and "최종점수" in df_m.columns:
            src = _agent_filter(df_m)
            coach_df = src.groupby("상담사").agg(
                평균점수=("최종점수", "mean"),
                응답건수=("최종점수", "count"),
                최저점수=("최종점수", "min"),
            ).round(1).reset_index()

            if "긍정부정" in src.columns:
                neg_per_agent = src[src["긍정부정"] == "부정"].groupby("상담사").size().reset_index(name="부정건수")
                coach_df = coach_df.merge(neg_per_agent, on="상담사", how="left")
                coach_df["부정건수"] = coach_df["부정건수"].fillna(0).astype(int)

            if "친절점수" in src.columns and "만족점수" in src.columns:
                gap_df = src.copy()
                gap_df["갭"] = (gap_df["친절점수"] - gap_df["만족점수"]).abs()
                avg_gap = gap_df.groupby("상담사")["갭"].mean().round(1).reset_index(name="평균갭")
                coach_df = coach_df.merge(avg_gap, on="상담사", how="left")

            coach_df["코칭등급"] = coach_df["평균점수"].apply(
                lambda x: "🔴 즉시코칭" if x < SCORE_CAUTION else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
            coach_df = coach_df.sort_values("평균점수", ascending=True)
            st.dataframe(coach_df, use_container_width=True, hide_index=True)

            # 코칭 대상 시각화
            section_title("상담사별 점수 & 응답건수 버블 차트", "🫧")
            if not coach_df.empty:
                fig_bubble = go.Figure()
                for _, row in coach_df.iterrows():
                    color = "#ef4444" if row["평균점수"] < SCORE_CAUTION else ("#f59e0b" if row["평균점수"] < SCORE_GOOD else "#22c55e")
                    fig_bubble.add_trace(go.Scatter(
                        x=[row["응답건수"]],
                        y=[row["평균점수"]],
                        mode="markers+text",
                        text=[row["상담사"]],
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(
                            size=max(18, min(50, row["응답건수"] * 2)),
                            color=color,
                            opacity=0.75,
                            line=dict(color="white", width=2),
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>{row['상담사']}</b><br>평균점수: {row['평균점수']}<br>응답건수: {row['응답건수']}<extra></extra>",
                    ))
                fig_bubble.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e",
                                     annotation_text="양호(90)", line_width=1.5)
                fig_bubble.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444",
                                     annotation_text="주의(70)", line_width=1.5)
                fig_bubble.update_layout(
                    height=420,
                    margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter, Noto Sans KR", size=12),
                    xaxis=dict(title="응답건수", showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                    yaxis=dict(title="평균점수", showgrid=True, gridcolor="rgba(226,232,240,0.6)",
                               range=[0,110]),
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.warning("상담사/점수 데이터가 없습니다.")

        # 근속별 교육 필요도
        if "근속" in df_m.columns and "최종점수" in df_m.columns:
            section_title("근속 구간별 평균점수 (교육 필요도)", "📅")
            tenure_df = pivot_avg(df_m, "근속")
            if not tenure_df.empty:
                col_t1, col_t2 = st.columns([1, 2])
                with col_t1:
                    st.dataframe(tenure_df, use_container_width=True, hide_index=True)
                with col_t2:
                    fig_t = go.Figure(go.Bar(
                        x=tenure_df["최종점수"] if "최종점수" in tenure_df.columns else [],
                        y=tenure_df["근속"] if "근속" in tenure_df.columns else [],
                        orientation="h",
                        marker=dict(
                            color=tenure_df["최종점수"] if "최종점수" in tenure_df.columns else [],
                            colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                            cmin=60, cmax=100, showscale=False,
                            line=dict(color="white", width=0.5),
                        ),
                        text=(tenure_df["최종점수"].astype(str) + "점") if "최종점수" in tenure_df.columns else [],
                        textposition="outside", textfont=dict(size=11),
                    ))
                    fig_t.add_vline(x=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
                    fig_t.add_vline(x=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
                    fig_t.update_layout(
                        height=280, margin=dict(l=10,r=60,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=12),
                        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", range=[0,110]),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig_t, use_container_width=True)

    # ────────────────────────────────────
    # TAB 2: 키워드 & VOC 패턴
    # ────────────────────────────────────
    with tab_keyword:
        section_title("이달의 키워드 TOP 30", "🔑")
        kws = extract_keywords(df_m, 30)
        if kws:
            kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
            fig_kw = go.Figure(go.Bar(
                x=kdf["빈도"], y=kdf["키워드"],
                orientation="h",
                marker=dict(
                    color=kdf["빈도"],
                    colorscale=[[0,"rgba(99,102,241,0.25)"],[1,"#6366f1"]],
                    showscale=False,
                    line=dict(color="white", width=0.5),
                ),
                text=kdf["빈도"], textposition="outside", textfont=dict(size=11),
            ))
            fig_kw.update_layout(
                height=680,
                margin=dict(l=10,r=40,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                yaxis=dict(showgrid=False, autorange="reversed"),
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.caption("키워드 없음 (주관식 컬럼 확인)")

        # 부정 응답 키워드 (별도)
        section_title("부정 응답 전용 키워드 분석", "🚨")
        if "긍정부정" in df_m.columns:
            neg_df = df_m[df_m["긍정부정"] == "부정"]
            if not neg_df.empty:
                neg_kws = extract_keywords(neg_df, 20)
                if neg_kws:
                    nkdf = pd.DataFrame(neg_kws, columns=["키워드","빈도"])
                    fig_nk = go.Figure(go.Bar(
                        x=nkdf["빈도"], y=nkdf["키워드"],
                        orientation="h",
                        marker=dict(
                            color=nkdf["빈도"],
                            colorscale=[[0,"rgba(239,68,68,0.2)"],[1,"#ef4444"]],
                            showscale=False,
                            line=dict(color="white", width=0.5),
                        ),
                        text=nkdf["빈도"], textposition="outside", textfont=dict(size=11),
                    ))
                    fig_nk.update_layout(
                        height=460,
                        margin=dict(l=10,r=40,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=12),
                        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                    )
                    st.plotly_chart(fig_nk, use_container_width=True)
            else:
                st.success("이번 달 부정 응답 없음")

    # ────────────────────────────────────
    # TAB 3: 개선 사례 (부정 + 저점수)
    # ────────────────────────────────────
    with tab_case:
        section_title("교육 활용 개선 사례 (부정 응답 + 저점수)", "📖")
        st.markdown("""
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);
                    border-radius:8px;padding:10px 16px;font-size:12px;color:#0f172a;margin-bottom:12px;">
            💡 <b>활용법</b>: 아래 사례를 교육 자료로 활용하세요.
            어떤 상황에서 고객 불만이 발생하는지 패턴을 파악하고,
            롤플레이 및 사례 학습에 사용하세요.
        </div>
        """, unsafe_allow_html=True)

        if "긍정부정" in df_m.columns:
            neg_cases = df_m[df_m["긍정부정"] == "부정"].copy()
            if "최종점수" in neg_cases.columns:
                neg_cases = neg_cases.sort_values("최종점수", ascending=True)

            if neg_cases.empty:
                st.success("이번 달 부정 응답 없음 — 우수한 달입니다!")
            else:
                # 유형별 부정 분포
                for gcol in ["상담유형대","채널_구분","브랜드"]:
                    if gcol in neg_cases.columns:
                        dist = neg_cases[gcol].value_counts().reset_index()
                        dist.columns = [gcol, "부정건수"]
                        col_d1, col_d2 = st.columns([1, 2])
                        with col_d1:
                            st.markdown(f"**{gcol}별 부정 분포**")
                            st.dataframe(dist, use_container_width=True, hide_index=True)
                        with col_d2:
                            fig_d = go.Figure(go.Bar(
                                x=dist["부정건수"], y=dist[gcol],
                                orientation="h",
                                marker=dict(color="#ef4444", opacity=0.8,
                                            line=dict(color="white", width=0.5)),
                                text=dist["부정건수"], textposition="outside",
                            ))
                            fig_d.update_layout(
                                height=max(180, len(dist)*38+40),
                                margin=dict(l=10,r=40,t=4,b=4),
                                plot_bgcolor="white", paper_bgcolor="white",
                                font=dict(family="Inter, Noto Sans KR", size=11),
                                xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                                yaxis=dict(showgrid=False, autorange="reversed"),
                            )
                            st.plotly_chart(fig_d, use_container_width=True)
                        break  # 첫 번째 유형만 표시 후 break

                # 사례 목록 (교육용 테이블)
                st.markdown(f"**📋 개선 사례 전체 목록 ({len(neg_cases)}건)**")
                preferred = ["회신일","상담사","브랜드","채널_구분","상담유형대",
                             "친절점수","만족점수","최종점수","주관식","긍정부정"]
                cols_disp = get_display_cols(neg_cases, preferred)
                st.dataframe(neg_cases[cols_disp].reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        # 저점수(70~89) 사례
        if "최종점수" in df_m.columns:
            mid_cases = df_m[
                (df_m["최종점수"] >= SCORE_CAUTION) &
                (df_m["최종점수"] < SCORE_GOOD)
            ].copy()
            if not mid_cases.empty:
                section_title(f"관찰 구간(70~89점) 사례 ({len(mid_cases)}건)", "🟡")
                preferred2 = ["회신일","상담사","브랜드","채널_구분",
                              "친절점수","만족점수","최종점수","주관식","긍정부정"]
                cols_disp2 = get_display_cols(mid_cases, preferred2)
                st.dataframe(
                    mid_cases.sort_values("최종점수")[cols_disp2].reset_index(drop=True),
                    use_container_width=True, hide_index=True
                )

    # ────────────────────────────────────
    # TAB 4: 우수 사례 (Best Practice)
    # ────────────────────────────────────
    with tab_best:
        section_title("교육 활용 우수 사례 (긍정 + 고점수)", "🌟")
        st.markdown("""
        <div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);
                    border-radius:8px;padding:10px 16px;font-size:12px;color:#0f172a;margin-bottom:12px;">
            💡 <b>활용법</b>: 우수 사례를 공유하고 칭찬 문화를 만들어보세요.
            어떤 응대 방식이 고객 만족을 이끄는지 분석하고,
            이를 교육 표준 스크립트 개발에 활용하세요.
        </div>
        """, unsafe_allow_html=True)

        if "긍정부정" in df_m.columns and "최종점수" in df_m.columns:
            best_cases = df_m[
                (df_m["긍정부정"] == "긍정") &
                (df_m["최종점수"] >= SCORE_GOOD)
            ].copy().sort_values("최종점수", ascending=False)

            if best_cases.empty:
                st.info("이번 달 우수 사례 데이터가 없습니다.")
            else:
                # 우수 상담사 TOP
                section_title(f"우수 상담사 TOP (긍정+90점↑ 기준, {len(best_cases)}건)", "🏆")
                if "상담사" in best_cases.columns:
                    best_ag = (best_cases.groupby("상담사")
                                .agg(우수건수=("최종점수","count"),
                                     평균점수=("최종점수","mean"))
                                .round(1).sort_values("우수건수", ascending=False)
                                .reset_index())
                    col_b1, col_b2 = st.columns([1, 2])
                    with col_b1:
                        st.dataframe(best_ag, use_container_width=True, hide_index=True)
                    with col_b2:
                        fig_best = go.Figure(go.Bar(
                            x=best_ag["우수건수"],
                            y=best_ag["상담사"],
                            orientation="h",
                            marker=dict(
                                color=best_ag["평균점수"],
                                colorscale=[[0,"rgba(34,197,94,0.3)"],[1,"#22c55e"]],
                                cmin=90, cmax=100, showscale=False,
                                line=dict(color="white", width=0.5),
                            ),
                            text=best_ag["우수건수"].astype(str) + "건",
                            textposition="outside", textfont=dict(size=11),
                        ))
                        fig_best.update_layout(
                            height=max(250, len(best_ag)*34+40),
                            margin=dict(l=10,r=60,t=10,b=10),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=12),
                            xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                            yaxis=dict(showgrid=False, autorange="reversed"),
                        )
                        st.plotly_chart(fig_best, use_container_width=True)

                # 우수 사례 원문
                section_title("우수 VOC 원문 (교육 활용)", "💬")
                voc_col = "주관식" if "주관식" in best_cases.columns else None
                if voc_col:
                    best_voc = best_cases[best_cases[voc_col].notna() &
                                          (best_cases[voc_col].astype(str).str.strip() != "")]
                    preferred_b = ["회신일","상담사","브랜드","채널_구분",
                                   "친절점수","만족점수","최종점수","주관식"]
                    cols_b = get_display_cols(best_voc, preferred_b)
                    st.dataframe(best_voc[cols_b].reset_index(drop=True),
                                 use_container_width=True, hide_index=True)
                else:
                    st.caption("주관식 컬럼 없음")

        # 우수 키워드
        if "긍정부정" in df_m.columns:
            pos_df_e = df_m[df_m["긍정부정"] == "긍정"]
            if not pos_df_e.empty:
                section_title("긍정 응답 키워드 (표준 스크립트 참고)", "✨")
                pos_kws = extract_keywords(pos_df_e, 20)
                if pos_kws:
                    pkdf = pd.DataFrame(pos_kws, columns=["키워드","빈도"])
                    fig_pk = go.Figure(go.Bar(
                        x=pkdf["빈도"], y=pkdf["키워드"],
                        orientation="h",
                        marker=dict(
                            color=pkdf["빈도"],
                            colorscale=[[0,"rgba(34,197,94,0.25)"],[1,"#22c55e"]],
                            showscale=False,
                            line=dict(color="white", width=0.5),
                        ),
                        text=pkdf["빈도"], textposition="outside", textfont=dict(size=11),
                    ))
                    fig_pk.update_layout(
                        height=460,
                        margin=dict(l=10,r=40,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=12),
                        xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        yaxis=dict(showgrid=False, autorange="reversed"),
                    )
                    st.plotly_chart(fig_pk, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 15. Streamlit App 메인 (개선)
# ══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="CSAT Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ── 세션 상태 초기화 (특수문자 없는 key 사용) ──
    if "menu" not in st.session_state:
        st.session_state["menu"] = "개요"
    # 구버전 key 잔존 방어: 유효하지 않은 key면 개요로 리셋
    valid_keys = {item["key"] for item in MENU_ITEMS}
    if st.session_state["menu"] not in valid_keys:
        st.session_state["menu"] = "개요"

    # ── 데이터 로드 ──
    try:
        df_raw = load_from_gsheets()
    except Exception as e:
        st.error("구글시트 연결 실패: 공개 설정(링크 공개)과 Secrets를 확인하세요.")
        st.exception(e)
        return

    # 데이터 준비 (캐시)
    @st.cache_data(ttl=600, show_spinner="데이터 처리 중...")
    def _prepare(raw_json: str):
        import json
        df_r = pd.read_json(raw_json, orient="split")
        return prepare_data_from_df(df_r)

    try:
        raw_json = df_raw.to_json(orient="split", date_format="iso")
        (df_all, df_active, df_scored, df_scored_all,
         available_months, retired_agents) = _prepare(raw_json)
    except Exception as e:
        st.error(f"데이터 처리 오류: {e}")
        st.exception(e)
        return

    # ── 사이드바 (기간선택 상단 포함) ──
    selected_menu, target_month, selected_date, selected_week = render_sidebar_nav(
        st.session_state["menu"],
        available_months=available_months,
        df_scored_all=df_scored_all,
        df_all=df_all,
    )
    st.session_state["menu"] = selected_menu

    # ── 월 기준 df_m ──
    df_m     = fm(df_scored,   target_month)
    df_m_all = fm_sent(df_all, target_month)

    # ── 페이지 라우팅 ──
    menu = st.session_state["menu"]

    try:
        if menu == "개요":
            page_overview(df_all, df_scored, df_scored_all, available_months,
                          target_month, selected_date, selected_week)

        elif menu == "일자주차":
            page_day_week(df_all, df_scored, df_scored_all, selected_date, selected_week)

        elif menu == "점수분석":
            page_scores(df_m)

        elif menu == "주관식분석":
            page_verbatim(df_m)

        elif menu == "히트맵":
            page_integrated(df_m)

        elif menu == "Action필요":
            page_action(df_m, df_m_all)

        elif menu == "상담사성과":
            page_daily_agent(df_m)

        elif menu == "70점미만":
            page_low_scores(df_scored_all, target_month=target_month)

        elif menu == "검색":
            page_search(df_scored_all, df_all)

        elif menu == "인사이트":
            page_insight(df_all, df_scored, df_scored_all, available_months, target_month)

        elif menu == "교육자료":
            page_education(df_m, df_scored_all, target_month)

        else:
            st.session_state["menu"] = "개요"
            st.rerun()

    except Exception as e:
        st.error(f"페이지 렌더링 오류 [{menu}]: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()
