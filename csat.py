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
# 0. 전역 상수 & 색상 팔레트
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

# QC 관련 상수
COMPLIANCE_COLS = {
    "acc_guidance":       10,
    "acc_process":        10,
    "acc_system":         10,
    "prof_tailored":      10,
    "prof_query":          5,
    "prof_voice_wait":     5,
    "kind_emotion":       10,
    "kind_listening":     15,
    "kind_language":       5,
    "promise_nonfulfill": 10,
    "promise_delay":       5,
}

COMPLIANCE_COL_LABELS = {
    "acc_guidance":       "정확한안내",
    "acc_process":        "프로세스",
    "acc_system":         "전산처리",
    "prof_tailored":      "맞춤설명",
    "prof_query":         "문의파악",
    "prof_voice_wait":    "음성숙련도/대기",
    "kind_emotion":       "감정연출/양해",
    "kind_listening":     "경청/즉각호응",
    "kind_language":      "언어표현",
    "promise_nonfulfill": "약속불이행",
    "promise_delay":      "약속지연이행",
}

ACCOUNTABILITY_COLORS = {
    "IBR":   "#3b82f6",
    "상담사": "#ef4444",
    "고객":   "#22c55e",
}


# ══════════════════════════════════════════════════════════════════
# 1. 날짜 파싱
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
# 2. 감성·의도 분류
# ══════════════════════════════════════════════════════════════════
_HIGH_PREC_NEG_PATTERNS = [
    r"(?:잠깐만|잠시만|조금만).{0,10}(?:요|요\.)",
    r"(?:또|다시|계속|반복).{0,10}(?:기다려|대기)",
    r"(?:처음|아까|어제|전날|이전|저번).{0,15}(?:다르|달라|바꿔|변경)",
    r"(?:말|설명|안내).{0,10}(?:달라|다르|바뀌|바꿔|모순|틀려)",
    r"(?:어제|전날|이미|아까).{0,10}(?:합의|협의|약속|확인).{0,15}(?:또|다시|추가)",
    r"(?:이미|전에).{0,10}(?:보냈|제출|드렸).{0,15}(?:또|다시|재|추가)",
    r"\d+\s*번.{0,10}(?:전화|문의|연락|상담)",
    r"(?:\d+|한|두|세|이|삼|사|오).{0,3}(?:주|달|개월).{0,5}(?:넘|지나|됐)",
    r"(?:같은|똑같은|동일한).{0,10}말만",
    r"(?:대응|처리|연락|답변|해결|교환|환불|배송|발송).{0,8}(?:안돼|안되|못해|안됩|불가|지연)",
    r"(?:너무|정말|진짜).{0,5}(?:늦어|느려|오래|답답|아쉽|실망|속상|허탈)",
    r"환불\s*(?:거부|안|못|지연)", r"취소\s*(?:거부|안|못)",
]
_NEGATION_PATTERNS = [
    r"불편함?\s*없", r"불만\s*없", r"문제\s*없", r"걱정\s*없",
    r"어렵지\s*않", r"나쁘지\s*않", r"부족하지\s*않",
    r"아쉽지\s*않", r"실망하지\s*않", r"불편하지\s*않",
    r"늦지\s*않", r"느리지\s*않",
    r"(?:전혀|하나도)\s*(?:불편|불만|문제)",
    r"전혀\s*없", r"하나도\s*없",
]
_RULE_POS_WORDS = [
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
_RULE_NEG_WORDS = [
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
    "절대추천안해","비추천","조금만기다려","말이달라","말을바꿔","앞에서는",
    "전에는된다고","다르게말했","모순됐","일관성없","혼란스럽","헷갈려","혼동됐",
    "설명이다달라","매번달라","어제협의했는데","이미교환하기로","이미환불하기로",
    "다시증빙","또증빙","재제출","전체반품","전부보내","불필요하게","과도한요구",
    "똑같은말","같은말만","동일한답변","똑같은답변","매번같은",
    "안돼요","안되요","안됩니다","불가하다","불가능하다","안된다고",
    "대응이안","처리가안","해결이안","연결이안",
    "너무늦어","너무느려","늦어요","느려요","언제까지","기다려야",
    "오래걸려","오래됐","한달넘","두달넘","일주일넘","이주넘","삼주넘",
    "두번째전화","세번째전화","또전화했","또연락했","재문의했","다시전화",
    "아쉽네요","아쉽습니다","실망이에요","실망입니다","아닌거같아요",
    "이건아닌","이러면안","이러시면","속상해요","속상합니다","허탈해요",
    "답답해요","답답합니다","황당해요","어이없어요",
    "택배가안","배송이안","아직도안","언제오는","안오네요",
]


def _rule_classify(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return None
    t = re.sub(r"\s+", "", text.strip())
    for pat in _HIGH_PREC_NEG_PATTERNS:
        if re.search(pat, text.strip()):
            return "부정"
    negated   = any(re.search(p, t) for p in _NEGATION_PATTERNS)
    pos_count = sum(1 for w in _RULE_POS_WORDS if w in t)
    neg_count = 0 if negated else sum(1 for w in _RULE_NEG_WORDS if w in t)
    if neg_count > 0 and not negated:
        return "부정" if neg_count * 1 >= pos_count * 0.6 else "긍정"
    if neg_count == 0 and pos_count == 0:
        short_pos = ["감사합니다","고맙습니다","잘됐어요","잘됐습니다","해결됐어요",
                     "완료됐습니다","수고하셨습니다","잘부탁드립니다"]
        return "긍정" if any(w in t for w in short_pos) else "중립"
    return "부정" if neg_count > pos_count else "긍정"


INTENT_LABELS = [
    "상담사평가","프로세스개선","정책분쟁","지연대기","상품배송주문","시스템오류","기타",
]

_INTENT_SEEDS = {
    "상담사평가": ["친절","불친절","상담사","직원","응대","태도","말투","무례","전문성","공감","설명","안내","퉁명","냉정","친근","따뜻","칭찬","고압적"],
    "프로세스개선": ["개선","자동화","절차","프로세스","시스템","불편","간편","쉽게","복잡","콜백","자동","UI","앱","화면","메뉴","기능","개편","바꿔"],
    "정책분쟁": ["환불","취소","반품","정책","규정","불공평","거부","납득","이해안","말이달라","약속","합의","협의","보상","처리기준"],
    "지연대기": ["지연","늦","오래","기다","대기","언제","몇번","또전화","반복","재문의","계속","응답없","연락없","처리안","미해결"],
    "상품배송주문": ["배송","택배","주문","상품","오배송","누락","누빠","파손","불량","결제","결제오류","주문취소","교환","반품","미입고","미배달"],
    "시스템오류": ["오류","에러","앱","앱오류","사이트","접속","로그인","결제오류","화면","버그","먹통","안됩니다","시스템","장애"],
    "기타": ["기타","상관없","해당없","모르겠","잘모르","없음"],
}

_SENTIMENT_MODEL = None
_INTENT_MODEL    = None
_SENTIMENT_VEC   = None
_INTENT_VEC      = None
_MODEL_TRAINED   = False


def _tokenize_korean(text: str) -> str:
    t = re.sub(r"[^\uAC00-\uD7A3\s]", " ", str(text))
    tokens = [w for w in re.findall(r"[\uAC00-\uD7A3]{2,}", t)]
    return " ".join(tokens)


def _make_weak_labels(texts, scores):
    labels = []
    for text, score in zip(texts, scores):
        rule = _rule_classify(text)
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = None
        if rule == "부정":
            labels.append("부정")
        elif s is not None and s < 70:
            labels.append("부정")
        elif s is not None and s >= 90 and rule == "긍정":
            labels.append("긍정")
        elif rule == "긍정" and s is not None and s >= 70:
            labels.append("긍정")
        elif rule == "중립" or rule is None:
            if s is not None and s >= 90:
                labels.append("긍정")
            elif s is not None and s < 70:
                labels.append("부정")
            else:
                labels.append("중립")
        else:
            labels.append(rule)
    return labels


def _make_intent_weak_labels(texts):
    labels = []
    for text in texts:
        t = re.sub(r"\s+", "", str(text))
        matched = []
        for intent, kws in _INTENT_SEEDS.items():
            if any(kw in t for kw in kws):
                matched.append(intent)
        if not matched:
            matched = ["기타"]
        labels.append(matched)
    return labels


def _train_models(df: pd.DataFrame):
    global _SENTIMENT_MODEL, _INTENT_MODEL, _SENTIMENT_VEC, _INTENT_VEC, _MODEL_TRAINED
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        return False
    voc_col   = next((c for c in ["주관식","verbatim","Q3","의견"] if c in df.columns), None)
    score_col = "최종점수" if "최종점수" in df.columns else None
    if not voc_col:
        return False
    sub = df[df[voc_col].notna()].copy()
    sub = sub[sub[voc_col].astype(str).str.strip() != ""].copy()
    if len(sub) < 20:
        return False
    texts  = sub[voc_col].astype(str).tolist()
    scores = sub[score_col].tolist() if score_col else [None] * len(texts)
    tokens = [_tokenize_korean(t) for t in texts]
    sent_labels = _make_weak_labels(texts, scores)
    _SENTIMENT_VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=8000, sublinear_tf=True)
    X_sent = _SENTIMENT_VEC.fit_transform(tokens)
    base_clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    _SENTIMENT_MODEL = CalibratedClassifierCV(base_clf, cv=min(3, len(set(sent_labels))))
    _SENTIMENT_MODEL.fit(X_sent, sent_labels)
    intent_labels = _make_intent_weak_labels(texts)
    mlb = MultiLabelBinarizer(classes=INTENT_LABELS)
    Y_intent = mlb.fit_transform(intent_labels)
    _INTENT_VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=8000, sublinear_tf=True)
    X_intent = _INTENT_VEC.fit_transform(tokens)
    _INTENT_MODEL = OneVsRestClassifier(LogisticRegression(max_iter=300, C=0.8, class_weight="balanced"))
    _INTENT_MODEL.fit(X_intent, Y_intent)
    _INTENT_MODEL._mlb = mlb
    _MODEL_TRAINED = True
    return True


def _ml_classify_batch(texts: list, scores: list = None) -> list:
    if not _MODEL_TRAINED or _SENTIMENT_MODEL is None:
        results = []
        for i, t in enumerate(texts):
            s = scores[i] if scores else None
            rule = _rule_classify(t)
            if rule is None:
                results.append({"감성": None, "신뢰도": None, "needs_review": False})
                continue
            try:
                sv = float(s) if s is not None else None
            except (TypeError, ValueError):
                sv = None
            if sv is not None and sv < 70 and rule != "부정":
                rule = "부정"
            elif sv is not None and 70 <= sv < 90 and rule == "긍정":
                rule = "중립"
            results.append({"감성": rule, "신뢰도": 0.6, "needs_review": False})
        return results
    tokens = [_tokenize_korean(t) for t in texts]
    X = _SENTIMENT_VEC.transform(tokens)
    proba = _SENTIMENT_MODEL.predict_proba(X)
    classes = _SENTIMENT_MODEL.classes_
    results = []
    for i, (t, prob_row) in enumerate(zip(texts, proba)):
        best_idx = int(np.argmax(prob_row))
        pred     = classes[best_idx]
        conf     = float(prob_row[best_idx])
        for pat in _HIGH_PREC_NEG_PATTERNS:
            if re.search(pat, str(t)):
                pred = "부정"
                conf = max(conf, 0.85)
                break
        sv = None
        if scores:
            try:
                sv = float(scores[i])
            except (TypeError, ValueError):
                sv = None
        if sv is not None and sv < 70 and pred != "부정":
            pred = "부정"
            conf = max(conf, 0.75)
        elif sv is not None and 70 <= sv < 90 and pred == "긍정":
            pred = "중립"
            conf = max(conf, 0.65)
        needs_review = (conf < 0.55) or (pred == "긍정" and sv is not None and sv < 70)
        results.append({"감성": pred, "신뢰도": round(conf, 3), "needs_review": needs_review})
    return results


def _ml_intent_batch(texts: list) -> list:
    if not _MODEL_TRAINED or _INTENT_MODEL is None:
        results = []
        for text in texts:
            weak = _make_intent_weak_labels([text])[0]
            results.append({"의도": weak, "의도_신뢰도": {k: 0.5 for k in weak}})
        return results
    tokens = [_tokenize_korean(t) for t in texts]
    X      = _INTENT_VEC.transform(tokens)
    proba  = _INTENT_MODEL.predict_proba(X)
    mlb    = _INTENT_MODEL._mlb
    results = []
    for i, prob_row in enumerate(proba):
        intent_conf = {label: round(float(p), 3) for label, p in zip(mlb.classes_, prob_row)}
        matched = [lbl for lbl, p in intent_conf.items() if p >= 0.35]
        if not matched:
            matched = [max(intent_conf, key=intent_conf.get)]
        results.append({"의도": matched, "의도_신뢰도": intent_conf})
    return results


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_classify(texts_tuple, scores_tuple):
    texts  = list(texts_tuple)
    scores = list(scores_tuple) if scores_tuple else [None] * len(texts)
    sent   = _ml_classify_batch(texts, scores)
    intent = _ml_intent_batch(texts)
    return sent, intent


def classify_sentiment(text: str) -> str:
    return _rule_classify(text)


def classify_with_score(text: str, score) -> str:
    base = _rule_classify(text)
    if base is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return base
    if base == "부정":
        return "부정"
    if s < 70:
        return "부정"
    if s < 90 and base == "긍정":
        return "중립"
    return base


def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    col       = next((c for c in ["주관식","verbatim","Q3","의견"] if c in df.columns), None)
    score_col = "최종점수" if "최종점수" in df.columns else None
    if not col:
        df["긍정부정"] = None
        return df
    mask       = df[col].notna() & (df[col].astype(str).str.strip() != "")
    texts_all  = df[col].astype(str).tolist()
    scores_all = df[score_col].tolist() if score_col else [None] * len(df)
    texts_valid  = [texts_all[i]  for i in range(len(df)) if mask.iloc[i]]
    scores_valid = [scores_all[i] for i in range(len(df)) if mask.iloc[i]]
    if texts_valid:
        try:
            sent_res, intent_res = _cached_classify(
                tuple(texts_valid), tuple(str(s) for s in scores_valid)
            )
        except Exception:
            sent_res   = [{"감성": _rule_classify(t), "신뢰도": 0.6, "needs_review": False} for t in texts_valid]
            intent_res = _make_intent_weak_labels(texts_valid)
            intent_res = [{"의도": r, "의도_신뢰도": {k: 0.5 for k in r}} for r in intent_res]
    else:
        sent_res   = []
        intent_res = []
    sent_map   = {}
    intent_map = {}
    valid_idx  = [i for i in range(len(df)) if mask.iloc[i]]
    for j, orig_i in enumerate(valid_idx):
        sent_map[orig_i]   = sent_res[j]
        intent_map[orig_i] = intent_res[j]
    def get_sent(i):
        r = sent_map.get(i, {})
        return r.get("감성", None)
    def get_conf(i):
        r = sent_map.get(i, {})
        return r.get("신뢰도", None)
    def get_review(i):
        r = sent_map.get(i, {})
        return r.get("needs_review", False)
    def get_intent_str(i):
        r = intent_map.get(i, {})
        return ", ".join(r.get("의도", []))
    df = df.copy()
    df["긍정부정"]    = [get_sent(i) for i in range(len(df))]
    df["감성(모델)"]  = df["긍정부정"]
    df["분류신뢰도"]  = [get_conf(i)   for i in range(len(df))]
    df["검토필요"]    = [get_review(i) for i in range(len(df))]
    df["의도분류"]    = [get_intent_str(i) for i in range(len(df))]
    return df


def train_classifiers_from_df(df: pd.DataFrame) -> bool:
    return _train_models(df)


MANUAL_LABEL_SHEET = "수동부정라벨"

def load_manual_neg_labels() -> set:
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_label = conn.read(worksheet=MANUAL_LABEL_SHEET, ttl=60)
        if df_label is None or df_label.empty:
            return set()
        if "상담KEY" in df_label.columns:
            return set(df_label["상담KEY"].dropna().astype(str).tolist())
        return set(df_label.iloc[:, 0].dropna().astype(str).tolist())
    except Exception:
        return set()


def save_manual_neg_labels(keys: list):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing = load_manual_neg_labels()
        all_keys = sorted(existing | set(str(k) for k in keys))
        df_save = pd.DataFrame({"상담KEY": all_keys,
                                "라벨": ["부정"] * len(all_keys),
                                "등록일시": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")] * len(all_keys)})
        conn.update(worksheet=MANUAL_LABEL_SHEET, data=df_save)
        return True
    except Exception as e:
        st.error(f"저장 실패: {e}")
        return False


def apply_manual_neg_labels(df: pd.DataFrame, manual_keys: set) -> pd.DataFrame:
    if not manual_keys or df.empty:
        return df
    df = df.copy()
    if "상담KEY" in df.columns:
        mask = df["상담KEY"].astype(str).isin(manual_keys)
    else:
        mask = df.index.astype(str).isin(manual_keys)
    df.loc[mask, "긍정부정"] = "부정"
    return df


# ══════════════════════════════════════════════════════════════════
# 3. 컬럼 매핑
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
# 4. 데이터 정제
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
# 5. 분석 함수
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
# 6. 주차/일자 필터
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
# 7. RAW 테이블 컬럼 정렬
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
# 8. Google Sheets 로드
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def load_from_gsheets() -> pd.DataFrame:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df   = conn.read(ttl="10m")
    df   = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return df


def prepare_data_from_df(df_raw: pd.DataFrame):
    df = normalize_columns(df_raw)
    df = build_time_columns(df)
    result = split_active_and_scored(df)
    global _MODEL_TRAINED
    if not _MODEL_TRAINED:
        try:
            train_classifiers_from_df(result[0])
        except Exception:
            pass
    return result


# ══════════════════════════════════════════════════════════════════
# ★ QC 모니터링 — 데이터 로드 & 파싱
#   (2행 헤더 처리 포함)
# ══════════════════════════════════════════════════════════════════

# QC 시트 컬럼 매핑: 원본 한글(부분) → 영문 key
QC_COL_MAP_PARTIAL = {
    "귀책분류":     "accountability",
    "문의불만사유": "complaint_reason",
    "상담이력KEY":  "consult_key",
    "상담이력key":  "consult_key",
    "이행점수":     "compliance_score",
    "이행\n점수":   "compliance_score",
    "상세분석":     "detail_analysis",
    "피드백여부":   "feedback_given",
    "회신월":       "reply_month",
    "상담사":       "qc_agent",
    "채널":         "qc_channel",
    "브랜드":       "qc_brand",
    # compliance 항목 (부분 매칭용)
    "정확한안내":   "acc_guidance",
    "프로세스":     "acc_process",
    "전산처리":     "acc_system",
    "맞춤설명":     "prof_tailored",
    "문의파악":     "prof_query",
    "음성숙련도":   "prof_voice_wait",
    "대기":         "prof_voice_wait",
    "감정연출":     "kind_emotion",
    "양해":         "kind_emotion",
    "경청":         "kind_listening",
    "즉각호응":     "kind_listening",
    "언어표현":     "kind_language",
    "약속불이행":   "promise_nonfulfill",
    "약속 불이행":  "promise_nonfulfill",
    "약속지연이행": "promise_delay",
    "약속 지연이행":"promise_delay",
}


def _normalize_qc_header(raw: str) -> str:
    """
    원본 헤더(다중 줄, 공백, 괄호 포함)를 영문 key로 변환.
    예: "정확한안내\\n(10)" → "acc_guidance"
    """
    clean = re.sub(r"[\n\r\s\(\)\d]+", "", str(raw))  # 공백·줄바꿈·괄호·숫자 제거
    # 완전 일치 우선
    for k, v in QC_COL_MAP_PARTIAL.items():
        k_clean = re.sub(r"[\n\r\s\(\)\d]+", "", k)
        if clean == k_clean:
            return v
    # 부분 포함
    for k, v in QC_COL_MAP_PARTIAL.items():
        k_clean = re.sub(r"[\n\r\s\(\)\d]+", "", k)
        if k_clean and k_clean in clean:
            return v
    return raw  # 매핑 실패 시 원본 반환


def _read_qc_sheet_with_2row_header(conn) -> pd.DataFrame:
    """
    2행 헤더 처리:
    - 1행: 카테고리 (정확성, 숙련도, 친절도, 약속이행)
    - 2행: 세부 항목명 + 배점
    → 두 행을 합쳐서 최종 헤더 생성 후 데이터 로드
    """
    sheet_names = ["QC모니터링", "Sheet2", "QC", "qc"]

    for sheet_name in sheet_names:
        try:
            # 헤더 없이 raw 로드
            try:
                df_raw = conn.read(worksheet=sheet_name, ttl="10m", header=None)
            except Exception:
                df_raw = conn.read(worksheet=sheet_name, ttl="10m")
                # 헤더 처리 없이 raw 리턴
                return df_raw, sheet_name

            if df_raw is None or df_raw.empty:
                continue

            df_raw = df_raw.reset_index(drop=True)

            # 2행 헤더 판별:
            # 첫 번째 행이 카테고리명(정확성/숙련도/친절도/약속이행)을 포함하면 2행 헤더
            first_row_vals = [str(v) for v in df_raw.iloc[0].tolist()]
            is_2row = any(kw in " ".join(first_row_vals) for kw in ["정확성","숙련도","친절도","약속이행"])

            if is_2row and len(df_raw) >= 2:
                row1 = [str(v) if pd.notna(v) and str(v) != "nan" else "" for v in df_raw.iloc[0]]
                row2 = [str(v) if pd.notna(v) and str(v) != "nan" else "" for v in df_raw.iloc[1]]

                # 두 행 합쳐서 헤더 생성 (row2가 주, row1은 보조)
                merged = []
                for r1, r2 in zip(row1, row2):
                    if r2.strip():
                        merged.append(r2.strip())
                    elif r1.strip():
                        merged.append(r1.strip())
                    else:
                        merged.append("_unnamed")

                df_data = df_raw.iloc[2:].copy().reset_index(drop=True)
                df_data.columns = merged
            else:
                # 1행 헤더
                df_data = df_raw.iloc[1:].copy().reset_index(drop=True)
                df_data.columns = [str(v) for v in df_raw.iloc[0]]

            return df_data, sheet_name

        except Exception:
            continue

    # 인덱스 기반 폴백
    try:
        df_raw = conn.read(worksheet=1, ttl="10m", header=None)
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(), "index_1"
        df_raw = df_raw.reset_index(drop=True)
        first_row_vals = [str(v) for v in df_raw.iloc[0].tolist()]
        is_2row = any(kw in " ".join(first_row_vals) for kw in ["정확성","숙련도","친절도","약속이행"])
        if is_2row and len(df_raw) >= 2:
            row1 = [str(v) if pd.notna(v) and str(v) != "nan" else "" for v in df_raw.iloc[0]]
            row2 = [str(v) if pd.notna(v) and str(v) != "nan" else "" for v in df_raw.iloc[1]]
            merged = []
            for r1, r2 in zip(row1, row2):
                if r2.strip():
                    merged.append(r2.strip())
                elif r1.strip():
                    merged.append(r1.strip())
                else:
                    merged.append("_unnamed")
            df_data = df_raw.iloc[2:].copy().reset_index(drop=True)
            df_data.columns = merged
        else:
            df_data = df_raw.iloc[1:].copy().reset_index(drop=True)
            df_data.columns = [str(v) for v in df_raw.iloc[0]]
        return df_data, "index_1"
    except Exception as e:
        return pd.DataFrame(), f"error: {e}"


@st.cache_data(ttl=600)
def load_qc_data() -> pd.DataFrame:
    """QC 모니터링 시트 로드 (2행 헤더 자동 처리)"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df, sheet_name = _read_qc_sheet_with_2row_header(conn)
        if df is None or df.empty:
            return pd.DataFrame()

        # 전체 비어있는 행/열 제거
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

        # 컬럼명 정규화 → 영문 key
        new_cols = []
        used = {}
        for col in df.columns:
            mapped = _normalize_qc_header(str(col))
            # 중복 방지
            if mapped in used:
                used[mapped] += 1
                mapped = f"{mapped}_{used[mapped]}"
            else:
                used[mapped] = 0
            new_cols.append(mapped)
        df.columns = new_cols

        # month_key 생성
        if "reply_month" in df.columns:
            df["month_key"] = extract_ym(df["reply_month"])
        
        # compliance_score 숫자 변환
        if "compliance_score" in df.columns:
            df["compliance_score"] = pd.to_numeric(df["compliance_score"], errors="coerce")

        # 미흡 파싱
        df = parse_deficiencies(df)

        return df

    except Exception as e:
        st.warning(f"QC 데이터 로드 실패: {e}")
        return pd.DataFrame()


def is_deficiency(val) -> bool:
    """셀 값이 미흡 항목(문자열 존재)이면 True"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip()
    return s not in ("", "nan", "NaN", "None", "NaT", "-", "O", "o")


def parse_deficiencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 compliance 컬럼에 대해:
    - _flag 컬럼 생성 (1=미흡, 0=정상)
    - 카테고리별 점수 계산
    - deficiency_list 컬럼 생성
    """
    df = df.copy()
    present_cols = {k: v for k, v in COMPLIANCE_COLS.items() if k in df.columns}

    # flag 컬럼
    for col in present_cols:
        df[f"{col}_flag"] = df[col].apply(lambda x: 1 if is_deficiency(x) else 0)

    # 카테고리별 점수
    acc_cols   = [c for c in ["acc_guidance","acc_process","acc_system"] if c in present_cols]
    prof_cols  = [c for c in ["prof_tailored","prof_query","prof_voice_wait"] if c in present_cols]
    kind_cols  = [c for c in ["kind_emotion","kind_listening","kind_language"] if c in present_cols]
    prom_cols  = [c for c in ["promise_nonfulfill","promise_delay"] if c in present_cols]

    def cat_score(row, cols, max_pts):
        deducted = sum(present_cols[c] for c in cols if row.get(f"{c}_flag", 0) == 1)
        return max(0, max_pts - deducted)

    df["accuracy_score"]    = df.apply(lambda r: cat_score(r, acc_cols,  30), axis=1)
    df["proficiency_score"] = df.apply(lambda r: cat_score(r, prof_cols, 20), axis=1)
    df["kindness_score"]    = df.apply(lambda r: cat_score(r, kind_cols, 30), axis=1)
    df["promise_score"]     = df.apply(lambda r: cat_score(r, prom_cols, 20), axis=1)

    # 미흡 항목 리스트
    def get_deficiency_list(row):
        items = []
        for col in present_cols:
            if row.get(f"{col}_flag", 0) == 1:
                label = COMPLIANCE_COL_LABELS.get(col, col)
                val   = str(row.get(col, "")).strip()
                items.append(f"{label}({val})" if val and val not in ("1","nan") else label)
        return items

    df["deficiency_list"] = df.apply(get_deficiency_list, axis=1)
    df["deficiency_count"] = df["deficiency_list"].apply(len)

    return df


# ══════════════════════════════════════════════════════════════════
# 9. 채널별 월별 추이
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
# 10. 3주 트렌드 데이터 생성
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
        f"W-2 ({w_prev2})"             if w_prev2 else "W-2",
        f"W-1 ({w_prev1})"             if w_prev1 else "W-1",
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
# 11. UI (CSS + 헬퍼)
# ══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;700;900&display=swap');
      :root {
        --background: #ffffff; --foreground: #0f172a;
        --muted: #f8fafc; --muted-foreground: #64748b;
        --border: rgba(226,232,240,0.8); --primary: #6366f1;
        --primary-hover: #4f46e5; --primary-lt: rgba(99,102,241,0.1);
        --success: #22c55e; --success-fg: #16a34a; --success-lt: rgba(34,197,94,0.1);
        --danger: #ef4444; --danger-fg: #dc2626; --danger-lt: rgba(239,68,68,0.1);
        --warning: #f59e0b; --warning-fg: #d97706; --warning-lt: rgba(245,158,11,0.1);
        --info: #3b82f6; --info-lt: rgba(59,130,246,0.1);
        --card-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
        --card-radius: 12px; --btn-radius: 8px; --badge-radius: 9999px;
        --transition: all 150ms cubic-bezier(0.4,0,0.2,1);
      }
      html, body, [class*="css"] { font-family: 'Inter','Noto Sans KR',sans-serif; color: var(--foreground); background-color: #F0F2F5; }
      .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1500px; background-color: #F0F2F5; }
      section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%); border-right: 1px solid rgba(255,255,255,0.06); min-width: 200px !important; max-width: 215px !important; }
      section[data-testid="stSidebar"] > div { padding: 0 !important; }
      section[data-testid="stSidebar"] label { font-size: 10px !important; color: rgba(148,163,184,0.7) !important; font-weight: 700 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
      section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div { font-size: 12px !important; min-height: 30px !important; padding: 2px 8px !important; background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e2e8f0 !important; border-radius: 6px !important; }
      .kpi { background: var(--background); border: 1px solid var(--border); border-radius: var(--card-radius); padding: 20px 22px; box-shadow: var(--card-shadow); transition: var(--transition); position: relative; overflow: hidden; }
      .kpi::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--primary); border-radius: 3px 0 0 3px; }
      .kpi:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); transform: translateY(-1px); }
      .kpi-label { font-size: 11px; font-weight: 700; color: var(--muted-foreground); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
      .kpi-value { font-size: 26px; font-weight: 800; color: var(--foreground); line-height: 1.1; letter-spacing: -0.025em; }
      .kpi-delta { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 700; margin-top: 8px; padding: 2px 8px; border-radius: var(--badge-radius); background: rgba(100,116,139,0.08); color: var(--muted-foreground); }
      .section-title { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; color: var(--foreground); letter-spacing: -0.01em; margin: 20px 0 10px 0; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
      .section-title-icon { width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; background: var(--primary-lt); border-radius: 6px; font-size: 13px; }
      .page-header { background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#2563eb 100%); border-radius: var(--card-radius); padding: 24px 28px; margin-bottom: 20px; color: white; box-shadow: 0 4px 16px rgba(99,102,241,0.18); position: relative; overflow: hidden; }
      .page-header::after { content: ''; position: absolute; right: -40px; top: -40px; width: 160px; height: 160px; background: rgba(255,255,255,0.04); border-radius: 50%; }
      .page-header-title { font-size: 20px; font-weight: 800; letter-spacing: -0.03em; line-height: 1.3; }
      .page-header-sub { font-size: 12px; opacity: 0.65; margin-top: 6px; font-weight: 400; letter-spacing: 0.01em; }
      .badge { display: inline-flex; align-items: center; border-radius: var(--badge-radius); font-size: 11px; font-weight: 700; padding: 2px 10px; letter-spacing: 0.02em; }
      .badge-default { background: var(--primary-lt); color: var(--primary); border: 1px solid rgba(99,102,241,0.2); }
      .badge-green { background: var(--success-lt); color: var(--success-fg); border: 1px solid rgba(34,197,94,0.2); }
      .badge-red { background: var(--danger-lt); color: var(--danger-fg); border: 1px solid rgba(239,68,68,0.2); }
      .badge-amber { background: var(--warning-lt); color: var(--warning-fg); border: 1px solid rgba(245,158,11,0.2); }
      div.stButton > button { border-radius: var(--btn-radius) !important; font-weight: 600 !important; font-size: 13px !important; height: 36px !important; padding: 0 16px !important; transition: var(--transition) !important; }
      div[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; border: 1px solid var(--border); }
      div[data-testid="stTabs"] > div:first-child { background: #f1f5f9; border-radius: 8px; padding: 4px; gap: 2px; border: none !important; }
      div[data-testid="stTabs"] button[role="tab"] { border-radius: 6px !important; font-size: 13px !important; font-weight: 500 !important; color: var(--muted-foreground) !important; transition: var(--transition) !important; border: none !important; padding: 6px 14px !important; }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: var(--background) !important; color: var(--foreground) !important; font-weight: 700 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important; }
      .spacer-sm { height: 8px; } .spacer-md { height: 16px; } .spacer-lg { height: 24px; }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, delta="-", delta_color=None):
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


def section_title(text, icon=""):
    icon_html = f'<div class="section-title-icon">{icon}</div>' if icon else ""
    st.markdown(f'<div class="section-title">{icon_html}<span>{text}</span></div>', unsafe_allow_html=True)


def page_header(title, sub=""):
    sub_html = f"<div class='page-header-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-title">{title}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 12. 월/발송 필터 헬퍼
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
        ms = fm_sent(df_all, m); mr = fm(df_scored_all, m)
        t  = len(ms);            r  = len(mr)
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


# ── 13-1. 개요 ───────────────────────────────────────────────────
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
    with c1: kpi_card("발송건수",  f"{total_sent:,}건",  d_sent_str,  dcol(d_sent_str))
    with c2: kpi_card("응답건수",  f"{total_scored:,}건", d_resp_str, dcol(d_resp_str))
    with c3: kpi_card("응답률",    f"{resp_rate}%",       d_rate_str, dcol(d_rate_str))
    with c4: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점", d_score_str, dcol(d_score_str))
    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5: kpi_card("친절점수",      "-" if avg_kind  is None else f"{avg_kind}점")
    with c6: kpi_card("만족점수",      "-" if avg_satis is None else f"{avg_satis}점")
    with c7: kpi_card("부정응답",      f"{neg_cnt:,}건", "-", C_RED)
    with c8: kpi_card("점수갭(20점↑)", f"{gap_cnt:,}건", "-", C_AMBER)
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    monthly_rate_df, monthly_score_df = compute_monthly_trends(df_all, df_scored_all)
    t1, t2 = st.columns(2)
    with t1:
        section_title("응답률 변화 트렌드", "📈")
        if monthly_rate_df.empty:
            st.caption("데이터 없음")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_rate_df["월"], y=monthly_rate_df["응답률(%)"],
                mode="lines+markers", name="응답률(%)",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=8, color="#6366f1", line=dict(color="white", width=2)),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12), showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#64748b")),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", tickfont=dict(size=11, color="#64748b")))
            st.plotly_chart(fig, use_container_width=True)
    with t2:
        section_title("월별 친절·만족·최종 점수 추이", "📊")
        if monthly_score_df.empty:
            st.caption("데이터 없음")
        else:
            fig = go.Figure()
            color_map = {"친절점수": ("#f59e0b","rgba(245,158,11,0.08)"), "만족점수": ("#22c55e","rgba(34,197,94,0.08)"), "최종점수": ("#6366f1","rgba(99,102,241,0.08)")}
            for sc, (color, fill) in color_map.items():
                if sc in monthly_score_df.columns:
                    fig.add_trace(go.Scatter(x=monthly_score_df["월"], y=monthly_score_df[sc],
                        mode="lines+markers", name=sc, line=dict(color=color, width=2.5),
                        marker=dict(size=7, color=color, line=dict(color="white", width=2))))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=30),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, Noto Sans KR", size=12),
                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#64748b")),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", tickfont=dict(size=11, color="#64748b")))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 1, 1.3])
    with b1:
        section_title("채널별 응답률", "📡")
        if "채널_구분" in df_m_all.columns:
            st.dataframe(calc_response_rate(df_m_all, df_m_kpi, "채널_구분"), use_container_width=True, hide_index=True)
        else:
            st.caption("채널 컬럼 없음")
    with b2:
        section_title("긍정/부정 분포", "😊")
        if "긍정부정" in df_m_kpi.columns:
            s = df_m_kpi["긍정부정"].value_counts().reset_index()
            s.columns = ["긍정/부정","건수"]
            fig = go.Figure(go.Pie(labels=s["긍정/부정"], values=s["건수"], hole=0.62,
                marker=dict(colors=["#22c55e","#f59e0b","#ef4444"], line=dict(color="white", width=2))))
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="white",
                legend=dict(orientation="v", x=1.02, y=0.5))
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


# ── 13-2. 일자·주차 ──────────────────────────────────────────────
def page_day_week(df_all, df_scored, df_scored_all, selected_date, selected_week):
    try:
        page_header("일자 / 주차 리포트", f"날짜: {selected_date or '미선택'}  |  주차: {selected_week or '미선택'}")
    except Exception:
        st.title("일자 / 주차 리포트")
    if not selected_date and not selected_week:
        st.info("💡 사이드바 상단에서 일자 또는 주차를 선택하면 상세 분석이 표시됩니다.")
        try:
            if "회신일" in df_scored_all.columns and "최종점수" in df_scored_all.columns:
                daily_sum = (df_scored_all[df_scored_all["회신일"].notna()]
                    .groupby("회신일").agg(응답건수=("최종점수","count"), 평균점수=("최종점수","mean"))
                    .round(1).reset_index().sort_values("회신일", ascending=True))
                chart_data = daily_sum.tail(30).copy()
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    section_title("일별 응답건수 (최근 30일)", "")
                    fig_ds = go.Figure(go.Bar(x=list(chart_data["회신일"]), y=list(chart_data["응답건수"]),
                        marker=dict(color="#6366f1", opacity=0.8)))
                    fig_ds.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(showgrid=False, tickangle=-45), yaxis=dict(showgrid=True))
                    st.plotly_chart(fig_ds, use_container_width=True)
                with col_s2:
                    section_title("일별 평균점수 (최근 30일)", "")
                    fig_dp = go.Figure(go.Scatter(x=list(chart_data["회신일"]), y=list(chart_data["평균점수"]),
                        mode="lines+markers", line=dict(color="#6366f1", width=2.5),
                        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)"))
                    fig_dp.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
                    fig_dp.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
                    fig_dp.update_layout(height=300, margin=dict(l=10,r=80,t=10,b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(showgrid=False, tickangle=-45),
                        yaxis=dict(showgrid=True, range=[50,105]))
                    st.plotly_chart(fig_dp, use_container_width=True)
                section_title("일별 응답 현황 전체 목록", "")
                st.dataframe(daily_sum.sort_values("회신일", ascending=False), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"일별 요약 표시 중 오류: {e}")
    tab_daily, tab_weekly = st.tabs(["📅 DAILY 상세", "📆 WEEKLY 상세"])
    with tab_daily:
        if not selected_date:
            st.info("👆 사이드바 상단 일자(선택)에서 날짜를 선택하세요.")
        else:
            df_day_all = filter_by_date_sent(df_all, selected_date)
            df_day_kpi = filter_by_date(df_scored_all, selected_date, date_col="회신일")
            if "긍정부정" not in df_day_kpi.columns and not df_day_kpi.empty:
                df_day_kpi = add_sentiment_column(df_day_kpi)
            if df_day_kpi.empty:
                st.warning(f"'{selected_date}' 에 해당하는 응답 데이터가 없습니다.")
            else:
                total_sent   = len(df_day_all); total_scored = len(df_day_kpi)
                resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0
                avg_final    = safe_mean(df_day_kpi, "최종점수")
                avg_kind     = safe_mean(df_day_kpi, "친절점수")
                avg_satis    = safe_mean(df_day_kpi, "만족점수")
                neg_cnt      = len(df_day_kpi[df_day_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_day_kpi.columns else 0
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: kpi_card("발송건수",  f"{total_sent:,}건")
                with c2: kpi_card("응답건수",  f"{total_scored:,}건")
                with c3: kpi_card("응답률",    f"{resp_rate}%")
                with c4: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점")
                with c5: kpi_card("친절/만족", f"{avg_kind or '-'} / {avg_satis or '-'}")
                with c6: kpi_card("부정응답",  f"{neg_cnt}건")
                st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
                d1, d2 = st.columns(2)
                with d1:
                    section_title("채널별 응답률 및 점수", "📡")
                    ch_rate_d  = calc_response_rate(df_day_all, df_day_kpi, "채널_구분")
                    ch_score_d = pivot_avg(df_day_kpi, "채널_구분")
                    try:
                        if not ch_score_d.empty and "채널_구분" in ch_score_d.columns:
                            _cs = ch_score_d.rename(columns={"채널_구분": "구분"})
                            ch_d = ch_rate_d.merge(_cs, on="구분", how="left") if "구분" in ch_rate_d.columns else ch_rate_d
                            ch_d = ch_d.loc[:, ~ch_d.columns.duplicated()]
                        else:
                            ch_d = ch_rate_d
                    except Exception:
                        ch_d = ch_rate_d
                    st.dataframe(ch_d, use_container_width=True, hide_index=True)
                    section_title("긍정/부정 분포", "😊")
                    st.dataframe(sentiment_summary(df_day_kpi), use_container_width=True, hide_index=True)
                with d2:
                    section_title("브랜드별 평균 점수", "🏷️")
                    if "브랜드" in df_day_kpi.columns:
                        st.dataframe(pivot_avg(df_day_kpi, "브랜드"), use_container_width=True, hide_index=True)
                    section_title("상담사별 점수 (재직자)", "👤")
                    if "상담사" in df_day_kpi.columns:
                        st.dataframe(pivot_avg(df_day_kpi, "상담사", agent_filter=True), use_container_width=True, hide_index=True)
                if "긍정부정" in df_day_kpi.columns:
                    neg_d = df_day_kpi[df_day_kpi["긍정부정"] == "부정"]
                    if not neg_d.empty:
                        section_title(f"부정 응답 상세 ({len(neg_d)}건)", "⚠️")
                        st.dataframe(neg_d[get_display_cols(neg_d, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식","긍정부정"])], use_container_width=True, hide_index=True)
                section_title("전체 응답 목록 (일자)", "📋")
                st.dataframe(df_day_kpi[get_display_cols(df_day_kpi, ["회신일","상담사","브랜드","채널_구분","상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"])].reset_index(drop=True), use_container_width=True, hide_index=True)
    with tab_weekly:
        if not selected_week:
            st.info("👆 사이드바 상단 주차(선택)에서 주차를 선택하세요.")
        else:
            df_week_all = filter_by_week_sent(df_all, selected_week)
            df_week_kpi = filter_by_week(df_scored_all, selected_week, week_col="회신주차_정제")
            if "긍정부정" not in df_week_kpi.columns and not df_week_kpi.empty:
                df_week_kpi = add_sentiment_column(df_week_kpi)
            if df_week_kpi.empty:
                st.warning(f"'{selected_week}' 에 해당하는 응답 데이터가 없습니다.")
            else:
                total_sent   = len(df_week_all); total_scored = len(df_week_kpi)
                resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0
                avg_final    = safe_mean(df_week_kpi, "최종점수")
                neg_cnt      = len(df_week_kpi[df_week_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_week_kpi.columns else 0
                gap_cnt      = len(detect_gaps(df_week_kpi))
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: kpi_card("발송건수",  f"{total_sent:,}건")
                with c2: kpi_card("응답건수",  f"{total_scored:,}건")
                with c3: kpi_card("응답률",    f"{resp_rate}%")
                with c4: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점")
                with c5: kpi_card("부정응답",  f"{neg_cnt}건")
                with c6: kpi_card("점수갭",    f"{gap_cnt}건")
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
                w1, w2 = st.columns(2)
                with w1:
                    section_title("채널별 응답률 및 점수", "📡")
                    ch_rate_w  = calc_response_rate(df_week_all, df_week_kpi, "채널_구분")
                    ch_score_w = pivot_avg(df_week_kpi, "채널_구분")
                    try:
                        if not ch_score_w.empty and "채널_구분" in ch_score_w.columns:
                            _csw = ch_score_w.rename(columns={"채널_구분": "구분"})
                            ch_w = ch_rate_w.merge(_csw, on="구분", how="left") if "구분" in ch_rate_w.columns else ch_rate_w
                            ch_w = ch_w.loc[:, ~ch_w.columns.duplicated()]
                        else:
                            ch_w = ch_rate_w
                    except Exception:
                        ch_w = ch_rate_w
                    st.dataframe(ch_w, use_container_width=True, hide_index=True)
                    section_title("긍정/부정 분포", "😊")
                    st.dataframe(sentiment_summary(df_week_kpi), use_container_width=True, hide_index=True)
                with w2:
                    section_title("브랜드별 평균 점수", "🏷️")
                    if "브랜드" in df_week_kpi.columns:
                        st.dataframe(pivot_avg(df_week_kpi, "브랜드"), use_container_width=True, hide_index=True)
                    section_title("상담사별 점수 (재직자)", "👤")
                    if "상담사" in df_week_kpi.columns:
                        st.dataframe(pivot_avg(df_week_kpi, "상담사", agent_filter=True), use_container_width=True, hide_index=True)
                section_title("키워드 TOP 20 (주차)", "🔑")
                kws = extract_keywords(df_week_kpi, 20)
                if kws:
                    kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
                    fig = go.Figure(go.Bar(x=kdf["빈도"], y=kdf["키워드"], orientation="h",
                        marker=dict(color="#6366f1", opacity=0.8), text=kdf["빈도"], textposition="outside"))
                    fig.update_layout(height=500, margin=dict(l=10,r=40,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(showgrid=True), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                if "긍정부정" in df_week_kpi.columns:
                    neg_w = df_week_kpi[df_week_kpi["긍정부정"] == "부정"]
                    if not neg_w.empty:
                        section_title(f"부정 응답 상세 ({len(neg_w)}건)", "⚠️")
                        st.dataframe(neg_w[get_display_cols(neg_w, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식","긍정부정"])], use_container_width=True, hide_index=True)
                section_title("전체 응답 목록 (주차)", "📋")
                st.dataframe(df_week_kpi[get_display_cols(df_week_kpi, ["회신주차_정제","회신일","상담사","브랜드","채널_구분","상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"])].reset_index(drop=True), use_container_width=True, hide_index=True)


# ── 13-3. 점수분석 ───────────────────────────────────────────────
def page_scores(df_m):
    page_header("점수 분석")
    for gcol, title, do_af in [
        ("상담사","상담사별 (재직/제외필터)",True),
        ("브랜드","브랜드별",False),
        ("상담유형대","상담유형(대)별",False),
        ("채널_구분","채널별",False),
        ("근속","근속별",False),
    ]:
        if gcol not in df_m.columns:
            continue
        section_title(title, "📊")
        piv = pivot_avg(df_m, gcol, agent_filter=do_af)
        st.dataframe(piv, use_container_width=True, hide_index=True)
        if "최종점수" in df_m.columns:
            src  = _agent_filter(df_m) if do_af else df_m
            grp  = src.groupby(gcol)["최종점수"].mean().round(1).sort_values(ascending=False).head(30).reset_index()
            grp.columns = [gcol, "최종점수(평균)"]
            fig = go.Figure(go.Bar(x=grp["최종점수(평균)"], y=grp[gcol], orientation="h",
                marker=dict(color=grp["최종점수(평균)"], colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                    cmin=50, cmax=100, line=dict(color="white", width=0.5)),
                text=grp["최종점수(평균)"].astype(str)+"점", textposition="outside"))
            fig.update_layout(height=max(360, len(grp)*32+60), margin=dict(l=10,r=60,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True, range=[0,110]), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    section_title("친절↔만족 점수 갭 분석 (20점↑)", "⚠️")
    gap = detect_gaps(df_m)
    if gap.empty:
        st.caption("없음")
    else:
        st.dataframe(gap[get_display_cols(gap, ["회신일","상담사","브랜드","채널_구분","친절점수","만족점수","최종점수","갭(친절-만족)","주관식"])], use_container_width=True, hide_index=True)


# ── 13-4. 주관식분석 ─────────────────────────────────────────────
def page_verbatim(df_m):
    page_header("주관식 분석")
    section_title("긍정/부정 분포", "😊")
    v1, v2 = st.columns([1, 1.5])
    with v1:
        st.dataframe(sentiment_summary(df_m), use_container_width=True, hide_index=True)
    with v2:
        if "긍정부정" in df_m.columns:
            s = df_m["긍정부정"].value_counts().reset_index()
            s.columns = ["긍정/부정","건수"]
            fig = go.Figure(go.Pie(labels=s["긍정/부정"], values=s["건수"], hole=0.65,
                marker=dict(colors=["#22c55e","#f59e0b","#ef4444"], line=dict(color="white", width=3))))
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="white",
                legend=dict(orientation="v", x=1.02, y=0.5))
            st.plotly_chart(fig, use_container_width=True)
    section_title("키워드 TOP 20", "🔑")
    kws = extract_keywords(df_m, 20)
    if kws:
        kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
        fig = go.Figure(go.Bar(x=kdf["빈도"], y=kdf["키워드"], orientation="h",
            marker=dict(color=kdf["빈도"], colorscale=[[0,"rgba(99,102,241,0.3)"],[1,"#6366f1"]]),
            text=kdf["빈도"], textposition="outside"))
        fig.update_layout(height=540, margin=dict(l=10,r=40,t=10,b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    section_title("부정 응답 상세", "⚠️")
    if "긍정부정" in df_m.columns:
        neg = df_m[df_m["긍정부정"] == "부정"]
        if neg.empty:
            st.caption("없음")
        else:
            st.dataframe(neg[get_display_cols(neg, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식"])], use_container_width=True, hide_index=True)


# ── 13-5. 통합분석(히트맵) ───────────────────────────────────────
def page_integrated(df_m):
    page_header("통합 분석 (히트맵)")
    def heatmap_section(df, idx, col_name, val, title, agent_filter=False, icon="🗺️"):
        src = _agent_filter(df) if agent_filter else df
        if src.empty or idx not in src.columns or col_name not in src.columns or val not in src.columns:
            return
        section_title(title, icon)
        p = src.pivot_table(values=val, index=idx, columns=col_name, aggfunc="mean").round(1)
        if p.empty:
            st.caption("데이터 없음")
            return
        fig = go.Figure(go.Heatmap(z=p.values, x=p.columns.tolist(), y=p.index.tolist(),
            text=p.values, texttemplate="%{text:.1f}", textfont=dict(size=12, color="white"),
            colorscale=[[0,"#ef4444"],[0.3,"#f97316"],[0.55,"#f59e0b"],[0.75,"#22c55e"],[1,"#16a34a"]],
            zmin=60, zmax=100))
        fig.update_layout(height=max(380, len(p)*38+80), margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter, Noto Sans KR", size=12))
        st.plotly_chart(fig, use_container_width=True)
    heatmap_section(df_m, "상담유형대", "채널_구분", "최종점수", "상담유형(대) × 채널 만족도", icon="🗺️")
    heatmap_section(df_m, "상담사", "채널_구분", "최종점수", "상담사 × 채널 만족도 (재직자 기준)", agent_filter=True, icon="👤")
    inq_col = next((c for c in ["문의유형","상담유형대","상담유형중"] if c in df_m.columns), None)
    if inq_col and "상담사" in df_m.columns:
        heatmap_section(df_m, "상담사", inq_col, "최종점수", f"상담사 × {inq_col} 만족도", agent_filter=True, icon="📋")
    heatmap_section(df_m, "브랜드", "상담유형대", "최종점수", "브랜드 × 상담유형(대) 만족도", icon="🏷️")
    if "근속" in df_m.columns:
        heatmap_section(df_m, "근속", "채널_구분", "최종점수", "근속 × 채널 만족도", icon="📅")
        section_title("근속별 평균 점수", "📊")
        st.dataframe(pivot_avg(df_m, "근속"), use_container_width=True, hide_index=True)


# ── 13-6. Action 필요 ────────────────────────────────────────────
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
            st.dataframe(neg[get_display_cols(neg, ["회신일","상담사","브랜드","채널_구분","최종점수","주관식"])], use_container_width=True, hide_index=True)


# ── 13-7. 일별 상담사 성과 ──────────────────────────────────────
def page_daily_agent(df_m):
    page_header("일별 상담사 성과")
    df_v = _agent_filter(df_m)
    if df_v.empty or "회신일" not in df_v.columns or "상담사" not in df_v.columns or "최종점수" not in df_v.columns:
        st.warning("일별 상담사 성과 데이터가 부족합니다.")
        return
    pivot_df = df_v.groupby(["회신일","상담사"])["최종점수"].mean().round(1).unstack(fill_value=None)
    if pivot_df.empty:
        st.caption("피벗 데이터 없음")
        return
    section_title("일자별 상담사 최종점수 트렌드", "📈")
    p_long = pivot_df.reset_index().melt(id_vars=["회신일"], var_name="상담사", value_name="최종점수")
    p_long = p_long.dropna()
    fig = go.Figure()
    for i, agent in enumerate(p_long["상담사"].unique()):
        sub = p_long[p_long["상담사"] == agent]
        color = CHART_COLORS[i % len(CHART_COLORS)]
        fig.add_trace(go.Scatter(x=sub["회신일"], y=sub["최종점수"], mode="lines+markers", name=agent,
            line=dict(color=color, width=2), marker=dict(size=7, color=color)))
    fig.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
    fig.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
    fig.update_layout(height=440, margin=dict(l=10,r=120,t=10,b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="v", x=1.02, y=0.5),
        yaxis=dict(range=[0,105]))
    st.plotly_chart(fig, use_container_width=True)
    section_title("일자별 상담사 피벗 테이블", "📋")
    pivot_disp = pivot_df.reset_index()
    pivot_disp.columns.name = None
    st.dataframe(pivot_disp, use_container_width=True, hide_index=True)
    section_title("상담사별 성과 요약", "👤")
    daily_avg = df_v.groupby("상담사")["최종점수"].agg(["mean","min","max","count"]).round(1).reset_index()
    daily_avg.columns = ["상담사","평균점수","최저점수","최고점수","응답건수"]
    daily_avg["상태"] = daily_avg["평균점수"].apply(
        lambda x: "🔴 주의" if x < SCORE_CAUTION else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
    daily_avg = daily_avg.sort_values("평균점수", ascending=False)
    st.dataframe(daily_avg, use_container_width=True, hide_index=True)
    if "채널_구분" in df_v.columns:
        section_title("일자별 채널별 응답 건수", "📡")
        daily_ch = df_v.groupby(["회신일","채널_구분"])["최종점수"].count().unstack(fill_value=0).reset_index()
        daily_ch.columns.name = None
        st.dataframe(daily_ch, use_container_width=True, hide_index=True)


# ── 13-8. 70점 미만 전체 ─────────────────────────────────────────
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
    k1, k2, k3 = st.columns(3)
    with k1: kpi_card("70점 미만 건수", f"{len(low):,}건")
    with k2: kpi_card("평균 점수", f"{low['최종점수'].mean():.1f}점")
    with k3:
        if "상담사" in low.columns:
            kpi_card("해당 상담사 수", f"{low['상담사'].nunique():,}명")
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        section_title("점수 구간별 분포", "📊")
        fig = go.Figure(go.Histogram(x=low["최종점수"], nbinsx=14,
            marker=dict(color="#ef4444", opacity=0.85, line=dict(color="white", width=1))))
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="최종점수"), yaxis=dict(title="건수"))
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        section_title("상담사별 70점 미만 건수 TOP 15", "👤")
        if "상담사" in low.columns:
            agent_low = low.groupby("상담사")["최종점수"].count().sort_values(ascending=False).head(15).reset_index()
            agent_low.columns = ["상담사","건수"]
            fig2 = go.Figure(go.Bar(x=agent_low["건수"], y=agent_low["상담사"], orientation="h",
                marker=dict(color="#ef4444", opacity=0.8), text=agent_low["건수"], textposition="outside"))
            fig2.update_layout(height=280, margin=dict(l=10,r=40,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
    section_title(f"70점 미만 전체 목록 ({len(low):,}건)", "📋")
    preferred = ["회신월_정제","회신일","상담사","브랜드","채널_구분","상담유형대","근속","친절점수","만족점수","최종점수","주관식","긍정부정"]
    st.dataframe(low[get_display_cols(low, preferred)].reset_index(drop=True), use_container_width=True, hide_index=True)


# ── 13-9. 검색 ───────────────────────────────────────────────────
def page_search(df_scored_all, df_all):
    page_header("검색", "상담KEY 또는 상담사 이름으로 원천 데이터를 검색합니다")
    s1, s2 = st.columns([2, 1])
    with s1:
        search_query = st.text_input("🔍 검색어 입력", placeholder="상담KEY 또는 상담사 이름", label_visibility="collapsed")
    with s2:
        search_type = st.selectbox("검색 유형", ["상담KEY + 상담사 이름 (전체)","상담KEY만","상담사 이름만"], label_visibility="collapsed")
    if not search_query.strip():
        st.info("검색어를 입력하면 해당 데이터를 바로 보여드립니다.")
        preferred = ["회신일","상담사","브랜드","채널_구분","상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"]
        st.dataframe(df_scored_all[get_display_cols(df_scored_all, preferred)].tail(50).reset_index(drop=True), use_container_width=True, hide_index=True)
        return
    q = search_query.strip()
    mask = pd.Series([False] * len(df_scored_all), index=df_scored_all.index)
    if search_type in ["상담KEY + 상담사 이름 (전체)", "상담KEY만"]:
        if "상담KEY" in df_scored_all.columns:
            mask = mask | df_scored_all["상담KEY"].astype(str).str.contains(q, case=False, na=False)
    if search_type in ["상담KEY + 상담사 이름 (전체)", "상담사 이름만"]:
        if "상담사" in df_scored_all.columns:
            mask = mask | df_scored_all["상담사"].astype(str).str.contains(q, case=False, na=False)
    result = df_scored_all[mask].copy()
    if result.empty:
        st.warning(f"'{q}'에 해당하는 데이터가 없습니다.")
        return
    st.success(f"✅ '{q}' 검색 결과: **{len(result):,}건**")
    k1,k2,k3,k4,k5 = st.columns(5)
    avg_f = safe_mean(result,"최종점수"); avg_k = safe_mean(result,"친절점수"); avg_s = safe_mean(result,"만족점수")
    neg_c = len(result[result["긍정부정"]=="부정"]) if "긍정부정" in result.columns else 0
    with k1: kpi_card("검색 결과", f"{len(result):,}건")
    with k2: kpi_card("최종점수", "-" if avg_f is None else f"{avg_f}점")
    with k3: kpi_card("친절점수", "-" if avg_k is None else f"{avg_k}점")
    with k4: kpi_card("만족점수", "-" if avg_s is None else f"{avg_s}점")
    with k5: kpi_card("부정응답", f"{neg_c}건")
    preferred = ["회신월_정제","회신일","상담사","브랜드","채널_구분","상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"]
    st.dataframe(result[get_display_cols(result, preferred)].reset_index(drop=True), use_container_width=True, hide_index=True)


# ── 13-10. 인사이트 ──────────────────────────────────────────────
def page_insight(df_all, df_scored, df_scored_all, available_months, target_month):
    page_header("💡 QA 인사이트 리포트", f"센터장·QA팀 회의용 종합 분석  |  기준월: {target_month}")
    df_m     = fm(df_scored,     target_month)
    df_m_kpi = fm(df_scored_all, target_month)
    df_m_all = fm_sent(df_all,   target_month)
    sorted_m  = sorted([m for m in available_months if m <= target_month]) if target_month else []
    prev_m    = sorted_m[-2] if len(sorted_m) >= 2 else None
    df_prev   = fm(df_scored_all, prev_m) if prev_m else pd.DataFrame()
    section_title("이달의 핵심 지표 요약", "📌")
    total_sent   = len(df_m_all); total_scored = len(df_m_kpi)
    resp_rate    = round(total_scored / total_sent * 100, 1) if total_sent > 0 else 0
    avg_final    = safe_mean(df_m_kpi, "최종점수")
    prev_final   = safe_mean(df_prev, "최종점수") if not df_prev.empty else None
    _, d_score_str = calc_mom(avg_final, prev_final, is_pp=False)
    neg_cnt = len(df_m_kpi[df_m_kpi["긍정부정"] == "부정"]) if "긍정부정" in df_m_kpi.columns else 0
    pos_cnt = len(df_m_kpi[df_m_kpi["긍정부정"] == "긍정"]) if "긍정부정" in df_m_kpi.columns else 0
    i1, i2, i3, i4, i5 = st.columns(5)
    with i1: kpi_card("응답건수",  f"{total_scored:,}건")
    with i2: kpi_card("응답률",    f"{resp_rate}%")
    with i3: kpi_card("최종점수",  "-" if avg_final is None else f"{avg_final}점", d_score_str, dcol(d_score_str))
    with i4: kpi_card("긍정 응답", f"{pos_cnt:,}건")
    with i5: kpi_card("부정 응답", f"{neg_cnt:,}건")
    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        section_title("월별 최종점수 추이 (전체)", "📈")
        _, monthly_score_df = compute_monthly_trends(df_all, df_scored_all)
        if not monthly_score_df.empty and "최종점수" in monthly_score_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_score_df["월"], y=monthly_score_df["최종점수"],
                mode="lines+markers+text", text=monthly_score_df["최종점수"].astype(str),
                textposition="top center", line=dict(color="#6366f1", width=3),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.07)"))
            fig.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
            fig.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
            fig.update_layout(height=300, margin=dict(l=10,r=80,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                yaxis=dict(range=[50,105]))
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        section_title("채널별 점수 비교 (이번 달)", "📡")
        if "채널_구분" in df_m_kpi.columns:
            ch_avg = pivot_avg(df_m_kpi, "채널_구분")
            if not ch_avg.empty and "최종점수" in ch_avg.columns:
                fig2 = go.Figure(go.Bar(x=ch_avg["채널_구분"], y=ch_avg["최종점수"],
                    marker=dict(color=ch_avg["최종점수"],
                        colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]], cmin=60, cmax=100),
                    text=ch_avg["최종점수"].astype(str)+"점", textposition="outside"))
                fig2.add_hline(y=SCORE_GOOD, line_dash="dash", line_color="#22c55e", line_width=1.5)
                fig2.add_hline(y=SCORE_CAUTION, line_dash="dash", line_color="#ef4444", line_width=1.5)
                fig2.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(range=[0,110]))
                st.plotly_chart(fig2, use_container_width=True)
    section_title("이달의 주요 이슈 & 시사점", "🚨")
    act = action_needed(df_m, df_m_all)
    if act.empty:
        st.success("✅ 이번 달 특이 이슈 없음")
    else:
        priority_color = {
            "🔴 긴급": "background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;",
            "🟡 주의": "background:rgba(245,158,11,0.08);border-left:3px solid #f59e0b;",
            "🟠 개선": "background:rgba(249,115,22,0.08);border-left:3px solid #f97316;",
        }
        for _, row in act.iterrows():
            style = priority_color.get(row.get("우선순위",""), "background:#f8fafc;border-left:3px solid #64748b;")
            st.markdown(f'<div style="{style} padding:10px 16px;border-radius:0 8px 8px 0;margin-bottom:6px;"><span style="font-size:12px;font-weight:700;">[{row.get("구분","")}] {row.get("항목","")}</span><span style="font-size:11px;color:#64748b;margin-left:10px;">{row.get("내용","")}</span></div>', unsafe_allow_html=True)
    section_title("상담사 상태 분포 (신호등)", "🚦")
    if "상담사" in df_m.columns and "최종점수" in df_m.columns:
        ag_perf = _agent_filter(df_m).groupby("상담사")["최종점수"].mean().round(1).reset_index()
        ag_perf.columns = ["상담사","평균점수"]
        red_agents   = ag_perf[ag_perf["평균점수"] < SCORE_CAUTION]
        amber_agents = ag_perf[(ag_perf["평균점수"] >= SCORE_CAUTION) & (ag_perf["평균점수"] < SCORE_GOOD)]
        green_agents = ag_perf[ag_perf["평균점수"] >= SCORE_GOOD]
        sg1, sg2, sg3 = st.columns(3)
        def _agent_list_html(df_ag):
            if df_ag.empty:
                return '<div style="font-size:12px;color:#64748b;">해당 없음</div>'
            return "".join([f'<div style="font-size:12px;padding:2px 0;">{r["상담사"]} <b>{r["평균점수"]}점</b></div>' for _, r in df_ag.iterrows()])
        with sg1:
            st.markdown(f'<div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:14px 16px;"><div style="font-size:11px;font-weight:700;color:#dc2626;margin-bottom:8px;">🔴 즉시 코칭 ({len(red_agents)}명)</div>{_agent_list_html(red_agents)}</div>', unsafe_allow_html=True)
        with sg2:
            st.markdown(f'<div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:14px 16px;"><div style="font-size:11px;font-weight:700;color:#d97706;margin-bottom:8px;">🟡 모니터링 ({len(amber_agents)}명)</div>{_agent_list_html(amber_agents)}</div>', unsafe_allow_html=True)
        with sg3:
            st.markdown(f'<div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:14px 16px;"><div style="font-size:11px;font-weight:700;color:#16a34a;margin-bottom:8px;">🟢 양호 ({len(green_agents)}명)</div>{_agent_list_html(green_agents)}</div>', unsafe_allow_html=True)


# ── 13-11. 교육자료 ──────────────────────────────────────────────
def page_education(df_m, df_scored_all, target_month):
    page_header("🎓 QA 교육·코칭 자료", f"기준월: {target_month}")
    tab_coach, tab_keyword, tab_case, tab_best = st.tabs(["🧑‍🏫 코칭 대상 분석","🔑 키워드·VOC 패턴","⚠️ 개선 사례","🌟 우수 사례"])
    with tab_coach:
        section_title("코칭 우선순위 매트릭스", "🎯")
        if "상담사" in df_m.columns and "최종점수" in df_m.columns:
            src = _agent_filter(df_m)
            coach_df = src.groupby("상담사").agg(평균점수=("최종점수","mean"), 응답건수=("최종점수","count"), 최저점수=("최종점수","min")).round(1).reset_index()
            if "긍정부정" in src.columns:
                neg_per = src[src["긍정부정"]=="부정"].groupby("상담사").size().reset_index(name="부정건수")
                coach_df = coach_df.merge(neg_per, on="상담사", how="left")
                coach_df["부정건수"] = coach_df["부정건수"].fillna(0).astype(int)
            coach_df["코칭등급"] = coach_df["평균점수"].apply(lambda x: "🔴 즉시코칭" if x < SCORE_CAUTION else ("🟡 관찰" if x < SCORE_GOOD else "🟢 양호"))
            coach_df = coach_df.sort_values("평균점수", ascending=True)
            st.dataframe(coach_df, use_container_width=True, hide_index=True)
    with tab_keyword:
        section_title("이달의 키워드 TOP 30", "🔑")
        kws = extract_keywords(df_m, 30)
        if kws:
            kdf = pd.DataFrame(kws, columns=["키워드","빈도"])
            fig = go.Figure(go.Bar(x=kdf["빈도"], y=kdf["키워드"], orientation="h",
                marker=dict(color="#6366f1"), text=kdf["빈도"], textposition="outside"))
            fig.update_layout(height=680, margin=dict(l=10,r=40,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    with tab_case:
        section_title("교육 활용 개선 사례 (부정 응답)", "📖")
        if "긍정부정" in df_m.columns:
            neg_cases = df_m[df_m["긍정부정"]=="부정"].copy()
            if neg_cases.empty:
                st.success("이번 달 부정 응답 없음!")
            else:
                preferred = ["회신일","상담사","브랜드","채널_구분","상담유형대","친절점수","만족점수","최종점수","주관식","긍정부정"]
                st.dataframe(neg_cases.sort_values("최종점수")[get_display_cols(neg_cases, preferred)].reset_index(drop=True), use_container_width=True, hide_index=True)
    with tab_best:
        section_title("교육 활용 우수 사례 (긍정+고점수)", "🌟")
        if "긍정부정" in df_m.columns and "최종점수" in df_m.columns:
            best_cases = df_m[(df_m["긍정부정"]=="긍정") & (df_m["최종점수"]>=SCORE_GOOD)].copy().sort_values("최종점수", ascending=False)
            if best_cases.empty:
                st.info("이번 달 우수 사례 데이터 없음")
            else:
                preferred_b = ["회신일","상담사","브랜드","채널_구분","친절점수","만족점수","최종점수","주관식"]
                st.dataframe(best_cases[get_display_cols(best_cases, preferred_b)].reset_index(drop=True), use_container_width=True, hide_index=True)


# ── 14-A. 텍스트 인텔리전스 ─────────────────────────────────────
def page_text_intelligence(df_m, df_scored_all, target_month):
    page_header("🧠 텍스트 인텔리전스", f"감성·의도 심층 분석  |  기준월: {target_month}")
    voc_col   = next((c for c in ["주관식","verbatim","Q3","의견"] if c in df_m.columns), None)
    sent_col  = "긍정부정"
    intent_col = "의도분류"
    if not voc_col:
        st.warning("주관식 컬럼이 없습니다.")
        return
    src = df_m[df_m[voc_col].notna() & (df_m[voc_col].astype(str).str.strip() != "")].copy()
    if src.empty:
        st.info("분석할 텍스트 데이터가 없습니다.")
        return
    model_badge = "🟢 ML 모델 활성" if _MODEL_TRAINED else "🟡 규칙 기반 폴백"
    st.markdown(f'<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:8px;padding:10px 16px;font-size:12px;margin-bottom:16px;"><b>분류 엔진:</b> {model_badge} &nbsp;|&nbsp; <b>분석 대상:</b> {len(src):,}건</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 감성 트렌드","🎯 의도 분석","🚨 검토 필요 큐","💡 개선요청 트래커"])
    with tab1:
        section_title("감성 분포 (이번 달)", "📊")
        if sent_col in src.columns:
            sent_cnt = src[sent_col].value_counts().reindex(["긍정","중립","부정"], fill_value=0)
            total_s  = sent_cnt.sum()
            c1, c2 = st.columns([1, 2])
            with c1:
                for lbl, cnt in sent_cnt.items():
                    pct = round(cnt / total_s * 100, 1) if total_s > 0 else 0
                    color = {"긍정":"#22c55e","중립":"#f59e0b","부정":"#ef4444"}.get(lbl, C_GRAY)
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 12px;border-radius:8px;margin-bottom:6px;background:rgba(0,0,0,0.02);border:1px solid rgba(0,0,0,0.06);"><span style="font-weight:600;color:{color};">{lbl}</span><span style="font-size:18px;font-weight:800;color:{color};">{cnt:,}건</span><span style="font-size:12px;color:#64748b;">{pct}%</span></div>', unsafe_allow_html=True)
            with c2:
                fig = go.Figure(go.Bar(x=["긍정","중립","부정"],
                    y=[int(sent_cnt.get("긍정",0)), int(sent_cnt.get("중립",0)), int(sent_cnt.get("부정",0))],
                    marker=dict(color=["#22c55e","#f59e0b","#ef4444"])))
                fig.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if intent_col not in src.columns:
            st.info("의도 분류 데이터가 없습니다.")
        else:
            section_title("의도 분포 (전체)", "🎯")
            intent_counter = Counter()
            for v in src[intent_col].dropna():
                for i in str(v).split(","):
                    if i.strip():
                        intent_counter[i.strip()] += 1
            if intent_counter:
                ic_df = pd.DataFrame(intent_counter.most_common(), columns=["의도","건수"])
                fig = go.Figure(go.Bar(x=ic_df["건수"], y=ic_df["의도"], orientation="h",
                    marker=dict(color="#6366f1"), text=ic_df["건수"], textposition="outside"))
                fig.update_layout(height=max(280, len(ic_df)*40+40), margin=dict(l=10,r=40,t=10,b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=True), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
    with tab3:
        section_title("검토 필요 항목", "🚨")
        if "검토필요" in src.columns:
            review_df = src[src["검토필요"] == True].copy()
        elif "분류신뢰도" in src.columns:
            review_df = src[src["분류신뢰도"] < 0.5].copy()
        else:
            review_df = pd.DataFrame()
        if review_df.empty:
            st.success("✅ 검토 필요 항목이 없습니다.")
        else:
            st.warning(f"⚠️ {len(review_df)}건 검토 필요")
            disp = get_display_cols(review_df, ["회신일","상담사","최종점수","주관식","긍정부정","분류신뢰도"])
            st.dataframe(review_df[disp].reset_index(drop=True), use_container_width=True, hide_index=True)
    with tab4:
        section_title("프로세스 개선 요청 목록", "📋")
        if intent_col in src.columns:
            improve_df = src[src[intent_col].str.contains("프로세스개선", na=False)].copy()
            if improve_df.empty:
                st.info("이번 달 프로세스 개선 요청이 없습니다.")
            else:
                st.success(f"✅ {len(improve_df)}건 감지")
                disp = get_display_cols(improve_df, ["회신일","상담사","브랜드","최종점수","주관식"])
                st.dataframe(improve_df[disp].reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("의도 분류 데이터가 없습니다.")


# ── 14-B. 모델 모니터 ────────────────────────────────────────────
def page_model_monitor(df_scored_all, target_month):
    page_header("🔬 모델 모니터", "분류 신뢰도·드리프트·약지도 평가")
    engine_label = "🟢 TF-IDF + LogReg ML 모델" if _MODEL_TRAINED else "🟡 규칙 기반 (Rule Engine)"
    st.markdown(f'<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:8px;padding:10px 16px;font-size:12px;margin-bottom:16px;"><b>현재 엔진:</b> {engine_label}</div>', unsafe_allow_html=True)
    if not _MODEL_TRAINED:
        if st.button("🔄 ML 모델 학습"):
            with st.spinner("학습 중..."):
                success = train_classifiers_from_df(df_scored_all)
            if success:
                st.success("✅ 모델 학습 완료!")
                st.rerun()
    tab_a, tab_b, tab_c = st.tabs(["신뢰도 분포","월별 드리프트","약지도 평가"])
    with tab_a:
        if "분류신뢰도" in df_scored_all.columns and df_scored_all["분류신뢰도"].notna().any():
            conf_data = df_scored_all["분류신뢰도"].dropna()
            low_conf_rate = round((conf_data < 0.55).sum() / len(conf_data) * 100, 1)
            c1, c2 = st.columns(2)
            with c1: kpi_card("평균 신뢰도", f"{conf_data.mean():.3f}")
            with c2: kpi_card("저신뢰도 비율", f"{low_conf_rate}%")
            fig = go.Figure(go.Histogram(x=conf_data, nbinsx=20,
                marker=dict(color="#6366f1", opacity=0.85)))
            fig.add_vline(x=0.55, line_dash="dash", line_color="#ef4444", line_width=1.5)
            fig.update_layout(height=280, margin=dict(l=10,r=80,t=10,b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="신뢰도", range=[0,1]), yaxis=dict(title="건수"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("신뢰도 데이터 없음")
    with tab_b:
        st.info("월별 드리프트 분석은 신뢰도 데이터 누적 후 활성화됩니다.")
    with tab_c:
        st.info("약지도 평가는 ML 모델 학습 후 활성화됩니다.")


# ══════════════════════════════════════════════════════════════════
# ★ QC 모니터링 페이지 (신규)
# ══════════════════════════════════════════════════════════════════
def page_qc_monitoring(df_main, df_qc):
    page_header("📋 이행률 관리 (QC 모니터링)", "70점 미만 RAW × QC 이행 분석")

    if df_qc is None or df_qc.empty:
        st.info("QC 모니터링 데이터가 없습니다. Google Sheets 두 번째 탭(QC모니터링/Sheet2)을 확인하세요.")
        return

    # 현재 보유 컬럼 디버그 (개발 시 확인용 - 운영 시 주석 처리)
    # with st.expander("🔧 QC 데이터 컬럼 확인 (디버그)"):
    #     st.write(df_qc.columns.tolist())
    #     st.dataframe(df_qc.head(3))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 책임 현황",
        "📉 미흡 항목 분석",
        "👤 개인별 분석",
        "🔀 CSAT×QC 교차",
        "🎯 코칭 우선순위",
    ])

    # ────────────────────────────────────
    # TAB 1: 책임 현황
    # ────────────────────────────────────
    with tab1:
        try:
            acc_col   = "accountability"
            month_col = "month_key"
            score_col = "compliance_score"

            total_qc = len(df_qc)
            avg_comp = round(df_qc[score_col].mean(), 1) if score_col in df_qc.columns and df_qc[score_col].notna().any() else None

            # 귀책 비율 KPI
            acc_counts = {}
            if acc_col in df_qc.columns:
                acc_counts = df_qc[acc_col].value_counts().to_dict()
            ibr_pct   = round(acc_counts.get("IBR", 0) / total_qc * 100, 1) if total_qc > 0 else 0
            agent_pct = round(acc_counts.get("상담사", 0) / total_qc * 100, 1) if total_qc > 0 else 0
            cust_pct  = round(acc_counts.get("고객", 0) / total_qc * 100, 1) if total_qc > 0 else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: kpi_card("QC 검토 건수", f"{total_qc:,}건")
            with c2: kpi_card("평균 이행점수", "-" if avg_comp is None else f"{avg_comp}점")
            with c3: kpi_card("IBR 비율",   f"{ibr_pct}%")
            with c4: kpi_card("상담사 귀책", f"{agent_pct}%")
            with c5: kpi_card("고객 귀책",   f"{cust_pct}%")

            st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

            if acc_col in df_qc.columns and month_col in df_qc.columns:
                # 월별 귀책 비율 스택 바
                section_title("월별 귀책 유형 비율 추이", "📈")
                try:
                    cross = df_qc.groupby([month_col, acc_col]).size().unstack(fill_value=0)
                    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
                    cross_pct = cross_pct.reset_index()

                    fig = go.Figure()
                    for acc_type, color in ACCOUNTABILITY_COLORS.items():
                        if acc_type in cross_pct.columns:
                            fig.add_trace(go.Bar(
                                x=cross_pct[month_col],
                                y=cross_pct[acc_type].round(1),
                                name=acc_type,
                                marker=dict(color=color, opacity=0.85),
                                text=cross_pct[acc_type].round(1).astype(str)+"%",
                                textposition="inside",
                            ))
                    fig.update_layout(
                        barmode="stack", height=340,
                        margin=dict(l=10,r=10,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=12),
                        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(title="%", showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"차트 오류: {e}")

            # 이번달 귀책 파이
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                section_title("귀책 분포 (전체)", "🥧")
                if acc_col in df_qc.columns:
                    acc_df = df_qc[acc_col].value_counts().reset_index()
                    acc_df.columns = ["귀책", "건수"]
                    colors_list = [ACCOUNTABILITY_COLORS.get(r, "#9ca3af") for r in acc_df["귀책"]]
                    fig_p = go.Figure(go.Pie(
                        labels=acc_df["귀책"], values=acc_df["건수"], hole=0.55,
                        marker=dict(colors=colors_list, line=dict(color="white", width=3)),
                    ))
                    fig_p.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="white")
                    st.plotly_chart(fig_p, use_container_width=True)
            with col_p2:
                section_title("귀책 × 월별 크로스탭", "📋")
                if acc_col in df_qc.columns and month_col in df_qc.columns:
                    try:
                        cross_raw = df_qc.groupby([month_col, acc_col]).size().unstack(fill_value=0).reset_index()
                        cross_raw.columns.name = None
                        st.dataframe(cross_raw, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"크로스탭 오류: {e}")

        except Exception as e:
            st.error(f"책임 현황 탭 오류: {e}")

    # ────────────────────────────────────
    # TAB 2: 미흡 항목 분석
    # ────────────────────────────────────
    with tab2:
        try:
            section_title("미흡 항목 빈도 분석", "📉")

            # 필터
            f1, f2, f3 = st.columns(3)
            with f1:
                months_qc = []
                if "month_key" in df_qc.columns:
                    months_qc = sorted([m for m in df_qc["month_key"].dropna().unique() if str(m) not in {"nan",""}])
                sel_month_qc = st.selectbox("월 선택", ["전체"] + months_qc, key="qc_month_filter")
            with f2:
                channels_qc = []
                if "qc_channel" in df_qc.columns:
                    channels_qc = sorted(df_qc["qc_channel"].dropna().unique().tolist())
                sel_ch_qc = st.selectbox("채널 선택", ["전체"] + channels_qc, key="qc_ch_filter")
            with f3:
                acc_list_qc = []
                if "accountability" in df_qc.columns:
                    acc_list_qc = sorted(df_qc["accountability"].dropna().unique().tolist())
                sel_acc_qc = st.selectbox("귀책 유형", ["전체"] + acc_list_qc, key="qc_acc_filter")

            df_filt = df_qc.copy()
            if sel_month_qc != "전체" and "month_key" in df_filt.columns:
                df_filt = df_filt[df_filt["month_key"].astype(str) == sel_month_qc]
            if sel_ch_qc != "전체" and "qc_channel" in df_filt.columns:
                df_filt = df_filt[df_filt["qc_channel"].astype(str) == sel_ch_qc]
            if sel_acc_qc != "전체" and "accountability" in df_filt.columns:
                df_filt = df_filt[df_filt["accountability"].astype(str) == sel_acc_qc]

            # 미흡 항목별 빈도
            flag_cols = [c for c in df_filt.columns if c.endswith("_flag")]
            if flag_cols:
                defi_counts = {}
                for fc in flag_cols:
                    base = fc.replace("_flag", "")
                    label = COMPLIANCE_COL_LABELS.get(base, base)
                    defi_counts[label] = int(df_filt[fc].sum())

                defi_df = pd.DataFrame(list(defi_counts.items()), columns=["미흡항목","건수"])
                defi_df = defi_df[defi_df["건수"] > 0].sort_values("건수", ascending=False)

                if not defi_df.empty:
                    b1, b2 = st.columns([2, 1])
                    with b1:
                        fig = go.Figure(go.Bar(
                            x=defi_df["건수"], y=defi_df["미흡항목"], orientation="h",
                            marker=dict(
                                color=defi_df["건수"],
                                colorscale=[[0,"rgba(239,68,68,0.3)"],[1,"#ef4444"]],
                                showscale=False,
                                line=dict(color="white", width=0.5),
                            ),
                            text=defi_df["건수"], textposition="outside",
                        ))
                        fig.update_layout(
                            height=max(300, len(defi_df)*38+40),
                            margin=dict(l=10,r=40,t=10,b=10),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=12),
                            xaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                            yaxis=dict(showgrid=False, autorange="reversed"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with b2:
                        st.dataframe(defi_df, use_container_width=True, hide_index=True)
                else:
                    st.success("필터 조건에서 미흡 항목 없음")

            # 상담사 × 미흡항목 히트맵
            if "qc_agent" in df_filt.columns and flag_cols:
                section_title("상담사 × 미흡항목 히트맵", "🗺️")
                try:
                    agents_qc = df_filt["qc_agent"].dropna().unique()
                    hmap_data = {}
                    for fc in flag_cols:
                        base  = fc.replace("_flag", "")
                        label = COMPLIANCE_COL_LABELS.get(base, base)
                        hmap_data[label] = df_filt.groupby("qc_agent")[fc].sum()
                    hmap_df = pd.DataFrame(hmap_data, index=agents_qc).fillna(0)
                    hmap_df = hmap_df[hmap_df.sum(axis=1) > 0]  # 미흡 있는 상담사만

                    if not hmap_df.empty:
                        fig_h = go.Figure(go.Heatmap(
                            z=hmap_df.values,
                            x=hmap_df.columns.tolist(),
                            y=hmap_df.index.tolist(),
                            text=hmap_df.values.astype(int),
                            texttemplate="%{text}",
                            textfont=dict(size=11),
                            colorscale=[[0,"#f0fdf4"],[0.3,"#fef9c3"],[0.7,"#fed7aa"],[1,"#ef4444"]],
                            zmin=0,
                        ))
                        fig_h.update_layout(
                            height=max(300, len(hmap_df)*36+60),
                            margin=dict(l=10,r=10,t=10,b=10),
                            paper_bgcolor="white", plot_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=11),
                        )
                        st.plotly_chart(fig_h, use_container_width=True)
                except Exception as e:
                    st.warning(f"히트맵 오류: {e}")

            # 채널 비교
            if "qc_channel" in df_filt.columns and flag_cols:
                section_title("채널별 미흡항목 비교 (전화IN vs 채팅)", "📡")
                try:
                    ch_defi = {}
                    for fc in flag_cols:
                        base  = fc.replace("_flag", "")
                        label = COMPLIANCE_COL_LABELS.get(base, base)
                        ch_defi[label] = df_filt.groupby("qc_channel")[fc].sum()
                    ch_df = pd.DataFrame(ch_defi).fillna(0).reset_index()
                    ch_df_melt = ch_df.melt(id_vars="qc_channel", var_name="미흡항목", value_name="건수")

                    fig_c = go.Figure()
                    channels_unique = ch_df_melt["qc_channel"].unique()
                    colors_ch = ["#6366f1","#f59e0b","#22c55e","#ef4444"]
                    for i, ch in enumerate(channels_unique):
                        sub = ch_df_melt[ch_df_melt["qc_channel"] == ch]
                        fig_c.add_trace(go.Bar(
                            x=sub["미흡항목"], y=sub["건수"],
                            name=str(ch),
                            marker=dict(color=colors_ch[i % len(colors_ch)], opacity=0.85),
                        ))
                    fig_c.update_layout(
                        barmode="group", height=320,
                        margin=dict(l=10,r=10,t=10,b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter, Noto Sans KR", size=11),
                        xaxis=dict(tickangle=-30),
                        yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
                    )
                    st.plotly_chart(fig_c, use_container_width=True)
                except Exception as e:
                    st.warning(f"채널 비교 차트 오류: {e}")

        except Exception as e:
            st.error(f"미흡 항목 분석 탭 오류: {e}")

    # ────────────────────────────────────
    # TAB 3: 개인별 분석
    # ────────────────────────────────────
    with tab3:
        try:
            if "qc_agent" not in df_qc.columns:
                st.info("상담사(qc_agent) 컬럼이 없습니다.")
            else:
                agents_list_qc = sorted(df_qc["qc_agent"].dropna().unique().tolist())
                sel_agent_qc = st.selectbox("상담사 선택", agents_list_qc, key="qc_agent_sel")

                agt_df = df_qc[df_qc["qc_agent"] == sel_agent_qc].copy()
                if agt_df.empty:
                    st.warning("데이터 없음")
                else:
                    total_agt = len(agt_df)
                    avg_agt   = round(agt_df["compliance_score"].mean(), 1) if "compliance_score" in agt_df.columns and agt_df["compliance_score"].notna().any() else None
                    top_defi  = "-"
                    if "deficiency_list" in agt_df.columns:
                        all_defis = []
                        for dl in agt_df["deficiency_list"].dropna():
                            if isinstance(dl, list):
                                all_defis.extend(dl)
                        if all_defis:
                            top_defi = Counter(all_defis).most_common(1)[0][0]

                    c1, c2, c3 = st.columns(3)
                    with c1: kpi_card("QC 건수", f"{total_agt}건")
                    with c2: kpi_card("평균 이행점수", "-" if avg_agt is None else f"{avg_agt}점")
                    with c3: kpi_card("최다 미흡", top_defi[:12] if len(str(top_defi)) > 12 else top_defi)

                    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)
                    r1, r2 = st.columns(2)

                    with r1:
                        # 레이더 차트: 4개 카테고리 점수 vs 팀 평균
                        section_title("4개 카테고리 점수 (vs 팀 평균)", "🎯")
                        cat_cols = ["accuracy_score","proficiency_score","kindness_score","promise_score"]
                        cat_labels = ["정확성(30)","숙련도(20)","친절도(30)","약속이행(20)"]
                        cat_max    = [30, 20, 30, 20]

                        agt_cats  = [agt_df[c].mean() if c in agt_df.columns else 0 for c in cat_cols]
                        team_cats = [df_qc[c].mean() if c in df_qc.columns else 0 for c in cat_cols]
                        # 100점 스케일로 정규화
                        agt_norm  = [round(v / m * 100, 1) if m > 0 else 0 for v, m in zip(agt_cats, cat_max)]
                        team_norm = [round(v / m * 100, 1) if m > 0 else 0 for v, m in zip(team_cats, cat_max)]

                        fig_r = go.Figure()
                        fig_r.add_trace(go.Scatterpolar(
                            r=agt_norm + [agt_norm[0]],
                            theta=cat_labels + [cat_labels[0]],
                            fill="toself", name=sel_agent_qc,
                            line=dict(color="#6366f1", width=2),
                            fillcolor="rgba(99,102,241,0.15)",
                        ))
                        fig_r.add_trace(go.Scatterpolar(
                            r=team_norm + [team_norm[0]],
                            theta=cat_labels + [cat_labels[0]],
                            fill="toself", name="팀 평균",
                            line=dict(color="#f59e0b", width=2, dash="dot"),
                            fillcolor="rgba(245,158,11,0.08)",
                        ))
                        fig_r.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                            height=320, margin=dict(l=10,r=10,t=10,b=10),
                            paper_bgcolor="white",
                            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                            font=dict(family="Inter, Noto Sans KR", size=12),
                        )
                        st.plotly_chart(fig_r, use_container_width=True)

                    with r2:
                        # 월별 이행점수 추이
                        section_title("월별 이행점수 추이", "📈")
                        if "month_key" in agt_df.columns and "compliance_score" in agt_df.columns:
                            monthly_agt  = agt_df.groupby("month_key")["compliance_score"].mean().round(1).reset_index()
                            monthly_team = df_qc.groupby("month_key")["compliance_score"].mean().round(1).reset_index() if "month_key" in df_qc.columns else pd.DataFrame()
                            fig_l = go.Figure()
                            fig_l.add_trace(go.Scatter(
                                x=monthly_agt["month_key"], y=monthly_agt["compliance_score"],
                                mode="lines+markers", name=sel_agent_qc,
                                line=dict(color="#6366f1", width=2.5),
                                marker=dict(size=8, color="#6366f1"),
                            ))
                            if not monthly_team.empty:
                                fig_l.add_trace(go.Scatter(
                                    x=monthly_team["month_key"], y=monthly_team["compliance_score"],
                                    mode="lines", name="팀 평균",
                                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                                ))
                            fig_l.update_layout(
                                height=320, margin=dict(l=10,r=10,t=10,b=30),
                                plot_bgcolor="white", paper_bgcolor="white",
                                font=dict(family="Inter, Noto Sans KR", size=12),
                                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.6)", range=[0,105]),
                            )
                            st.plotly_chart(fig_l, use_container_width=True)

                    # 미흡항목 상세
                    section_title("미흡항목 세부 내역", "📋")
                    flag_cols_agt = [c for c in agt_df.columns if c.endswith("_flag")]
                    if flag_cols_agt:
                        agt_defi = {COMPLIANCE_COL_LABELS.get(fc.replace("_flag",""), fc.replace("_flag","")): int(agt_df[fc].sum()) for fc in flag_cols_agt}
                        agt_defi_df = pd.DataFrame(list(agt_defi.items()), columns=["미흡항목","건수"])
                        agt_defi_df = agt_defi_df[agt_defi_df["건수"] > 0].sort_values("건수", ascending=False)
                        if not agt_defi_df.empty:
                            st.dataframe(agt_defi_df, use_container_width=True, hide_index=True)
                        else:
                            st.success("미흡 항목 없음")

                    # 전체 기록
                    section_title("전체 QC 기록", "📝")
                    disp_qc_cols = [c for c in ["month_key","qc_channel","qc_brand","accountability","compliance_score","detail_analysis","deficiency_count"] if c in agt_df.columns]
                    st.dataframe(agt_df[disp_qc_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"개인별 분석 탭 오류: {e}")

    # ────────────────────────────────────
    # TAB 4: CSAT × QC 교차분석
    # ────────────────────────────────────
    with tab4:
        try:
            section_title("CSAT × QC 이행점수 교차분석", "🔀")

            # 조인 키 탐색
            main_key = None
            qc_key   = None
            for mk in ["상담KEY","상담이력KEY","consult_key","상담key"]:
                if mk in df_main.columns:
                    main_key = mk
                    break
            for qk in ["consult_key","상담이력KEY","상담KEY","상담key"]:
                if qk in df_qc.columns:
                    qc_key = qk
                    break

            if not main_key or not qc_key:
                st.warning(f"조인 키를 찾을 수 없습니다. (CSAT키: {main_key}, QC키: {qc_key})")
                st.info("상담KEY 또는 상담이력KEY 컬럼이 양쪽 시트에 있어야 합니다.")
                # 키 없어도 QC 데이터 단독 분석 제공
                if "compliance_score" in df_qc.columns and "accountability" in df_qc.columns:
                    section_title("QC 이행점수 분포 (단독)", "📊")
                    fig_box = go.Figure()
                    for acc_type, color in ACCOUNTABILITY_COLORS.items():
                        sub = df_qc[df_qc["accountability"] == acc_type]["compliance_score"].dropna()
                        if not sub.empty:
                            fig_box.add_trace(go.Box(y=sub, name=acc_type, marker_color=color))
                    fig_box.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="이행점수", range=[0,105]))
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                # 조인
                df_join = df_main.merge(
                    df_qc.rename(columns={qc_key: main_key}),
                    on=main_key, how="inner", suffixes=("_csat","_qc")
                )

                if df_join.empty:
                    st.warning(f"조인 결과가 없습니다. (CSAT: {len(df_main)}건 / QC: {len(df_qc)}건)")
                    st.info("상담KEY 값 형식(문자/숫자)이 양쪽에서 동일한지 확인하세요.")
                else:
                    st.success(f"✅ 조인 성공: {len(df_join):,}건")
                    csat_col = "최종점수"
                    qc_col   = "compliance_score"

                    # 스캐터 플롯
                    if csat_col in df_join.columns and qc_col in df_join.columns:
                        section_title("CSAT 최종점수 × QC 이행점수 산점도", "⭕")
                        acc_col_j = "accountability" if "accountability" in df_join.columns else None
                        fig_s = go.Figure()
                        if acc_col_j:
                            for acc_type, color in ACCOUNTABILITY_COLORS.items():
                                sub = df_join[df_join[acc_col_j] == acc_type]
                                if not sub.empty:
                                    fig_s.add_trace(go.Scatter(
                                        x=sub[csat_col], y=sub[qc_col],
                                        mode="markers", name=acc_type,
                                        marker=dict(color=color, size=7, opacity=0.7,
                                                    line=dict(color="white", width=1)),
                                    ))
                        else:
                            fig_s.add_trace(go.Scatter(
                                x=df_join[csat_col], y=df_join[qc_col],
                                mode="markers",
                                marker=dict(color="#6366f1", size=7, opacity=0.7),
                            ))
                        fig_s.update_layout(
                            height=360, margin=dict(l=10,r=10,t=10,b=10),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=12),
                            xaxis=dict(title="CSAT 최종점수", showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                            yaxis=dict(title="QC 이행점수",   showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                        )
                        st.plotly_chart(fig_s, use_container_width=True)

                        # 귀책별 CSAT 박스플롯
                        if acc_col_j:
                            section_title("귀책 유형별 CSAT 분포", "📦")
                            fig_box = go.Figure()
                            for acc_type, color in ACCOUNTABILITY_COLORS.items():
                                sub = df_join[df_join[acc_col_j] == acc_type][csat_col].dropna()
                                if not sub.empty:
                                    fig_box.add_trace(go.Box(y=sub, name=acc_type, marker_color=color))
                            fig_box.update_layout(
                                height=300, margin=dict(l=10,r=10,t=10,b=10),
                                plot_bgcolor="white", paper_bgcolor="white",
                                yaxis=dict(title="CSAT 최종점수", range=[0,105]),
                            )
                            st.plotly_chart(fig_box, use_container_width=True)

                        # 월별 듀얼 축 트렌드
                        if "month_key" in df_join.columns:
                            section_title("월별 CSAT × QC 점수 추이", "📈")
                            monthly_j = df_join.groupby("month_key").agg(
                                avg_csat=(csat_col, "mean"),
                                avg_qc=(qc_col, "mean"),
                            ).round(1).reset_index()
                            fig_d = go.Figure()
                            fig_d.add_trace(go.Scatter(
                                x=monthly_j["month_key"], y=monthly_j["avg_csat"],
                                mode="lines+markers", name="CSAT 평균",
                                line=dict(color="#6366f1", width=2.5),
                                marker=dict(size=8),
                            ))
                            fig_d.add_trace(go.Scatter(
                                x=monthly_j["month_key"], y=monthly_j["avg_qc"],
                                mode="lines+markers", name="QC 이행점수",
                                line=dict(color="#f59e0b", width=2.5),
                                marker=dict(size=8),
                                yaxis="y2",
                            ))
                            fig_d.update_layout(
                                height=300, margin=dict(l=10,r=60,t=10,b=30),
                                plot_bgcolor="white", paper_bgcolor="white",
                                font=dict(family="Inter, Noto Sans KR", size=12),
                                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                                xaxis=dict(showgrid=False),
                                yaxis=dict(title="CSAT", showgrid=True, range=[0,105]),
                                yaxis2=dict(title="QC", overlaying="y", side="right", range=[0,105]),
                            )
                            st.plotly_chart(fig_d, use_container_width=True)

                    # 조인 상세 테이블
                    section_title("조인 상세 목록", "📋")
                    disp_j = [c for c in [main_key, "상담사", "브랜드", "채널_구분", csat_col, qc_col, "accountability", "긍정부정", "주관식", "detail_analysis"] if c in df_join.columns]
                    st.dataframe(df_join[disp_j].reset_index(drop=True), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"교차분석 탭 오류: {e}")

    # ────────────────────────────────────
    # TAB 5: 코칭 우선순위
    # ────────────────────────────────────
    with tab5:
        try:
            section_title("코칭 우선순위 매트릭스", "🎯")

            flag_cols_all = [c for c in df_qc.columns if c.endswith("_flag")]

            # 미흡항목 × CSAT 영향 매트릭스 (조인 데이터 기반)
            main_key2 = None
            qc_key2   = None
            for mk in ["상담KEY","상담이력KEY","consult_key"]:
                if mk in df_main.columns:
                    main_key2 = mk
                    break
            for qk in ["consult_key","상담이력KEY","상담KEY"]:
                if qk in df_qc.columns:
                    qc_key2 = qk
                    break

            if main_key2 and qc_key2:
                df_join2 = df_main.merge(
                    df_qc.rename(columns={qc_key2: main_key2}),
                    on=main_key2, how="inner", suffixes=("_csat","_qc")
                )
                csat_col2 = "최종점수"

                if not df_join2.empty and csat_col2 in df_join2.columns and flag_cols_all:
                    defi_impact = []
                    for fc in flag_cols_all:
                        base  = fc.replace("_flag","")
                        label = COMPLIANCE_COL_LABELS.get(base, base)
                        freq  = int(df_join2[fc].sum()) if fc in df_join2.columns else 0
                        if freq == 0:
                            continue
                        avg_csat_with_defi    = df_join2[df_join2[fc]==1][csat_col2].mean() if fc in df_join2.columns else None
                        avg_csat_without_defi = df_join2[df_join2[fc]==0][csat_col2].mean() if fc in df_join2.columns else None
                        impact = round((avg_csat_without_defi or 0) - (avg_csat_with_defi or 0), 1)
                        defi_impact.append({"미흡항목": label, "발생빈도": freq, "CSAT영향(점)": impact})

                    if defi_impact:
                        di_df = pd.DataFrame(defi_impact)
                        med_freq   = di_df["발생빈도"].median()
                        med_impact = di_df["CSAT영향(점)"].median()

                        # 사분면 색상
                        def quadrant(row):
                            if row["발생빈도"] >= med_freq and row["CSAT영향(점)"] >= med_impact:
                                return "🔴 즉시개선"
                            elif row["발생빈도"] < med_freq and row["CSAT영향(점)"] >= med_impact:
                                return "🟡 모니터링"
                            elif row["발생빈도"] >= med_freq and row["CSAT영향(점)"] < med_impact:
                                return "🟠 유지"
                            else:
                                return "🟢 Best Practice"

                        di_df["사분면"] = di_df.apply(quadrant, axis=1)
                        color_map_q = {"🔴 즉시개선":"#ef4444","🟡 모니터링":"#f59e0b","🟠 유지":"#f97316","🟢 Best Practice":"#22c55e"}
                        colors_q = [color_map_q.get(q,"#6366f1") for q in di_df["사분면"]]

                        fig_m = go.Figure()
                        fig_m.add_trace(go.Scatter(
                            x=di_df["발생빈도"], y=di_df["CSAT영향(점)"],
                            mode="markers+text",
                            text=di_df["미흡항목"],
                            textposition="top center",
                            textfont=dict(size=10),
                            marker=dict(size=14, color=colors_q, opacity=0.85,
                                        line=dict(color="white", width=2)),
                            hovertemplate="<b>%{text}</b><br>빈도: %{x}<br>CSAT영향: %{y}점<extra></extra>",
                        ))
                        fig_m.add_vline(x=med_freq, line_dash="dash", line_color="#94a3b8", line_width=1)
                        fig_m.add_hline(y=med_impact, line_dash="dash", line_color="#94a3b8", line_width=1)
                        for lbl, (xpos, ypos) in [
                            ("즉시개선", (di_df["발생빈도"].max()*0.9, di_df["CSAT영향(점)"].max()*0.95)),
                            ("모니터링", (di_df["발생빈도"].min()*1.1, di_df["CSAT영향(점)"].max()*0.95)),
                            ("유지",     (di_df["발생빈도"].max()*0.9, di_df["CSAT영향(점)"].min()*1.05)),
                            ("Best Practice",(di_df["발생빈도"].min()*1.1, di_df["CSAT영향(점)"].min()*1.05)),
                        ]:
                            fig_m.add_annotation(text=lbl, x=xpos, y=ypos,
                                showarrow=False, font=dict(size=10, color="#94a3b8"))
                        fig_m.update_layout(
                            height=400, margin=dict(l=10,r=10,t=10,b=10),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(family="Inter, Noto Sans KR", size=12),
                            xaxis=dict(title="미흡 발생빈도", showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                            yaxis=dict(title="CSAT 영향도(점)", showgrid=True, gridcolor="rgba(226,232,240,0.6)"),
                        )
                        st.plotly_chart(fig_m, use_container_width=True)
                        st.dataframe(di_df.sort_values("CSAT영향(점)", ascending=False), use_container_width=True, hide_index=True)

            # 상담사별 코칭 우선순위 테이블
            section_title("상담사별 코칭 우선순위", "👤")
            if "qc_agent" in df_qc.columns and "compliance_score" in df_qc.columns:
                coach_qc = df_qc.groupby("qc_agent").agg(
                    평균이행점수=("compliance_score","mean"),
                    QC건수=("compliance_score","count"),
                    총미흡수=("deficiency_count","sum") if "deficiency_count" in df_qc.columns else ("compliance_score","count"),
                ).round(1).reset_index()

                # 최악 카테고리 식별
                for cat, label in [("kindness_score","친절도"),("accuracy_score","정확성"),("proficiency_score","숙련도"),("promise_score","약속이행")]:
                    if cat in df_qc.columns:
                        worst = df_qc.groupby("qc_agent")[cat].mean()
                        coach_qc["최저카테고리"] = coach_qc["qc_agent"].map(lambda a: label if worst.get(a, 100) < 70 else "-")
                        break

                # 자동 코칭 추천
                def auto_recommend(row_agent):
                    recs = []
                    agent_rows = df_qc[df_qc["qc_agent"] == row_agent]
                    if "kindness_score" in agent_rows.columns and agent_rows["kindness_score"].mean() < 20:
                        recs.append("친절도 집중 코칭")
                    if "acc_guidance_flag" in agent_rows.columns and agent_rows["acc_guidance_flag"].sum() > 0:
                        recs.append("정확성 오안내 — 스크립트 재교육")
                    if "promise_nonfulfill_flag" in agent_rows.columns and agent_rows["promise_nonfulfill_flag"].sum() > 0:
                        recs.append("약속이행 — 콜백 프로세스 점검")
                    return " / ".join(recs) if recs else "이상 없음"

                coach_qc["코칭 추천"] = coach_qc["qc_agent"].apply(auto_recommend)
                coach_qc["코칭등급"]  = coach_qc["평균이행점수"].apply(
                    lambda x: "🔴 즉시코칭" if x < 70 else ("🟡 관찰" if x < 85 else "🟢 양호"))
                coach_qc = coach_qc.sort_values("평균이행점수", ascending=True)
                st.dataframe(coach_qc, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"코칭 우선순위 탭 오류: {e}")


# ══════════════════════════════════════════════════════════════════
# 15. 사이드바 네비게이션
# ══════════════════════════════════════════════════════════════════
MENU_ITEMS = [
    {"key": "개요",             "icon": "🏠", "label": "개요"},
    {"key": "일자주차",         "icon": "📅", "label": "일자·주차"},
    {"key": "점수분석",         "icon": "📊", "label": "점수분석"},
    {"key": "주관식분석",       "icon": "💬", "label": "주관식"},
    {"key": "히트맵",           "icon": "🗺️", "label": "히트맵"},
    {"key": "Action필요",       "icon": "⚠️", "label": "Action"},
    {"key": "상담사성과",       "icon": "👤", "label": "상담사성과"},
    {"key": "70점미만",         "icon": "🔴", "label": "70점미만"},
    {"key": "QC모니터링",       "icon": "📋", "label": "이행률관리"},   # ★ 신규
    {"key": "검색",             "icon": "🔍", "label": "검색"},
    {"key": "인사이트",         "icon": "💡", "label": "인사이트"},
    {"key": "교육자료",         "icon": "🎓", "label": "교육자료"},
    {"key": "텍스트인텔리전스", "icon": "🧠", "label": "텍스트AI"},
    {"key": "모델모니터",       "icon": "🔬", "label": "모델모니터"},
]


def render_sidebar_nav(current_menu: str, available_months=None,
                       df_scored_all=None, df_all=None):
    MISSING_SET = {"", "nan", "NaT", "None", "NaN", "<NA>", "NA"}
    with st.sidebar:
        st.markdown("""
        <div style="padding:10px 12px 8px;border-bottom:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:13px;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em;">📊 CSAT Dashboard</div>
            <div style="font-size:10px;color:rgba(148,163,184,0.6);margin-top:1px;">고객만족도 분석 시스템</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="padding:8px 12px 4px;font-size:9px;font-weight:800;color:rgba(148,163,184,0.55);letter-spacing:1px;text-transform:uppercase;">📆 기간 선택</div>', unsafe_allow_html=True)
        if available_months:
            target_month = st.selectbox("월(필수)", available_months, index=len(available_months)-1, key="sel_month")
        else:
            target_month = st.text_input("월(예: 2026-01)", value="", key="sel_month_txt")
        available_dates = []
        if df_scored_all is not None and "회신일" in df_scored_all.columns:
            available_dates = sorted([str(d) for d in df_scored_all["회신일"].dropna().unique() if str(d) not in MISSING_SET])
        selected_date = st.selectbox("일자(선택)", [""]+available_dates, index=0, key="sel_date")
        selected_date = selected_date or None
        available_weeks = []
        if df_scored_all is not None:
            week_col = "회신주차_정제" if "회신주차_정제" in df_scored_all.columns else "회신주차"
            if week_col in df_scored_all.columns:
                available_weeks = sorted([str(w) for w in df_scored_all[week_col].dropna().unique() if str(w) not in MISSING_SET])
        selected_week = st.selectbox("주차(선택)", [""]+available_weeks, index=0, key="sel_week")
        selected_week = selected_week or None
        st.markdown("<hr style='margin:6px 10px 4px;border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        st.markdown('<div style="padding:4px 12px 3px;font-size:9px;font-weight:800;color:rgba(148,163,184,0.55);letter-spacing:1px;text-transform:uppercase;">메뉴</div>', unsafe_allow_html=True)
        if "menu" not in st.session_state:
            st.session_state["menu"] = MENU_ITEMS[0]["key"]
        for item in MENU_ITEMS:
            key = item["key"]; icon = item["icon"]; label = item["label"]
            is_active = st.session_state["menu"] == key
            display = f"{icon} {label}"
            if is_active:
                st.markdown(f'<div style="margin:1px 6px;padding:6px 10px;border-radius:6px;background:rgba(99,102,241,0.22);border-left:2px solid #6366f1;"><span style="font-size:12px;font-weight:600;color:#a5b4fc;">{display}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="margin:1px 6px;padding:6px 10px;border-radius:6px;border-left:2px solid transparent;"><span style="font-size:12px;font-weight:400;color:rgba(203,213,225,0.8);">{display}</span></div>', unsafe_allow_html=True)
            if st.button(display, key=f"nav_{key}", use_container_width=True):
                st.session_state["menu"] = key
                st.rerun()
        st.markdown("<hr style='margin:6px 10px 4px;border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        st.markdown(f'<div style="padding:2px 12px 6px;font-size:9px;color:rgba(148,163,184,0.5);line-height:1.6;">🟢 양호 ≥{SCORE_GOOD} &nbsp;|&nbsp; 🔴 주의 &lt;{SCORE_CAUTION}</div>', unsafe_allow_html=True)
        if st.button("🔄 새로고침", use_container_width=True, key="refresh_btn"):
            load_from_gsheets.clear()
            st.rerun()
    return st.session_state["menu"], target_month, selected_date, selected_week


# ══════════════════════════════════════════════════════════════════
# 16. Streamlit App 메인
# ══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="CSAT Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    if "menu" not in st.session_state:
        st.session_state["menu"] = "개요"
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

    # ── QC 데이터 로드 (★ 신규) ──
    try:
        df_qc = load_qc_data()
    except Exception as e:
        df_qc = pd.DataFrame()
        st.warning(f"QC 데이터 로드 실패: {e}")

    # ── 수동 부정 라벨 ──
    manual_neg_keys = load_manual_neg_labels()
    if manual_neg_keys:
        df_scored     = apply_manual_neg_labels(df_scored,     manual_neg_keys)
        df_scored_all = apply_manual_neg_labels(df_scored_all, manual_neg_keys)

    # ── 사이드바 ──
    selected_menu, target_month, selected_date, selected_week = render_sidebar_nav(
        st.session_state["menu"],
        available_months=available_months,
        df_scored_all=df_scored_all,
        df_all=df_all,
    )
    st.session_state["menu"] = selected_menu

    df_m     = fm(df_scored,   target_month)
    df_m_all = fm_sent(df_all, target_month)
    menu = st.session_state["menu"]

    try:
        if   menu == "개요":             page_overview(df_all, df_scored, df_scored_all, available_months, target_month, selected_date, selected_week)
        elif menu == "일자주차":         page_day_week(df_all, df_scored, df_scored_all, selected_date, selected_week)
        elif menu == "점수분석":         page_scores(df_m)
        elif menu == "주관식분석":       page_verbatim(df_m)
        elif menu == "히트맵":           page_integrated(df_m)
        elif menu == "Action필요":       page_action(df_m, df_m_all)
        elif menu == "상담사성과":       page_daily_agent(df_m)
        elif menu == "70점미만":         page_low_scores(df_scored_all, target_month=target_month)
        elif menu == "QC모니터링":       page_qc_monitoring(df_scored_all, df_qc)   # ★ 신규
        elif menu == "검색":             page_search(df_scored_all, df_all)
        elif menu == "인사이트":         page_insight(df_all, df_scored, df_scored_all, available_months, target_month)
        elif menu == "교육자료":         page_education(df_m, df_scored_all, target_month)
        elif menu == "텍스트인텔리전스": page_text_intelligence(df_m, df_scored_all, target_month)
        elif menu == "모델모니터":       page_model_monitor(df_scored_all, target_month)
        else:
            st.session_state["menu"] = "개요"
            st.rerun()
    except Exception as e:
        st.error(f"페이지 렌더링 오류 [{menu}]: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")


if __name__ == "__main__":
    main()
