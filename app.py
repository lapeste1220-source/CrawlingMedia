import math
import time
import re
from datetime import datetime, date, timedelta
from dateutil.parser import parse as dateparse

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"  # 공식 엔드포인트 :contentReference[oaicite:3]{index=3}

st.set_page_config(page_title="언어와 매체: 기사 분석 도구", layout="wide")
st.title("📰 언어와 매체 수행평가: 기사 수집 · 분석 (Naver News API)")

# -------------------------
# 1) 초보-friendly 유틸
# -------------------------
def normalize_keywords(raw: str) -> list[str]:
    # 쉼표/줄바꿈/세미콜론으로 분리
    parts = re.split(r"[,\n;]+", raw)
    cleaned = []
    for p in parts:
        k = p.strip()
        if len(k) >= 2:  # 너무 짧은 건 제외
            cleaned.append(k)
    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for k in cleaned:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def naver_api_headers():
    # Streamlit secrets에서 키 읽기 :contentReference[oaicite:4]{index=4}
    try:
        cid = st.secrets["NAVER_CLIENT_ID"]
        csec = st.secrets["NAVER_CLIENT_SECRET"]
    except Exception:
        st.error("secrets 설정이 없습니다. (NAVER_CLIENT_ID / NAVER_CLIENT_SECRET)")
        st.stop()

    return {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec,
    }

def clean_html(text: str) -> str:
    # 네이버 결과에 <b> 태그가 섞여 나오는 경우가 많아서 제거
    return re.sub(r"<[^>]+>", "", text or "")

def within_range(pub_dt: datetime, start_d: date, end_d: date) -> bool:
    return (pub_dt.date() >= start_d) and (pub_dt.date() <= end_d)

# -------------------------
# 2) 네이버 뉴스 API 호출
# -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_one_keyword(keyword: str, start_d: date, end_d: date, target_n: int) -> pd.DataFrame:
    """
    네이버 뉴스 검색 API로 keyword에 대한 기사 목록 수집.
    - display는 100까지 가능(문서 기준). 안전하게 100 사용.
    - start는 1~1000 범위에서 페이지네이션.
    - 기간 필터는 API가 직접 주지 않으므로 pubDate 파싱 후 앱에서 걸러냄.
    """
    headers = naver_api_headers()
    display = 100
    max_start = 1000  # 네이버 검색 API start 제한 범위 내에서만 돌림(일반적으로 문서/관행상) :contentReference[oaicite:5]{index=5}

    rows = []
    start = 1
    safety_pages = 0

    while True:
        params = {
            "query": keyword,
            "display": display,
            "start": start,
            "sort": "date",  # 최신순
        }
        r = requests.get(NAVER_NEWS_URL, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"네이버 API 오류: {r.status_code} / {r.text}")

        data = r.json()
        items = data.get("items", [])
        if not items:
            break

        for it in items:
            pub_raw = it.get("pubDate", "")
            try:
                pub_dt = dateparse(pub_raw)
            except Exception:
                continue

            # 기간 필터
            if not within_range(pub_dt, start_d, end_d):
                continue

            rows.append({
                "keyword": keyword,
                "pubDate": pub_dt.strftime("%Y-%m-%d %H:%M"),
                "press": clean_html(it.get("originallink", "")),  # 원문 링크(보조)
                "title": clean_html(it.get("title", "")),
                "description": clean_html(it.get("description", "")),
                "link": it.get("link", ""),
                "originallink": it.get("originallink", ""),
            })

        # 목표치 달성하면 종료
        if len(rows) >= target_n:
            break

        # 다음 페이지
        start += display
        safety_pages += 1
        if start > max_start:
            break
        if safety_pages >= 12:  # 무한 루프 방지(최대 12페이지=최대 1200 시도 느낌)
            break

        time.sleep(0.2)

    df = pd.DataFrame(rows)
    return df

def dedup_articles(df: pd.DataFrame) -> pd.DataFrame:
    # link 기준 중복 제거
    if "link" in df.columns:
        df = df.drop_duplicates(subset=["link"])
    # title+pubDate로 한 번 더
    if {"title", "pubDate"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["title", "pubDate"])
    return df.reset_index(drop=True)

# -------------------------
# 3) UI: 입력
# -------------------------
with st.sidebar:
    st.header("검색 조건")

    start_d, end_d = st.date_input(
        "기간 설정",
        value=(date.today() - timedelta(days=30), date.today()),
    )

    raw_keywords = st.text_area(
        "키워드 여러 개 입력 (쉼표/줄바꿈 가능)",
        value="저출산, 출생률, 인구절벽",
        height=120,
    )

    target_total = st.number_input("목표 기사 수 (최소 50)", min_value=50, value=60, step=10)
    per_keyword_cap = st.number_input("키워드당 최대 수집 목표(안전장치)", min_value=30, value=120, step=10)

    run = st.button("수집 시작", type="primary")

# -------------------------
# 4) 실행
# -------------------------
if run:
    keywords = normalize_keywords(raw_keywords)
    if not keywords:
        st.warning("키워드를 1개 이상 입력하세요. (예: 저출산, 출생률)")
        st.stop()

    st.info(f"키워드 {len(keywords)}개: {', '.join(keywords)}")
    per_need = math.ceil(target_total / len(keywords))
    per_need = min(per_need, int(per_keyword_cap))

    frames = []
    with st.spinner("기사 수집 중..."):
        for kw in keywords:
            try:
                df_kw = fetch_news_one_keyword(kw, start_d, end_d, per_need)
            except Exception as e:
                st.error(f"'{kw}' 수집 실패: {e}")
                df_kw = pd.DataFrame()
            frames.append(df_kw)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        st.error("수집된 기사가 없습니다. 기간/키워드를 바꿔보세요.")
        st.stop()

    df = dedup_articles(df)

    # 부족하면 1개 키워드로 추가 수집해서 채우기(초보용 단순 보정)
    if len(df) < target_total:
        st.warning(f"현재 {len(df)}개만 수집됨 → 추가 수집 시도")
        remain = target_total - len(df)
        extra = fetch_news_one_keyword(keywords[0], start_d, end_d, remain + 30)
        df = pd.concat([df, extra], ignore_index=True)
        df = dedup_articles(df)

    st.success(f"최종 수집: {len(df)}개 (목표 {target_total})")

    # -------------------------
    # 5) 화면: 기사 목록
    # -------------------------
    st.subheader("① 기사 목록")
    st.dataframe(df[["pubDate", "keyword", "title", "link"]], use_container_width=True)

    st.download_button(
        "CSV 다운로드(기사 목록)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="articles.csv",
        mime="text/csv",
    )

    # -------------------------
    # 6) 화면: 기본 통계 대시보드
    # -------------------------
    st.subheader("② 통계 대시보드")

    # 날짜별 기사량
    df["date"] = df["pubDate"].str.slice(0, 10)
    by_date = df.groupby("date")["title"].count().reset_index(name="count")
    fig1 = px.line(by_date, x="date", y="count", markers=True, title="날짜별 기사량")
    st.plotly_chart(fig1, use_container_width=True)

    # 키워드별 기사량
    by_kw = df.groupby("keyword")["title"].count().reset_index(name="count").sort_values("count", ascending=False)
    fig2 = px.bar(by_kw, x="keyword", y="count", title="키워드별 기사량")
    st.plotly_chart(fig2, use_container_width=True)

    # 제목 강조어(간단 예시)
    st.subheader("③ 제목 강조어 빈도(간단)")
    hype_words = ["충격", "논란", "파장", "긴급", "폭로", "충돌", "경악", "비상", "전격"]
    counts = []
    for w in hype_words:
        counts.append({"word": w, "count": int(df["title"].str.contains(w).sum())})
    hype_df = pd.DataFrame(counts).sort_values("count", ascending=False)
    fig3 = px.bar(hype_df, x="word", y="count", title="강조/선정 표현 빈도(제목 기준)")
    st.plotly_chart(fig3, use_container_width=True)

    st.info("다음 단계: 기사별 ‘근거 문장 2개 + 프레임 체크’ 입력 화면과, HTML 보고서 생성(→PDF 저장)을 붙입니다.")
else:
    st.caption("왼쪽에서 기간/키워드를 넣고 ‘수집 시작’을 누르세요.")
