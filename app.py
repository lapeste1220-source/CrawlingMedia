import math
import time
import re
from datetime import datetime, date, timedelta

import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------
# 설정
# -------------------------
NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

st.set_page_config(page_title="언어와 매체: 기사 분석 도구", layout="wide")
st.title("📰 언어와 매체 수행평가: 기사 수집 · 분석 (Naver News API)")

# -------------------------
# 유틸
# -------------------------
def normalize_keywords(raw: str) -> list[str]:
    parts = re.split(r"[,\n;]+", raw)
    cleaned = []
    for p in parts:
        k = p.strip()
        if len(k) >= 2:
            cleaned.append(k)

    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for k in cleaned:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def safe_text(s: str) -> str:
    """HTML에 넣어도 깨지지 않게 최소한의 escape"""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def parse_pubdate_to_dt(pub_raw: str):
    """
    네이버 pubDate 예: 'Mon, 02 Feb 2026 11:04:00 +0900'
    datetime.strptime로 처리
    """
    try:
        return datetime.strptime(pub_raw, "%a, %d %b %Y %H:%M:%S %z")
    except Exception:
        return None


def naver_api_headers():
    try:
        cid = st.secrets["NAVER_CLIENT_ID"]
        csec = st.secrets["NAVER_CLIENT_SECRET"]
    except Exception:
        st.error("Secrets에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 없습니다.")
        st.stop()

    return {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec,
    }


def dedup_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "link" in df.columns:
        df = df.drop_duplicates(subset=["link"])
    if {"title", "pubDate"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["title", "pubDate"])
    return df.reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_one_keyword(keyword: str, start_d: date, end_d: date, target_n: int, per_page: int = 100) -> pd.DataFrame:
    """
    네이버 뉴스 검색 API로 기사 목록 수집.
    - display 최대 100 (per_page)
    - start 1부터 페이지네이션
    - 기간은 pubDate를 파싱해서 앱에서 필터링
    """
    headers = naver_api_headers()
    rows = []

    start = 1
    safety_pages = 0
    max_start = 1000  # 안전장치

    while True:
        params = {
            "query": keyword,
            "display": per_page,
            "start": start,
            "sort": "date",
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
            pub_dt = parse_pubdate_to_dt(pub_raw)
            if pub_dt is None:
                continue

            # 기간 필터(로컬 날짜 기준)
            pub_local_date = pub_dt.astimezone().date()
            if not (start_d <= pub_local_date <= end_d):
                continue

            rows.append({
                "keyword": keyword,
                "pubDate": pub_dt.astimezone().strftime("%Y-%m-%d %H:%M"),
                "title": clean_html(it.get("title", "")),
                "description": clean_html(it.get("description", "")),
                "link": it.get("link", ""),
                "originallink": it.get("originallink", ""),
            })

        if len(rows) >= target_n:
            break

        start += per_page
        safety_pages += 1
        if start > max_start:
            break
        if safety_pages >= 12:  # 무한루프 방지
            break

        time.sleep(0.2)

    return pd.DataFrame(rows)


def build_report_html(df: pd.DataFrame, evidence: dict, start_d: date, end_d: date) -> str:
    """
    보고서 HTML 생성 (안전하게 문자열 합치기 방식)
    """
    # 근거 2문장 있는 것만
    valid_items = [(k, v) for k, v in evidence.items() if v.get("e1") and v.get("e2")]

    # 표 rows
    trs = []
    for idx, v in valid_items:
        frames = ", ".join(v.get("frame", []))
        tr = (
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{safe_text(v.get('pubDate',''))}</td>"
            f"<td>{safe_text(v.get('keyword',''))}</td>"
            f"<td>{safe_text(v.get('title',''))}</td>"
            f"<td>{safe_text(frames)}</td>"
            f"<td>{safe_text(v.get('level',''))}</td>"
            f"<td>{safe_text(v.get('e1',''))}</td>"
            f"<td>{safe_text(v.get('e2',''))}</td>"
            f"<td><a href=\"{safe_text(v.get('link',''))}\" target=\"_blank\">link</a></td>"
            "</tr>"
        )
        trs.append(tr)

    rows_html = "\n".join(trs)

    # 페이지 메타
    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    kws = ", ".join(sorted(set(df["keyword"].tolist()))) if not df.empty else ""
    n_articles = len(df)

    # CSS의 중괄호 때문에 f-string을 쓰지 않고 단순 문자열로 조립
    html = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'/>"
        "<title>언어와 매체 수행평가 보고서</title>"
        "<style>"
        "body{font-family:Arial, sans-serif; line-height:1.4; padding:18px;}"
        "table{border-collapse:collapse; width:100%;}"
        "th,td{border:1px solid #ccc; padding:8px; vertical-align:top;}"
        "th{background:#f2f2f2;}"
        "h1{margin-bottom:6px;}"
        ".meta{color:#555; margin:8px 0 16px 0;}"
        ".note{margin-top:14px; color:#333;}"
        "</style>"
        "</head><body>"
        "<h1>언어와 매체 수행평가 보고서</h1>"
        f"<div class='meta'>"
        f"생성 시각: {created}<br/>"
        f"입력 키워드: {safe_text(kws)}<br/>"
        f"기사 수집 기간: {start_d} ~ {end_d}<br/>"
        f"수집 기사 수: {n_articles}"
        f"</div>"
        "<h2>Claim–Evidence–Source 표</h2>"
        "<p>※ 각 항목은 학생이 입력한 ‘근거 문장’을 기반으로 구성됩니다.</p>"
        "<table>"
        "<thead><tr>"
        "<th>No</th><th>날짜</th><th>키워드</th><th>기사 제목</th>"
        "<th>프레임</th><th>근거 수준</th><th>근거 문장 1</th><th>근거 문장 2</th><th>출처</th>"
        "</tr></thead>"
        "<tbody>"
        f"{rows_html}"
        "</tbody>"
        "</table>"
        "<div class='note'><b>PDF 저장:</b> 이 HTML을 열고 브라우저 인쇄(Ctrl+P) → ‘PDF로 저장’</div>"
        "</body></html>"
    )
    return html


# -------------------------
# 사이드바 입력
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
# 실행
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
                frames.append(df_kw)
            except Exception as e:
                st.error(f"'{kw}' 수집 실패: {e}")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        st.error("수집된 기사가 없습니다. 기간/키워드를 바꿔보세요.")
        st.stop()

    df = dedup_articles(df)

    # 부족하면 첫 키워드로 추가 수집해서 채우기
    if len(df) < target_total:
        st.warning(f"현재 {len(df)}개만 수집됨 → 추가 수집 시도")
        remain = target_total - len(df)
        extra = fetch_news_one_keyword(keywords[0], start_d, end_d, remain + 30)
        df = pd.concat([df, extra], ignore_index=True)
        df = dedup_articles(df)

    st.success(f"최종 수집: {len(df)}개 (목표 {target_total})")
    # ✅ rerun돼도 데이터 유지(핵심)
    st.session_state["df"] = df
    st.session_state["start_d"] = start_d
    st.session_state["end_d"] = end_d
    st.session_state["data_ready"] = True

    
    # -------------------------
    # 탭 UI (①~④)
    # -------------------------
    tabs = st.tabs(["① 기사 목록", "② 통계 대시보드", "③ 근거 입력", "④ 보고서"])

    with tabs[0]:
        st.subheader("① 기사 목록")
        st.dataframe(df[["pubDate", "keyword", "title", "link"]], use_container_width=True)
        st.download_button(
            "CSV 다운로드(기사 목록)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="articles.csv",
            mime="text/csv",
        )

    with tabs[1]:
        st.subheader("② 통계 대시보드")

        df["date"] = df["pubDate"].str.slice(0, 10)

        by_date = df.groupby("date")["title"].count().reset_index(name="count")
        fig1 = px.line(by_date, x="date", y="count", markers=True, title="날짜별 기사량")
        st.plotly_chart(fig1, use_container_width=True)

        by_kw = (
            df.groupby("keyword")["title"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig2 = px.bar(by_kw, x="keyword", y="count", title="키워드별 기사량")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("③ 제목 강조어 빈도(간단)")
        hype_words = ["충격", "논란", "파장", "긴급", "폭로", "충돌", "경악", "비상", "전격"]
        hype_df = pd.DataFrame({
            "word": hype_words,
            "count": [int(df["title"].str.contains(w).sum()) for w in hype_words]
        }).sort_values("count", ascending=False)

        fig3 = px.bar(hype_df, x="word", y="count", title="강조/선정 표현 빈도(제목 기준)")
        st.plotly_chart(fig3, use_container_width=True)

    with tabs[2]:
        st.subheader("③ 근거 입력")
        st.write("기사별로 **근거 문장 2개** + **프레임**을 입력하고 저장하세요. (이게 있어야 보고서 생성 가능)")

        if "evidence" not in st.session_state:
            st.session_state.evidence = {}

        idx = st.number_input("기사 번호 선택(0부터)", min_value=0, max_value=len(df)-1, value=0, step=1)
        row = df.iloc[int(idx)]

        st.markdown(f"**제목:** {row['title']}")
        st.markdown(f"**키워드:** {row.get('keyword','')}")
        st.markdown(f"**날짜:** {row.get('pubDate','')}")
        st.markdown(f"**링크:** {row.get('link','')}")

        saved = st.session_state.evidence.get(int(idx), {})
        e1 = st.text_area("근거 문장 1(기사에서 그대로 복사)", value=saved.get("e1", ""), height=80)
        e2 = st.text_area("근거 문장 2(기사에서 그대로 복사)", value=saved.get("e2", ""), height=80)

        frame = st.multiselect(
            "프레임(복수 선택 가능)",
            ["갈등/대립", "책임 귀인", "경제/비용", "도덕/가치", "공포/위험", "해결/정책", "인물 중심", "데이터/연구 중심"],
            default=saved.get("frame", [])
        )

        levels = ["데이터/보고서 명시", "실명 전문가/기관 인용", "당사자 인터뷰", "익명 관계자", "추정/가능성 표현 위주"]
        level_saved = saved.get("level", levels[0])
        level_index = levels.index(level_saved) if level_saved in levels else 0

        evidence_level = st.selectbox("근거 수준", levels, index=level_index)

        if st.button("이 기사 입력 저장", type="primary"):
            st.session_state.evidence[int(idx)] = {
                "e1": e1.strip(),
                "e2": e2.strip(),
                "frame": frame,
                "level": evidence_level,
                "title": row.get("title", ""),
                "link": row.get("link", ""),
                "pubDate": row.get("pubDate", ""),
                "keyword": row.get("keyword", ""),
            }
            st.success("저장 완료!")

        st.divider()
        ev = st.session_state.evidence
        valid = [k for k, v in ev.items() if v.get("e1") and v.get("e2")]
        st.info(f"근거 2문장 입력 완료: {len(valid)}개 기사")

    with tabs[3]:
        st.subheader("④ 보고서")

        ev = st.session_state.get("evidence", {})
        valid_items = [(k, v) for k, v in ev.items() if v.get("e1") and v.get("e2")]

        min_required = 3
        st.write(f"근거 입력 완료 기사 수: **{len(valid_items)}개** / 필요: **{min_required}개**")

        if len(valid_items) < min_required:
            st.warning("③ 근거 입력에서 최소 3개 기사에 근거 문장 2개를 입력하고 저장하세요.")
            st.stop()

        html = build_report_html(df, ev, start_d, end_d)

        st.download_button(
            "HTML 보고서 다운로드",
            data=html.encode("utf-8"),
            file_name="report.html",
            mime="text/html",
        )
        st.info("PDF는 report.html을 열고 브라우저 인쇄(Ctrl+P) → ‘PDF로 저장’이 가장 안정적입니다.")
# ✅ 수집 버튼을 누르지 않아도, session_state에 df가 있으면 계속 보여주기
if st.session_state.get("data_ready") and "df" in st.session_state:
    df = st.session_state["df"]
    start_d = st.session_state["start_d"]
    end_d = st.session_state["end_d"]

    # -------------------------
    # 탭 UI (①~④)  ← 기존 탭 코드 통째로 여기로 옮겨도 되고,
    # 이미 if run 안에 있다면 "그 부분을 잘라서" 여기로 붙여넣으면 가장 깔끔합니다.
    # -------------------------
    tabs = st.tabs(["① 기사 목록", "② 통계 대시보드", "③ 근거 입력", "④ 보고서"])

    with tabs[0]:
        st.subheader("① 기사 목록")
        st.dataframe(df[["pubDate", "keyword", "title", "link"]], use_container_width=True)
        st.download_button(
            "CSV 다운로드(기사 목록)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="articles.csv",
            mime="text/csv",
        )

    with tabs[1]:
        st.subheader("② 통계 대시보드")
        df["date"] = df["pubDate"].str.slice(0, 10)
        by_date = df.groupby("date")["title"].count().reset_index(name="count")
        st.plotly_chart(px.line(by_date, x="date", y="count", markers=True, title="날짜별 기사량"), use_container_width=True)

        by_kw = df.groupby("keyword")["title"].count().reset_index(name="count").sort_values("count", ascending=False)
        st.plotly_chart(px.bar(by_kw, x="keyword", y="count", title="키워드별 기사량"), use_container_width=True)

    with tabs[2]:
        st.subheader("③ 근거 입력")

        if "evidence" not in st.session_state:
            st.session_state.evidence = {}

        idx = st.number_input("기사 번호 선택(0부터)", min_value=0, max_value=len(df)-1, value=0, step=1)
        row = df.iloc[int(idx)]

        st.markdown(f"**제목:** {row['title']}")
        st.markdown(f"**키워드:** {row.get('keyword','')}")
        st.markdown(f"**날짜:** {row.get('pubDate','')}")
        st.markdown(f"**링크:** {row.get('link','')}")

        saved = st.session_state.evidence.get(int(idx), {})
        e1 = st.text_area("근거 문장 1(기사에서 그대로 복사)", value=saved.get("e1", ""), height=80)
        e2 = st.text_area("근거 문장 2(기사에서 그대로 복사)", value=saved.get("e2", ""), height=80)

        frame = st.multiselect(
            "프레임(복수 선택 가능)",
            ["갈등/대립", "책임 귀인", "경제/비용", "도덕/가치", "공포/위험", "해결/정책", "인물 중심", "데이터/연구 중심"],
            default=saved.get("frame", [])
        )

        levels = ["데이터/보고서 명시", "실명 전문가/기관 인용", "당사자 인터뷰", "익명 관계자", "추정/가능성 표현 위주"]
        level_saved = saved.get("level", levels[0])
        level_index = levels.index(level_saved) if level_saved in levels else 0
        evidence_level = st.selectbox("근거 수준", levels, index=level_index)

        if st.button("이 기사 입력 저장", type="primary"):
            st.session_state.evidence[int(idx)] = {
                "e1": e1.strip(),
                "e2": e2.strip(),
                "frame": frame,
                "level": evidence_level,
                "title": row.get("title", ""),
                "link": row.get("link", ""),
                "pubDate": row.get("pubDate", ""),
                "keyword": row.get("keyword", ""),
            }
            st.success("저장 완료!")

        valid = [k for k, v in st.session_state.evidence.items() if v.get("e1") and v.get("e2")]
        st.info(f"근거 2문장 입력 완료: {len(valid)}개 기사")

    with tabs[3]:
        st.subheader("④ 보고서")
        ev = st.session_state.get("evidence", {})
        valid_items = [(k, v) for k, v in ev.items() if v.get("e1") and v.get("e2")]

        min_required = 3
        st.write(f"근거 입력 완료 기사 수: **{len(valid_items)}개** / 필요: **{min_required}개**")

        if len(valid_items) < min_required:
            st.warning("③ 근거 입력에서 최소 3개 기사에 근거 문장 2개를 입력하고 저장하세요.")
        else:
            html = build_report_html(df, ev, start_d, end_d)
            st.download_button(
                "HTML 보고서 다운로드",
                data=html.encode("utf-8"),
                file_name="report.html",
                mime="text/html",
            )
            st.info("PDF는 report.html을 열고 브라우저 인쇄(Ctrl+P) → ‘PDF로 저장’이 가장 안정적입니다.")
else:
    st.caption("왼쪽에서 기간/키워드 입력 → ‘수집 시작’을 누르세요.")

