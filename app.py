import math
import time
import re
from datetime import datetime, date, timedelta

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# -------------------------
# 설정
# -------------------------
NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

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
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def parse_pubdate_to_dt(pub_raw: str):
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
    return {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}

def get_openai_key_and_model():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    return api_key, model

def dedup_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "link" in df.columns:
        df = df.drop_duplicates(subset=["link"])
    if {"title", "pubDate"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["title", "pubDate"])
    return df.reset_index(drop=True)

# -------------------------
# API 수집
# -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_one_keyword(keyword: str, start_d: date, end_d: date, target_n: int, per_page: int = 100) -> pd.DataFrame:
    headers = naver_api_headers()
    rows = []
    start = 1
    safety_pages = 0
    max_start = 1000  # 안전장치

    while True:
        params = {"query": keyword, "display": per_page, "start": start, "sort": "date"}
        r = requests.get(NAVER_NEWS_URL, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"네이버 API 오류: {r.status_code} / {r.text}")

        data = r.json()
        items = data.get("items", [])
        if not items:
            break

        for it in items:
            pub_dt = parse_pubdate_to_dt(it.get("pubDate", ""))
            if pub_dt is None:
                continue

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
        if start > max_start or safety_pages >= 12:
            break
        time.sleep(0.2)

    return pd.DataFrame(rows)

# -------------------------
# OpenAI 분석 (대시보드 해석) - Responses API
# -------------------------
def _extract_responses_text(data: dict) -> str:
    """
    Responses API 응답에서 텍스트를 최대한 안전하게 추출
    """
    # 문서에 따라 output_text가 제공되는 경우가 있음
    if isinstance(data, dict) and data.get("output_text"):
        return str(data["output_text"]).strip()

    out_chunks = []
    for item in data.get("output", []) if isinstance(data, dict) else []:
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_chunks.append(c.get("text", ""))
            # 일부 형식에서는 type이 text일 수도 있어 방어
            if c.get("type") == "text":
                out_chunks.append(c.get("text", ""))

    text = "".join(out_chunks).strip()
    return text

def openai_analyze_dashboard(stats_text: str) -> str:
    api_key, model = get_openai_key_and_model()
    if not api_key:
        return "OPENAI_API_KEY가 설정되지 않았습니다. (Secrets 확인)"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = f"""
너는 고3 ‘언어와 매체’ 수행평가 조교다.

규칙(매우 중요):
- 아래 <통계 요약>에 있는 숫자/사실만 사용한다.
- 통계에 없는 내용(추정, 일반론, 외부지식)은 금지.
- 각 주장 문장 끝에 반드시 (근거: 통계 요약의 어떤 항목인지) 한 줄로 표기한다.

형식(반드시 지켜라):
[1] 핵심 관찰(3~5개) : 각 문장에 수치 1개 이상 포함
[2] 프레임 해석(2~3개) : 책임귀인/갈등/경제/해결/공포/데이터 중 무엇이 보이는지 + 수치 근거
[3] 추가 탐구 질문(3개) : 기사 본문 확인이 필요한 질문만

<통계 요약>
{stats_text}
""".strip()

    payload = {
        "model": model,
        # 최신 권장: Responses API input
        "input": prompt,
        "max_output_tokens": 900,
    }

    last_err = None
    for _ in range(2):  # 2회 재시도
        try:
            resp = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=90)

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text}"
                time.sleep(1)
                continue

            data = resp.json()
            text = _extract_responses_text(data)

            if not text:
                # 진단용 일부 필드 노출
                return f"OpenAI 응답 텍스트가 비었습니다. raw_keys={list(data.keys())}"

            if len(text) < 120:
                text += "\n\n⚠️ 응답이 매우 짧습니다. (모델 권한/쿼터/필터/네트워크 문제 가능)"
            return text

        except Exception as e:
            last_err = repr(e)
            time.sleep(1)

    return f"OpenAI 호출 실패: {last_err}"

# -------------------------
# 보고서 HTML 생성
# -------------------------
def build_report_html(df: pd.DataFrame, evidence: dict, start_d: date, end_d: date,
                      student_id: str, student_name: str, reflection: str,
                      dashboard_ai_summary: str) -> str:
    valid_items = [(k, v) for k, v in evidence.items() if v.get("e1") and v.get("e2")]

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

    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    kws = ", ".join(sorted(set(df["keyword"].tolist()))) if not df.empty else ""
    n_articles = len(df)

    html = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'/>"
        "<title>언어와 매체 수행평가 보고서</title>"
        "<style>"
        "body{font-family:Arial, sans-serif; line-height:1.5; padding:18px;}"
        "table{border-collapse:collapse; width:100%;}"
        "th,td{border:1px solid #ccc; padding:8px; vertical-align:top;}"
        "th{background:#f2f2f2;}"
        "h1{margin-bottom:6px;}"
        ".meta{color:#555; margin:8px 0 16px 0;}"
        ".box{border:1px solid #ddd; padding:12px; background:#fafafa; white-space:pre-wrap;}"
        ".note{margin-top:14px; color:#333;}"
        "</style>"
        "</head><body>"
        "<h1>언어와 매체 수행평가 보고서</h1>"
        f"<div class='meta'>"
        f"<b>학번</b>: {safe_text(student_id)} &nbsp;&nbsp; <b>성명</b>: {safe_text(student_name)}<br/>"
        f"생성 시각: {created}<br/>"
        f"입력 키워드: {safe_text(kws)}<br/>"
        f"기사 수집 기간: {start_d} ~ {end_d}<br/>"
        f"수집 기사 수: {n_articles}"
        f"</div>"

        "<h2>통계 대시보드 해석(AI)</h2>"
        f"<div class='box'>{safe_text(dashboard_ai_summary)}</div>"

        "<h2>개인 생각(소감/비판적 관점)</h2>"
        f"<div class='box'>{safe_text(reflection)}</div>"

        "<h2 style='margin-top:18px;'>Claim–Evidence–Source 표</h2>"
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
# 수집 실행
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

    if len(df) < target_total:
        st.warning(f"현재 {len(df)}개만 수집됨 → 추가 수집 시도")
        remain = target_total - len(df)
        extra = fetch_news_one_keyword(keywords[0], start_d, end_d, remain + 30)
        df = pd.concat([df, extra], ignore_index=True)
        df = dedup_articles(df)

    st.success(f"최종 수집: {len(df)}개 (목표 {target_total})")

    # ✅ rerun돼도 유지
    st.session_state["df"] = df
    st.session_state["start_d"] = start_d
    st.session_state["end_d"] = end_d
    st.session_state["data_ready"] = True

# -------------------------
# 메인 표시(세션에 데이터 있으면 계속 유지)
# -------------------------
if st.session_state.get("data_ready") and "df" in st.session_state:
    df = st.session_state["df"]
    start_d = st.session_state["start_d"]
    end_d = st.session_state["end_d"]

    tabs = st.tabs(["① 기사 목록", "② 통계 대시보드", "③ 근거 입력", "④ 보고서"])

    # ① 기사 목록
    with tabs[0]:
        st.subheader("① 기사 목록")
        st.dataframe(df[["pubDate", "keyword", "title", "link"]], use_container_width=True)
        st.download_button(
            "CSV 다운로드(기사 목록)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="articles.csv",
            mime="text/csv",
        )

    # ② 통계 + OpenAI 해석
    with tabs[1]:
        st.subheader("② 통계 대시보드")

        df_local = df.copy()
        df_local["date"] = df_local["pubDate"].str.slice(0, 10)

        by_date = df_local.groupby("date")["title"].count().reset_index(name="count")
        st.plotly_chart(px.line(by_date, x="date", y="count", markers=True, title="날짜별 기사량"), use_container_width=True)

        by_kw = df_local.groupby("keyword")["title"].count().reset_index(name="count").sort_values("count", ascending=False)
        st.plotly_chart(px.bar(by_kw, x="keyword", y="count", title="키워드별 기사량"), use_container_width=True)

        st.subheader("③ 제목 강조어 빈도(간단)")
        hype_words = ["충격", "논란", "파장", "긴급", "폭로", "충돌", "경악", "비상", "전격"]
        hype_df = pd.DataFrame({
            "word": hype_words,
            "count": [int(df_local["title"].str.contains(w).sum()) for w in hype_words]
        }).sort_values("count", ascending=False)
        st.plotly_chart(px.bar(hype_df, x="word", y="count", title="강조/선정 표현 빈도(제목 기준)"), use_container_width=True)

        st.divider()
        st.subheader("④ (업그레이드) OpenAI로 통계 해석 생성")

        top_kw = by_kw.head(10).to_dict("records")
        peak = by_date.sort_values("count", ascending=False).head(1).to_dict("records")
        hype_top = hype_df.head(6).to_dict("records")

        stats_text = (
            f"- 기간: {start_d} ~ {end_d}\n"
            f"- 수집 기사 수: {len(df_local)}\n"
            f"- 키워드별 기사량(상위): {top_kw}\n"
            f"- 날짜별 기사량(피크): {peak}\n"
            f"- 제목 강조어 빈도(상위): {hype_top}\n"
        )

        with st.expander("AI에게 전달되는 통계 요약(검증용)"):
            st.code(stats_text)

        if "dashboard_ai" not in st.session_state:
            st.session_state["dashboard_ai"] = ""
        if "dashboard_ai_err" not in st.session_state:
            st.session_state["dashboard_ai_err"] = ""

        if st.button("OpenAI로 통계 해석 생성", type="primary"):
            st.session_state["dashboard_ai"] = ""
            st.session_state["dashboard_ai_err"] = ""
            with st.spinner("OpenAI가 통계 해석을 작성 중..."):
                try:
                    st.session_state["dashboard_ai"] = openai_analyze_dashboard(stats_text)
                except Exception as e:
                    st.session_state["dashboard_ai_err"] = f"OpenAI 분석 중 예외: {repr(e)}"

        if st.session_state.get("dashboard_ai_err"):
            st.error(st.session_state["dashboard_ai_err"])

        if st.session_state.get("dashboard_ai"):
            st.text_area("AI 해석 결과(전체)", value=st.session_state["dashboard_ai"], height=420)

    # ③ 근거 입력
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

        valid = [k for k, v in st.session_state.evidence.items() if v.get("e1") and v.get("e2")]
        st.info(f"근거 2문장 입력 완료: {len(valid)}개 기사")

    # ④ 보고서
    with tabs[3]:
        st.subheader("④ 보고서")

        if "student_id" not in st.session_state:
            st.session_state["student_id"] = ""
        if "student_name" not in st.session_state:
            st.session_state["student_name"] = ""
        if "reflection" not in st.session_state:
            st.session_state["reflection"] = ""

        col1, col2 = st.columns(2)
        with col1:
            st.session_state["student_id"] = st.text_input("학번", value=st.session_state["student_id"])
        with col2:
            st.session_state["student_name"] = st.text_input("성명", value=st.session_state["student_name"])

        st.session_state["reflection"] = st.text_area(
            "개인 생각(소감/비판적 관점) — 통계+근거문장에 기반해 작성",
            value=st.session_state["reflection"],
            height=160
        )

        ev = st.session_state.get("evidence", {})
        valid_items = [(k, v) for k, v in ev.items() if v.get("e1") and v.get("e2")]

        min_required = 3
        st.write(f"근거 입력 완료 기사 수: **{len(valid_items)}개** / 필요: **{min_required}개**")

        ok_evidence = len(valid_items) >= min_required
        ok_student = bool(st.session_state["student_id"].strip()) and bool(st.session_state["student_name"].strip())
        ok_reflection = bool(st.session_state["reflection"].strip())
        ok_ai = bool(st.session_state.get("dashboard_ai", "").strip())

        if not ok_student:
            st.warning("학번/성명을 입력하세요.")
        if not ok_reflection:
            st.warning("개인 생각(소감)을 입력하세요.")
        if not ok_evidence:
            st.warning("③ 근거 입력에서 최소 3개 기사에 근거 문장 2개를 입력하고 저장하세요.")
        if not ok_ai:
            st.warning("② 통계 대시보드에서 ‘OpenAI 통계 해석’을 생성하면 보고서 완성도가 올라갑니다. (선택이지만 권장)")

        can_make = ok_student and ok_reflection and ok_evidence

        if can_make:
            html = build_report_html(
                df=df,
                evidence=ev,
                start_d=start_d,
                end_d=end_d,
                student_id=st.session_state["student_id"],
                student_name=st.session_state["student_name"],
                reflection=st.session_state["reflection"],
                dashboard_ai_summary=st.session_state.get("dashboard_ai", "")
            )

            st.subheader("보고서 미리보기")
            components.html(html, height=520, scrolling=True)

            st.download_button(
                "HTML 보고서 다운로드(조건 충족)",
                data=html.encode("utf-8"),
                file_name="report.html",
                mime="text/html",
            )
            st.info("PDF는 report.html을 열고 브라우저 인쇄(Ctrl+P) → ‘PDF로 저장’이 가장 안정적입니다.")
        else:
            st.info("위 조건을 모두 채우면 ‘미리보기’와 ‘다운로드’가 활성화됩니다.")
else:
    st.caption("왼쪽에서 기간/키워드 입력 → ‘수집 시작’을 누르세요.")
