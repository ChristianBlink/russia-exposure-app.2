
import os
import io
import re
import json
import time
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Tuple

import requests
import pandas as pd
from rapidfuzz import fuzz
from bs4 import BeautifulSoup
import streamlit as st

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# PDF text extraction
from pdfminer.high_level import extract_text as pdf_extract_text

st.set_page_config(page_title="Russia Exposure Reporter", page_icon="🛰️", layout="wide")

# --------------- Helpers ---------------

HEADERS = {"User-Agent": os.getenv("SEC_EMAIL","RussiaExposureApp/1.6 (+contact: user)")}

KEYWORDS = [
    r"russia", r"russian federation", r"moscow", r"saint petersburg", r"st[.\s-]?petersburg",
    r"belarus", r"cis region", r"ukraine", r"sanction", r"export control", r"ofac", r"eu sanction",
    r"withdraw(al|n)?", r"exit(ed|ing)?", r"suspend(ed|ing)?", r"continue(s|d)?"
]

def kw_regex():
    return re.compile("|".join(KEYWORDS), re.IGNORECASE)

def find_snippets(text: str, kw_re=None, window: int = 120, max_snips: int = 3) -> Tuple[int, List[str]]:
    if not text:
        return 0, []
    if kw_re is None:
        kw_re = kw_regex()
    hits = []
    it = kw_re.finditer(text)
    for m in it:
        start = max(0, m.start()-window)
        end = min(len(text), m.end()+window)
        snippet = text[start:end].replace("\n"," ").replace("  "," ").strip()
        hits.append(snippet)
        if len(hits) >= max_snips:
            break
    # Count all matches (iterate again to count fully)
    total = len(list(kw_re.finditer(text)))
    return total, hits

@st.cache_data(ttl=60*60)
def fetch_gdelt(company: str, years: int = 5, maxrecords: int = 120) -> List[Dict[str, Any]]:
    q = f'("{company}" AND (Russia OR Russian OR Moscow OR "St Petersburg" OR "Saint Petersburg" OR Belarus))'
    params = {"query": q, "mode": "ArtList", "format": "json", "timespan": f"{years}Y", "maxrecords": str(maxrecords), "sort": "DateDesc"}
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("articles", [])

@st.cache_data(ttl=60*60*6)
def fetch_ofac_sdn() -> pd.DataFrame:
    try:
        url = "https://www.treasury.gov/ofac/downloads/sdn.csv"
        df = pd.read_csv(url, dtype=str, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        return df.fillna("")
    except Exception as e:
        st.warning(f"Could not load OFAC SDN: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60*60*6)
def fetch_eu_sanctions() -> pd.DataFrame:
    try:
        url = "https://webgate.ec.europa.eu/fsd/fsf/public/files/json/cfsp/sanctionslist.json"
        r = requests.get(url, headers=HEADERS, timeout=45)
        r.raise_for_status()
        j = r.json()
        rows = []
        for e in j.get("sanctions", []):
            name = e.get("name", "")
            program = e.get("regulation", {}).get("programme", "")
            country = e.get("nameAlias", [{}])[0].get("country", "")
            rows.append({"name": name, "program": program, "country": country})
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Could not load EU sanctions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60*60*6)
def fetch_uk_hmt() -> pd.DataFrame:
    try:
        url = "https://ofsistorage.blob.core.windows.net/publishlive/ConList.csv"
        df = pd.read_csv(url, dtype=str).fillna("")
        name_col = None
        for c in df.columns:
            if c.lower() in ["name", "names"]:
                name_col = c; break
        if name_col is None:
            possible = [c for c in df.columns if "name" in c.lower()]
            name_col = possible[0] if possible else df.columns[0]
        df.rename(columns={name_col: "Name"}, inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not load UK HMT/OFSI list: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60*60*6)
def fetch_un_consolidated() -> pd.DataFrame:
    try:
        url = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        from lxml import etree
        root = etree.fromstring(r.content)
        ns = {"x": root.nsmap.get(None) or ""}
        rows = []
        for ind in root.findall(".//x:INDIVIDUAL", namespaces=ns):
            primary = ind.findtext(".//x:NAME_ORIGINAL_SCRIPT", namespaces=ns) or ""
            first = ind.findtext(".//x:FIRST_NAME", namespaces=ns) or ""
            last = ind.findtext(".//x:SECOND_NAME", namespaces=ns) or ""
            third = ind.findtext(".//x:THIRD_NAME", namespaces=ns) or ""
            whole = " ".join([first, last, third]).strip()
            nm = primary or whole
            if nm:
                rows.append({"name": nm, "type": "INDIVIDUAL"})
        for ent in root.findall(".//x:ENTITY", namespaces=ns):
            nm = ent.findtext(".//x:NAME", namespaces=ns) or ""
            if nm:
                rows.append({"name": nm, "type": "ENTITY"})
        return pd.DataFrame(rows).fillna("")
    except Exception as e:
        st.warning(f"Could not load UN Consolidated List: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=0)
def scrape_leave_russia() -> pd.DataFrame:
    url = "https://leave-russia.org"
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []
    tables = soup.find_all(["table"])
    for tbl in tables:
        for tr in tbl.find_all("tr"):
            tds = tr.find_all(["td","th"])
            if len(tds) < 2:
                continue
            text_cells = [td.get_text(strip=True) for td in tds]
            name = text_cells[0]
            cat = None
            for cell in text_cells[1:]:
                if any(k in cell.lower() for k in ["exit", "suspend", "continue", "withdraw", "remain", "partial"]):
                    cat = cell; break
            desc = " ".join(text_cells[1:]) if len(text_cells) > 1 else ""
            link = None
            a = tr.find("a")
            if a and a.get("href"):
                link = urllib.parse.urljoin(url, a["href"])
            if name and len(name) > 1:
                rows.append({"name": name, "category": cat or "", "description": desc, "href": link or url})

    cards = soup.find_all(["article","div","li"], class_=lambda c: c and any(k in c.lower() for k in ["card","entry","company","item","list"]))
    for card in cards:
        name_el = card.find(["h2","h3","strong","b","a"])
        if not name_el: continue
        name = name_el.get_text(strip=True)
        desc_el = card.find("p")
        desc = desc_el.get_text(strip=True) if desc_el else ""
        cat = ""
        for span in card.find_all("span"):
            s = span.get_text(strip=True).lower()
            if any(k in s for k in ["exit","suspend","continue","withdraw","remain","partial"]):
                cat = span.get_text(strip=True); break
        link = None
        a = card.find("a")
        if a and a.get("href"):
            link = urllib.parse.urljoin(url, a["href"])
        if name and len(name) > 1:
            rows.append({"name": name, "category": cat, "description": desc, "href": link or url})

    df = pd.DataFrame(rows).drop_duplicates()
    if not df.empty:
        df["name"] = df["name"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()
        df["description"] = df["description"].astype(str).str.strip()
        df["href"] = df["href"].astype(str).str.strip()
    return df

@st.cache_data(ttl=60*60)
def resolve_lei(lei: str) -> Dict[str, Any]:
    try:
        lei = lei.strip().upper()
        url = f"https://api.gleif.org/api/v1/lei-records/{lei}"
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return {}
        j = r.json()
        data = j.get("data") or {}
        attrs = data.get("attributes") or {}
        ent = attrs.get("entity") or {}
        legal = ent.get("legalName", {}).get("name", "")
        other_names = [n.get("name") for n in (ent.get("otherNames") or []) if n.get("name")]
        return {"legal_name": legal, "aliases": other_names}
    except Exception:
        return {}

@st.cache_data(ttl=60*60)
def opensanctions_match(name: str = "", lei: str = "", isin: str = "") -> List[Dict[str, Any]]:
    matches = []
    try:
        url = "https://api.opensanctions.org/match"
        q = {"queries": [{"name": name or "", "identifiers": {}}], "datasets": ["sanctions"]}
        if lei: q["queries"][0]["identifiers"]["lei"] = [lei]
        if isin: q["queries"][0]["identifiers"]["isin"] = [isin]
        r = requests.post(url, headers={"Accept":"application/json","User-Agent":HEADERS["User-Agent"]}, json=q, timeout=45)
        if r.status_code == 200:
            j = r.json()
            for res in j.get("responses", []):
                for m in res.get("matches", []):
                    ent = m.get("entity", {})
                    matches.append({
                        "score": m.get("score", 0),
                        "name": ent.get("caption") or ent.get("name") or "",
                        "id": ent.get("id",""),
                        "datasets": ent.get("datasets", []),
                        "schema": ent.get("schema", ""),
                    })
    except Exception:
        pass
    if not matches and (name or lei or isin):
        try:
            qs = urllib.parse.urlencode({"q": name or lei or isin})
            url = f"https://api.opensanctions.org/search?{qs}"
            r = requests.get(url, headers={"Accept":"application/json","User-Agent":HEADERS["User-Agent"]}, timeout=45)
            if r.status_code == 200:
                j = r.json()
                for m in j.get("results", []):
                    matches.append({
                        "score": m.get("score", 0),
                        "name": m.get("caption") or "",
                        "id": m.get("id",""),
                        "datasets": m.get("datasets", []),
                        "schema": m.get("schema", ""),
                    })
        except Exception:
            pass
    return matches

def normalize(s: str) -> str:
    return (s or "").strip().lower()

def exact_or_fuzzy_hits(target: str, names: List[str], topn: int = 5) -> List[Tuple[str, float, str]]:
    t = normalize(target)
    exact = [(n, 100.0, "exact") for n in names if normalize(n) == t]
    if exact:
        return exact[:topn]
    scored = [(n, fuzz.QRatio(t, normalize(n)), "fuzzy") for n in names]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topn]

# ---------------- SEC & IR scanning ----------------

@st.cache_data(ttl=60*60)
def sec_ticker_map() -> pd.DataFrame:
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=HEADERS, timeout=45)
        r.raise_for_status()
        j = r.json()
        rows = []
        for _, v in j.items():
            rows.append({"ticker": v.get("ticker",""), "cik": str(v.get("cik_str","")).zfill(10), "title": v.get("title","")})
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Could not load SEC ticker map: {e}")
        return pd.DataFrame(columns=["ticker","cik","title"])

def resolve_cik(ticker: str, company: str, ticker_df: pd.DataFrame) -> Tuple[str, str]:
    ticker = (ticker or "").strip().upper()
    if ticker and not ticker_df.empty:
        row = ticker_df[ticker_df["ticker"] == ticker]
        if not row.empty:
            return row.iloc[0]["cik"], row.iloc[0]["title"]
    if company and not ticker_df.empty:
        cand = exact_or_fuzzy_hits(company, ticker_df["title"].tolist(), topn=1)
        if cand and cand[0][1] >= 90:
            name = cand[0][0]
            row = ticker_df[ticker_df["title"] == name].iloc[0]
            return row["cik"], row["title"]
    return "", ""

def fetch_sec_filings(cik: str, forms: List[str] = ["10-K","10-Q","20-F","6-K"], max_docs: int = 6) -> List[Dict[str,str]]:
    if not cik:
        return []
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=HEADERS, timeout=45); r.raise_for_status()
        j = r.json()
        filings = j.get("filings", {}).get("recent", {})
        res = []
        for form, acc, doc, date in zip(filings.get("form", []),
                                        filings.get("accessionNumber", []),
                                        filings.get("primaryDocument", []),
                                        filings.get("filingDate", [])):
            if form not in forms: continue
            acc_nodash = acc.replace("-","")
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
            res.append({"form": form, "date": date, "url": base, "accession": acc, "doc": doc})
            if len(res) >= max_docs:
                break
        return res
    except Exception:
        return []

def fetch_url_text(url: str, max_bytes: int = 6_000_000) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type","").lower()
        data = r.content[:max_bytes]
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                return pdf_extract_text(io.BytesIO(data))
            except Exception:
                return ""
        else:
            soup = BeautifulSoup(data, "lxml")
            for s in soup(["script","style","noscript"]):
                s.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

def scan_sec_and_ir(cik: str, ticker: str, company: str, ir_url: str) -> Dict[str, Any]:
    results = {"sec": [], "ir": []}
    kw_re = kw_regex()

    filings = fetch_sec_filings(cik) if cik else []
    for f in filings:
        text = fetch_url_text(f["url"])
        count, snips = find_snippets(text, kw_re=kw_re, window=150, max_snips=3)
        f_out = f.copy()
        f_out.update({"hits": count, "snippets": snips})
        results["sec"].append(f_out)

    if ir_url:
        try:
            page_text = fetch_url_text(ir_url)
            count, snips = find_snippets(page_text, kw_re=kw_re, window=150, max_snips=3)
            results["ir"].append({"type": "IR page", "url": ir_url, "hits": count, "snippets": snips})
            r = requests.get(ir_url, headers=HEADERS, timeout=45)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            links = []
            for a in soup.find_all("a", href=True):
                href = urllib.parse.urljoin(ir_url, a["href"])
                txt = (a.get_text(strip=True) or "").lower()
                if any(k in txt for k in ["annual", "report", "esg", "sustainab", "press", "release"]) or href.lower().endswith(".pdf"):
                    links.append(href)
            seen = set(); dedup = []
            for h in links:
                if h in seen: continue
                seen.add(h); dedup.append(h)
            for h in dedup[:3]:
                t = fetch_url_text(h)
                c, s = find_snippets(t, kw_re=kw_re, window=150, max_snips=3)
                results["ir"].append({"type": "IR linked doc", "url": h, "hits": c, "snippets": s})
        except Exception:
            pass
    return results

# ---------------- Core classification from news ----------------

def keyword_flags(text: str) -> Dict[str, int]:
    patterns = {
        "exit": r"\b(exit(ed|ing)?|withdraw(n|al)?|left|pull(ed)?\s+out|divest(ed|ment)?)\b",
        "suspend": r"\b(suspend(ed|ing)?|halt(ed|ing)?|pause(d)?)\b",
        "resume": r"\b(resume(d|s)?|reopen(ed)?|restart(ed)?)\b",
        "continue": r"\b(continue(s|d)?|remain(s|ed)?)\b",
        "sanction": r"\b(sanction(s|ed|ing)?)\b",
        "contract": r"\b(contract(s|ed)?|deal(s)?)\b",
        "office": r"\b(office|store|factory|plant|subsidiar(y|ies))\b",
        "revenue": r"\b(revenue|sales|%|percent|exposure)\b",
    }
    flags = {k: 0 for k in patterns}
    for k, p in patterns.items():
        matches = re.findall(p, text, flags=re.IGNORECASE)
        flags[k] = len(matches)
    return flags

def classify_exposure(articles_df: pd.DataFrame) -> Dict[str, Any]:
    if articles_df.empty:
        return {"label": "No signal", "confidence": 0.2, "rationale": "No relevant articles found."}
    joined = " ".join((articles_df["title"].fillna("") + " " + articles_df["seendate"].fillna("") + " " + articles_df["url"].fillna("")).tolist())
    flags = keyword_flags(joined)
    rationale_bits = [f"{k}×{v}" for k,v in flags.items() if v]
    rationale = ", ".join(rationale_bits) if rationale_bits else "No strong keywords."
    if flags["exit"] >= 2 and flags["resume"] == 0 and flags["continue"] <= 1:
        return {"label": "Exited Russia", "confidence": 0.7, "rationale": rationale}
    if flags["suspend"] >= 1 and flags["resume"] == 0:
        return {"label": "Suspended operations", "confidence": 0.6, "rationale": rationale}
    if flags["continue"] >= 2 and flags["exit"] == 0:
        return {"label": "Continues operations", "confidence": 0.6, "rationale": rationale}
    return {"label": "Inconclusive", "confidence": 0.45, "rationale": rationale}

def build_report(company: str,
                 match_name: str,
                 articles: List[Dict[str, Any]],
                 ofac_df: pd.DataFrame,
                 eu_df: pd.DataFrame,
                 uk_df: pd.DataFrame,
                 un_df: pd.DataFrame,
                 leave_df: pd.DataFrame,
                 opensanctions: List[Dict[str, Any]],
                 filings_scan: Dict[str, Any],
                 meta: Dict[str, Any]) -> Dict[str, Any]:

    art_df = pd.DataFrame(articles)
    if not art_df.empty:
        cols = ["title","seendate","url","sourceCountry","language","domain","socialimage"]
        for c in cols:
            if c not in art_df.columns:
                art_df[c] = ""
        art_df = art_df[cols].copy()
    else:
        art_df = pd.DataFrame(columns=["title","seendate","url","sourceCountry","language","domain","socialimage"])

    sanctions_hits, os_hits = [], []
    name_for_matching = match_name or company

    # OFAC/EU/UK/UN
    if not ofac_df.empty and "SDN_Name" in ofac_df.columns:
        for name, score, method in exact_or_fuzzy_hits(name_for_matching, ofac_df["SDN_Name"].astype(str).tolist(), topn=5):
            if score >= 90:
                sanctions_hits.append({"list": "OFAC SDN", "name": name, "score": score, "method": method})
    if not eu_df.empty and "name" in eu_df.columns:
        for name, score, method in exact_or_fuzzy_hits(name_for_matching, eu_df["name"].astype(str).tolist(), topn=5):
            if score >= 90:
                sanctions_hits.append({"list": "EU CFSP", "name": name, "score": score, "method": method})
    if not uk_df.empty:
        series = uk_df["Name"] if "Name" in uk_df.columns else uk_df.iloc[:,0]
        for name, score, method in exact_or_fuzzy_hits(name_for_matching, series.astype(str).tolist(), topn=5):
            if score >= 90:
                sanctions_hits.append({"list": "UK HMT (OFSI)", "name": name, "score": score, "method": method})
    if not un_df.empty and "name" in un_df.columns:
        for name, score, method in exact_or_fuzzy_hits(name_for_matching, un_df["name"].astype(str).tolist(), topn=5):
            if score >= 90:
                sanctions_hits.append({"list": "UN Consolidated", "name": name, "score": score, "method": method})

    # OpenSanctions
    for m in opensanctions or []:
        dsets = m.get("datasets") or []
        if any("sanctions" in ds for ds in dsets):
            os_hits.append({
                "list": "OpenSanctions",
                "name": m.get("name",""),
                "score": float(m.get("score", 0))*100 if m.get("score",0) <= 1 else float(m.get("score",0)),
                "method": "api",
                "datasets": ", ".join(dsets),
                "id": m.get("id","")
            })
    sanctions_hits.extend(os_hits)

    # Leave-Russia.org
    leave_hits = []
    if not leave_df.empty:
        candidates = exact_or_fuzzy_hits(name_for_matching, leave_df["name"].astype(str).tolist(), topn=5)
        for nm, sc, method in candidates:
            row = leave_df[leave_df["name"] == nm].iloc[0]
            leave_hits.append({
                "name": nm, "score": sc, "method": method,
                "category": row.get("category",""), "description": row.get("description",""),
                "url": row.get("href","https://leave-russia.org")
            })

    exposure = classify_exposure(art_df)

    # Risk score (1 low, 5 high)
    risk = 1
    if sanctions_hits:
        risk = 5
    else:
        if leave_hits:
            cat_text = " ".join([str(h.get("category","")).lower() for h in leave_hits])
            if any(k in cat_text for k in ["continue", "remain", "operat"]):
                risk = 4
            elif any(k in cat_text for k in ["suspend", "partial"]):
                risk = 3
            elif any(k in cat_text for k in ["exit","withdraw","left","pulled out"]):
                risk = 2
            else:
                risk = 2
        else:
            if exposure["label"] == "Continues operations":
                risk = 4
            elif exposure["label"] == "Suspended operations":
                risk = 3
            elif exposure["label"] == "Exited Russia":
                risk = 2
            else:
                risk = 2 if len(art_df) > 0 else 1

    return {
        "company_input": company,
        "match_name_used": name_for_matching,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "risk_score_1_to_5": risk,
        "sanctions_hits": sanctions_hits,
        "opensanctions_hits": os_hits,
        "leave_russia_hits": leave_hits,
        "exposure_classification": exposure,
        "articles_df": art_df,
        "entity_meta": meta,
        "filings_scan": filings_scan,
    }

def render_pdf_summary(rep: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x_margin = 2 * cm
    y = height - 2 * cm

    def line(text, size=11, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        for chunk in textwrap(text, 95 if size<=11 else 80):
            c.drawString(x_margin, y, chunk)
            y -= leading

    def textwrap(s, width):
        words = s.split()
        out, cur, ln = [], [], 0
        for w in words:
            ln += len(w) + 1
            cur.append(w)
            if ln > width:
                out.append(" ".join(cur))
                cur, ln = [], 0
        if cur:
            out.append(" ".join(cur))
        return out

    c.setTitle("Russia Exposure Report")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, f"Russia Exposure Report — {rep.get('match_name_used') or rep.get('company_input')}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(x_margin, y, f"Generated: {rep.get('generated_at','')}")
    y -= 18

    em = rep.get("entity_meta", {})
    line(f"Company input: {rep.get('company_input','')}")
    if rep.get("match_name_used"):
        line(f"Match name used: {rep.get('match_name_used','')}")
    if em.get("legal_name"):
        line(f"Legal name (LEI): {em['legal_name']}")
    if em.get("LEI"):
        line(f"LEI: {em['LEI']}")
    if em.get("ISIN"):
        line(f"ISIN: {em['ISIN']}")

    y -= 10
    line(f"Risk score (1–5): {rep.get('risk_score_1_to_5', '')}", size=12)

    x = rep.get("exposure_classification", {})
    y -= 2
    line(f"Exposure classification: {x.get('label','')} — {x.get('rationale','')}")

    y -= 6
    line("Sanctions screening:", size=12)
    if rep.get("sanctions_hits"):
        for h in rep["sanctions_hits"][:12]:
            line(f"- {h['list']}: {h['name']} (score {round(h['score'],1)})")
    else:
        line("- No high-confidence matches")

    y -= 6
    line("Leave-Russia.org:", size=12)
    hits = rep.get("leave_russia_hits", [])
    if hits:
        for h in hits[:6]:
            line(f"- {h.get('name')} — {h.get('category') or 'N/A'}")
    else:
        line("- No matching entry found")

    # SEC/IR summary
    y -= 6
    line("Filings & IR scan:", size=12)
    fs = rep.get("filings_scan", {})
    sec_hits = sum(1 for f in fs.get("sec", []) if f.get("hits",0) > 0)
    ir_hits = sum(1 for f in fs.get("ir", []) if f.get("hits",0) > 0)
    line(f"- SEC filings with hits: {sec_hits}")
    line(f"- IR docs with hits: {ir_hits}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def build_summary_row(rep: Dict[str, Any]) -> Dict[str, Any]:
    """Create a concise row for the evidence grid / portfolio summary."""
    sanctions = rep.get("sanctions_hits", [])
    s_sources = sorted(set(h.get("list","") for h in sanctions))
    s_exact = any(h.get("method") == "exact" for h in sanctions)
    s_any = bool(sanctions)
    leave = rep.get("leave_russia_hits", [])
    leave_cat = ""
    if leave:
        # pick the top exact, else first
        exacts = [l for l in leave if l.get("method") == "exact"]
        tgt = exacts[0] if exacts else leave[0]
        leave_cat = tgt.get("category","")
    fs = rep.get("filings_scan", {})
    sec_hits = sum(1 for f in fs.get("sec", []) if f.get("hits",0) > 0)
    ir_hits = sum(1 for f in fs.get("ir", []) if f.get("hits",0) > 0)
    exposure = rep.get("exposure_classification", {})
    art_n = len(rep.get("articles_df", pd.DataFrame()))
    return {
        "Company": rep.get("match_name_used") or rep.get("company_input"),
        "RiskScore": rep.get("risk_score_1_to_5"),
        "ExposureLabel": exposure.get("label",""),
        "SanctionsHit": "Yes" if s_any else "No",
        "SanctionsSources": ", ".join(s_sources),
        "ExactSanctionsMatch": "Yes" if s_exact else "No",
        "LeaveRussiaStatus": leave_cat,
        "SEC_FilingsWithHits": sec_hits,
        "IR_DocsWithHits": ir_hits,
        "NewsArticlesCount": art_n,
        "GeneratedAt": rep.get("generated_at","")
    }

def render_summary_table(rows: List[Dict[str, Any]]):
    if not rows:
        st.info("No results to summarize.")
        return
    df = pd.DataFrame(rows)
    st.markdown("### Evidence grid (summary across sources)")
    st.dataframe(df)
    csv_bytes = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    st.download_button("Download summary CSV", data=csv_bytes, file_name="russia_exposure_summary.csv", mime="text/csv")

def process_company(entry: Dict[str, str], shared: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the full pipeline for a single company entry. Returns (report, summary_row)."""
    company = entry.get("company","").strip()
    legal_name = entry.get("legal_name","").strip()
    isin = entry.get("isin","").strip()
    lei = entry.get("lei","").strip()
    ticker = entry.get("ticker","").strip().upper()
    ir_url = entry.get("ir_url","").strip()
    years = shared.get("years", 5)

    # Resolve LEI -> legal
    meta = {"CompanyInput": company, "LegalNameInput": legal_name, "ISIN": isin, "LEI": lei, "Ticker": ticker, "IR_URL": ir_url}
    if lei:
        try:
            lei_info = resolve_lei(lei)
            if lei_info:
                meta["legal_name"] = lei_info.get("legal_name")
                meta["aliases"] = lei_info.get("aliases")
        except Exception:
            pass
    match_name = (meta.get("legal_name") or legal_name or company).strip()

    # SEC ticker -> CIK
    cik, sec_title = "", ""
    try:
        if shared.get("tmap") is None:
            shared["tmap"] = sec_ticker_map()
        tmap = shared["tmap"]
        if ticker or match_name:
            cik, sec_title = resolve_cik(ticker, match_name, tmap)
            if cik:
                meta["CIK"] = cik; meta["SECTitle"] = sec_title
    except Exception:
        pass

    # Fetch sources (reuse cached functions)
    articles = fetch_gdelt(company or match_name, years=years)
    ofac = shared["ofac"]; eu = shared["eu"]; uk = shared["uk"]; un = shared["un"]; leave_df = shared["leave"]
    os_matches = opensanctions_match(name=match_name, lei=lei, isin=isin)

    filings_scan = scan_sec_and_ir(cik, ticker, match_name, ir_url)

    rep = build_report(company, match_name, articles, ofac, eu, uk, un, leave_df, os_matches, filings_scan, meta)
    row = build_summary_row(rep)
    return rep, row

# --------------- UI ---------------

st.title("🛰️ Company Russia Exposure Reporter")

tab1, tab2 = st.tabs(["Single company", "Portfolio (multi-company)"])

with tab1:
    with st.expander("What this app does / limitations", expanded=False):
        st.markdown("""
**Data sources**
- News: **GDELT** (N-year lookback)
- Sanctions: **OFAC SDN**, **EU CFSP**, **UK HMT (OFSI)**, **UN**, **OpenSanctions** (API)
- NGO: **leave-russia.org** (live scrape each run)
- Filings/IR: **SEC EDGAR** (10-K/10-Q/20-F/6-K) + **optional IR URL**

**Features**
- Exact-first name matching, fallback fuzzy (≥90)
- Optional entity resolution via **LEI → GLEIF** (legal name + aliases)
- Filings and IR pages scanned for "Russia/Belarus/..." with **snippets**
- **Evidence grid**: summary table consolidating key signals

**Limitations**
- Endpoints and site layouts can change; adjust scraper if needed.
- Use a proper **User-Agent** (set ENV `SEC_EMAIL="you@yourdomain.com"` for SEC).
- Always confirm with official filings; this tool is indicative.
""")

    with st.container():
        colA, colB, colC = st.columns([2,1,1])
        with colA:
            company = st.text_input("Company (free-text)", placeholder="e.g., Ikea, BP, Nestlé, Ericsson")
            legal_name = st.text_input("Official Legal Name (optional)", placeholder="e.g., IKEA Holding B.V.")
            ir_url = st.text_input("Investor Relations URL (optional)", placeholder="e.g., https://www.company.com/investors")
        with colB:
            isin = st.text_input("ISIN (optional)", placeholder="e.g., SE0000108656")
            lei = st.text_input("LEI (optional)", placeholder="e.g., 5493001KJTIIGC8Y1R12")
        with colC:
            ticker = st.text_input("US Ticker (optional)", placeholder="e.g., MSFT")

    years = st.slider("News lookback (years)", min_value=1, max_value=10, value=5, key="years_single")
    run = st.button("Generate report", key="run_single")

    if run and (company.strip() or legal_name.strip() or lei.strip() or isin.strip() or ticker.strip()):
        with st.spinner("Resolving entity, scanning filings, and building report..."):
            # Load shared sources once
            shared = {
                "years": years,
                "ofac": fetch_ofac_sdn(),
                "eu": fetch_eu_sanctions(),
                "uk": fetch_uk_hmt(),
                "un": fetch_un_consolidated(),
                "leave": scrape_leave_russia(),
                "tmap": None
            }
            entry = {"company": company, "legal_name": legal_name, "isin": isin, "lei": lei, "ticker": ticker, "ir_url": ir_url}
            rep, row = process_company(entry, shared)

        # Render full report
        # ---- report header & sections ----
        # (reuse earlier renderer but inline to avoid extra complexity)
        # Use previous render_report function blocks for single-company
        def render_report(rep: Dict[str, Any]):
            st.markdown(f"### Russia Exposure Report — **{rep['match_name_used'] or rep['company_input']}**")
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                st.metric("Risk score (1–5)", rep["risk_score_1_to_5"])
            with col2:
                st.write("**Exposure classification**")
                st.write(f"{rep['exposure_classification']['label']}  \n*{rep['exposure_classification']['rationale']}*")
            with col3:
                st.caption(f"Generated at {rep['generated_at']}")

            em = rep.get("entity_meta", {})
            with st.expander("Entity details (resolution inputs)", expanded=False):
                st.json(em)

            st.markdown("---")
            st.markdown("#### Sanctions screening (OFAC/EU/UK/UN/OpenSanctions)")
            if rep["sanctions_hits"]:
                st.warning("**Potential matches found — manual verification required**")
                st.dataframe(pd.DataFrame(rep["sanctions_hits"]))
            else:
                st.success("No close matches across OFAC/EU/UK/UN/OpenSanctions (≥90 or exact).")

            st.markdown("#### Leave-Russia.org check (live scrape)")
            hits = rep.get("leave_russia_hits", [])
            if hits:
                df = pd.DataFrame(hits)
                st.dataframe(df[["name","score","method","category","description","url"]])
            else:
                st.info("No matching entry found on leave-russia.org.")

            st.markdown("#### Recent articles mentioning Russia (with links)")
            art_df = rep["articles_df"]
            if art_df.empty:
                st.info("No recent relevant articles found via GDELT for the chosen lookback.")
            else:
                max_items = 50
                items = []
                for _, row in art_df.head(max_items).iterrows():
                    title = (row.get("title") or "").strip() or "(no title)"
                    url = row.get("url") or ""
                    domain = row.get("domain") or ""
                    seendate = row.get("seendate") or ""
                    items.append(f"- [{title}]({url}) — {domain} — {seendate}")
                st.markdown("\n".join(items))
                with st.expander("Raw articles table", expanded=False):
                    st.dataframe(art_df[["seendate","title","domain","sourceCountry","url"]])

            # Filings & IR
            st.markdown("#### Filings & IR scan (keywords: Russia/Belarus/etc.)")
            fs = rep.get("filings_scan", {})
            sec_df = pd.DataFrame(fs.get("sec", []))
            ir_df = pd.DataFrame(fs.get("ir", []))
            if not sec_df.empty:
                st.subheader("SEC filings")
                st.dataframe(sec_df[["form","date","url","hits"]])
                with st.expander("SEC snippets", expanded=False):
                    for _, r in sec_df.iterrows():
                        if r.get("hits",0) > 0:
                            st.markdown(f"**{r['form']} — {r['date']}**  \n<{r['url']}>")
                            for sn in r.get("snippets", [])[:3]:
                                st.markdown(f"> {sn}")
                            st.markdown("---")
            else:
                st.info("No SEC filings scanned or none available for this company.")

            if not ir_df.empty:
                st.subheader("IR page & linked docs")
                st.dataframe(ir_df[["type","url","hits"]])
                with st.expander("IR snippets", expanded=False):
                    for _, r in ir_df.iterrows():
                        if r.get("hits",0) > 0:
                            st.markdown(f"**{r['type']}**  \n<{r['url']}>")
                            for sn in r.get("snippets", [])[:3]:
                                st.markdown(f"> {sn}")
                            st.markdown("---")
            else:
                st.info("No IR URL provided or no hits found.")

            # Downloads
            export = rep.copy()
            df = export.pop("articles_df")
            json_bytes = io.BytesIO(json.dumps(export, indent=2).encode("utf-8"))
            st.download_button("Download JSON summary", data=json_bytes, file_name=f"{(rep['match_name_used'] or rep['company_input']).replace(' ','_')}_russia_report.json", mime="application/json")

            if not df.empty:
                csv_bytes = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
                st.download_button("Download articles CSV", data=csv_bytes, file_name=f"{(rep['match_name_used'] or rep['company_input']).replace(' ','_')}_russia_articles.csv", mime="text/csv")

            try:
                pdf_bytes = render_pdf_summary(rep)
                st.download_button("Download PDF summary", data=pdf_bytes, file_name=f"{(rep['match_name_used'] or rep['company_input']).replace(' ','_')}_russia_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"PDF export failed: {e}")

        render_report(rep)

        # Evidence grid for single company (one row)
        render_summary_table([row])

    elif run:
        st.error("Please provide at least one of: Company, Legal Name, LEI, ISIN, or US Ticker.")

with tab2:
    st.markdown("Enter multiple companies (one per line) using any of these formats:  \n"
                "`Company Name`  \n"
                "`Company Name, ISIN`  \n"
                "`Company Name, ISIN, LEI`  \n"
                "`Company Name, ISIN, LEI, US Ticker`  \n"
                "You can also upload a CSV with columns: company, isin, lei, ticker, legal_name, ir_url")

    colU1, colU2 = st.columns([2,1])
    with colU1:
        multi_text = st.text_area("Companies list", height=160, placeholder="Ikea\nBP, GB0007980591\nEricsson, SE0000108656, 5493000KM0LZFH1Z1E08, ERIC\n...")
    with colU2:
        file = st.file_uploader("Or upload CSV", type=["csv"])

    years_multi = st.slider("News lookback (years)", min_value=1, max_value=10, value=5, key="years_multi")
    run_multi = st.button("Run portfolio scan", key="run_multi")

    def parse_lines(text: str) -> List[Dict[str,str]]:
        rows = []
        for line in (text or "").splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            entry = {"company": "", "isin": "", "lei": "", "ticker": "", "legal_name": "", "ir_url": ""}
            if len(parts) >= 1: entry["company"] = parts[0]
            if len(parts) >= 2: entry["isin"] = parts[1]
            if len(parts) >= 3: entry["lei"] = parts[2]
            if len(parts) >= 4: entry["ticker"] = parts[3]
            rows.append(entry)
        return rows

    if run_multi:
        entries = []
        if file is not None:
            try:
                df = pd.read_csv(file).fillna("")
                for _, r in df.iterrows():
                    entries.append({
                        "company": r.get("company",""),
                        "isin": r.get("isin",""),
                        "lei": r.get("lei",""),
                        "ticker": r.get("ticker",""),
                        "legal_name": r.get("legal_name",""),
                        "ir_url": r.get("ir_url",""),
                    })
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
        else:
            entries = parse_lines(multi_text)

        if not entries:
            st.error("Please provide companies via the text box or upload a CSV.")
        else:
            with st.spinner(f"Scanning {len(entries)} companies..."):
                shared = {
                    "years": years_multi,
                    "ofac": fetch_ofac_sdn(),
                    "eu": fetch_eu_sanctions(),
                    "uk": fetch_uk_hmt(),
                    "un": fetch_un_consolidated(),
                    "leave": scrape_leave_russia(),
                    "tmap": None
                }
                reports = []
                rows = []
                for entry in entries:
                    try:
                        rep, row = process_company(entry, shared)
                        reports.append(rep); rows.append(row)
                    except Exception as e:
                        rows.append({"Company": entry.get("company","(unknown)"), "RiskScore": "", "ExposureLabel": "", "SanctionsHit": "ERROR", "SanctionsSources": str(e)})
                # Render evidence grid
                render_summary_table(rows)

                # Offer a zip of JSON reports (optional small bundle)
                try:
                    import zipfile
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, mode="w") as z:
                        for rep in reports:
                            fn = f"{(rep['match_name_used'] or rep['company_input']).replace(' ','_')}_russia_report.json"
                            z.writestr(fn, json.dumps({k:v for k,v in rep.items() if k!='articles_df'}, indent=2))
                    buf.seek(0)
                    st.download_button("Download all JSON reports (zip)", data=buf, file_name="russia_reports_bundle.zip", mime="application/zip")
                except Exception:
                    pass
