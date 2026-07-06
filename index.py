"""
Flintel — Website Market Intelligence
--------------------------------------
FastAPI backend that:
  1. Accepts a domain/URL from the user.
  2. Fetches and cleans the text of that website (homepage + a few likely
     secondary pages such as /about and /pricing).
  3. Sends that text to Claude with a carefully engineered prompt that
     returns a structured market-intelligence report: what the company
     does, who it's for, what pain points its buyers talk about, who its
     competitors are, and where to reach them.
  4. Matches that report against the `fx_signals` collection (pre-scraped
     social/search posts) to surface conversations relevant to this
     specific domain, and tags the matched documents so the match sticks.
  5. Persists every report in MongoDB so a domain's history can be
     re-fetched without re-analyzing.
  6. Serves a single-page UI (templates/web.html) that drives the whole
     flow.

Run:
    pip install -r requirements.txt
    uvicorn index:app --reload
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import anthropic
import httpx
from bs4 import BeautifulSoup
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logger = logging.getLogger("flintel")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Haiku 4.5 is the default: this task is text extraction + structured
# summarization against a fixed schema, not multi-step reasoning, so the
# cheapest current-generation model is the right fit. Override via env var
# (e.g. to claude-sonnet-5) if you ever need deeper reasoning per site.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "flintel")

SECONDARY_PATHS = ["/about", "/about-us", "/product", "/products", "/pricing", "/features"]
MAX_SITE_CHARS = 12000          # cap on scraped text sent to Claude
REQUEST_TIMEOUT_SECONDS = 15.0

# How many fx_signals documents to match & return per analysis.
MAX_MATCHED_SIGNALS = 30

if not ANTHROPIC_API_KEY:
    # Fail loudly at startup rather than on the first request.
    print("WARNING: ANTHROPIC_API_KEY is not set. /api/analyze will fail until it is.")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]
reports_collection = db["reports"]
fx_signals_collection = db["fx_signals"]  # pre-scraped social/search posts, matched per domain

app = FastAPI(title="Flintel — Website Market Intelligence", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Serve static assets (favicon.ico, logo, css/js if ever split out) from a
# top-level "static" folder, sibling to "templates". Mounted as "static" so
# templates can reference files via {{ request.url_for('static', path=...) }}.
app.mount("/static", StaticFiles(directory="static"), name="static")


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=300, description="Website URL or bare domain")

    @field_validator("url")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("URL cannot be blank")
        return v


# --------------------------------------------------------------------------
# Helpers: fetching & cleaning website content
# --------------------------------------------------------------------------

def normalize_url(raw: str) -> str:
    """Turn 'getflintel.com' or partial input into a fetchable https:// URL."""
    raw = raw.strip()
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc or "." not in parsed.netloc:
        raise ValueError("Not a valid website address")
    return raw


def extract_visible_text(soup: BeautifulSoup, max_lines: int = 400) -> str:
    """Strip non-content tags and return readable, deduplicated text lines."""
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    seen = set()
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


async def fetch_site_text(url: str) -> dict:
    """Fetch the homepage plus a handful of likely secondary pages."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FlintelBot/1.0; +https://getflintel.com)"}
    title, meta_description = "", ""
    page_texts: list[str] = []

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=REQUEST_TIMEOUT_SECONDS, headers=headers
    ) as http:
        try:
            resp = await http.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=400, detail=f"Could not reach that website ({exc}).")

        soup = BeautifulSoup(resp.text, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            meta_description = desc_tag["content"].strip()
        page_texts.append(extract_visible_text(soup))

        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        for path in SECONDARY_PATHS:
            try:
                sub_resp = await http.get(base + path)
                if sub_resp.status_code == 200:
                    sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                    page_texts.append(extract_visible_text(sub_soup))
            except httpx.HTTPError:
                continue  # secondary pages are best-effort

    combined = "\n\n".join(page_texts)[:MAX_SITE_CHARS]
    if not combined.strip():
        combined = "(No readable text could be extracted from this website.)"

    return {"title": title, "meta_description": meta_description, "content": combined}


# --------------------------------------------------------------------------
# Claude analysis
# --------------------------------------------------------------------------

# This system prompt is the core "product logic" of Flintel. It is written so
# that Claude behaves like a senior market analyst, reasons carefully about
# the raw scraped text, and returns ONLY machine-readable JSON matching a
# fixed schema — never prose, never markdown fences, never commentary.
ANALYSIS_SYSTEM_PROMPT = """\
You are Flintel's senior market-intelligence analyst. You specialize in reading a \
company's own website and reverse-engineering an accurate picture of its business, \
its buyers, and how to reach them — the same way a sharp growth consultant would \
after 30 minutes on the site.

You will be given the domain name, page title, meta description, and cleaned text \
scraped from a company's homepage and a few likely secondary pages (about, product, \
pricing, features). The scrape may be incomplete or messy. Do your best with what is \
given; never invent specific facts (numbers, customer names, funding, integrations) \
that are not supported by the text. If the site does not give you enough to answer a \
field confidently, make a clearly reasonable inference from context and keep it general \
rather than fabricating specifics.

Think step by step, privately, before answering:
1. What does this company actually sell or offer? Separate marketing language from the \
   real underlying product or service.
2. Who is the buyer — by role, business type, or life situation — not just "everyone."
3. What problem, frustration, or task drives that buyer to search for a solution like \
   this? What words would they type into Google when frustrated? Write them as raw, \
   emotional, casual language — not corporate language. \
   Bad example: "payment processing reliability concerns". \
   Good example: "stripe froze my account with no warning and no explanation".
4. Who are the 3-5 most direct competitors this buyer is currently using or \
   considering? For each one, what is the main reason buyers get frustrated with \
   them and start looking for an alternative?
5. What makes this offering genuinely different from those competitors — not just \
   marketing claims, but real functional differences?
6. Where would this buyer realistically be reached online — search, paid ads, \
   industry communities, marketplaces, or content platforms? Do not suggest Reddit \
   or specific subreddits; this analysis is about the buyer's broader online \
   footprint, not any single platform.

After reasoning, respond with ONLY a single valid JSON object — no markdown code fences, \
no preamble, no trailing commentary — matching exactly this schema:

{
  "company_summary": string,
  "problem_solved": string,
  "target_audience": [
    { "persona": string, "description": string }
  ],
  "buyer_pain_points": [string],
  "competitors": [
    { "name": string, "why_buyers_leave": string }
  ],
  "unique_value_prop": string,
  "discovery_keywords": [string],
  "recommended_channels": [string],
  "confidence_notes": string
}

Rules:
- Output valid JSON only. No ```json fences. No text before or after the object.
- Keep every string concise and specific to this company — never generic filler that \
  could apply to any business.
- "buyer_pain_points" must read like real frustrated search queries, not corporate \
  language. Use the bad/good example above as your benchmark for every phrase.
- "competitors" must include 3-5 real named competitors, not vague categories. \
  "why_buyers_leave" should be one specific frustration, not a general statement.
- "recommended_channels" must NOT include Reddit or specific subreddits. Focus on \
  search/SEO, paid channels, industry communities, marketplaces, review sites, \
  newsletters, and platform-specific content (e.g. LinkedIn, YouTube, niche forums \
  outside Reddit) instead.
- If the scraped text is empty, contradictory, or clearly not a real product/company \
  site, say so honestly inside the relevant fields and set "confidence_notes" \
  accordingly, but still return the full JSON schema.
"""


def _extract_json(raw_text: str) -> dict:
    """Best-effort parse of Claude's JSON reply, tolerant of stray fences/whitespace."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def analyze_with_claude(domain: str, site_data: dict) -> dict:
    user_prompt = f"""Domain: {domain}
Page title: {site_data["title"] or "N/A"}
Meta description: {site_data["meta_description"] or "N/A"}

Cleaned website text (homepage + secondary pages, may be truncated):
\"\"\"
{site_data["content"]}
\"\"\"

Analyze this site and return the market-intelligence report as JSON only, following the schema exactly."""

    try:
        message = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Analysis service error: {exc}")

    raw_text = "".join(block.text for block in message.content if block.type == "text")
    try:
        return _extract_json(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="The analysis service returned an unreadable report.")


# --------------------------------------------------------------------------
# fx_signals matching
# --------------------------------------------------------------------------
# `fx_signals` is a separately-populated collection of scraped social/search
# posts (Reddit, forums, etc.) with fields like:
#   message_id, platform, post_url, text, username, subreddit_or_channel,
#   posted_at, google_rank, search_volume, upvotes, comments, search_keyword,
#   intent_score, is_relevant, reply_draft, status
#
# After a site is analyzed, we build a keyword set from that report
# (discovery keywords, pain-point phrases, competitor names) and look for
# fx_signals documents whose `text` or `search_keyword` mentions any of them.
# Matches get tagged with a `flintel_web_data` sub-object so the association
# between a signal and a domain persists in the collection itself, and so a
# signal already matched to one domain can still be found for another.

def _build_match_keywords(report: dict) -> list[str]:
    keywords: set[str] = set()

    for kw in report.get("discovery_keywords") or []:
        if isinstance(kw, str):
            k = kw.strip()
            if k and len(k) > 2 and k is not Ellipsis:
                keywords.add(k)

    for pain in report.get("buyer_pain_points") or []:
        if isinstance(pain, str):
            p = pain.strip()
            if p and len(p) > 2 and p is not Ellipsis:
                keywords.add(p)

    for comp in report.get("competitors") or []:
        name = (comp or {}).get("name")
        if isinstance(name, str):
            n = name.strip()
            if n and len(n) > 1 and n is not Ellipsis:
                keywords.add(n)

    return list(keywords)


def _safe_str(val: object) -> str:
    """Return a safe string for non-string / Ellipsis values."""
    if val is None:
        return ""
    if val is Ellipsis:
        return ""
    if isinstance(val, str):
        return val
    try:
        return str(val)
    except Exception:
        return ""


def _serialize_signal(doc: dict) -> dict:
    """Make a fx_signals doc JSON-safe (ObjectId / datetime -> str)."""
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    for key in ("posted_at",):
        val = doc.get(key)
        if isinstance(val, datetime):
            doc[key] = val.isoformat()
    fwd = doc.get("flintel_web_data")
    if isinstance(fwd, dict) and isinstance(fwd.get("matched_at"), datetime):
        fwd["matched_at"] = fwd["matched_at"].isoformat()
    return doc


async def match_and_tag_signals(domain: str, url: str, report: dict) -> list[dict]:
    """
    Find fx_signals documents relevant to this domain's report, tag them with
    flintel_web_data (so the match persists in the collection), and return the
    matched documents for display.
    """
    keywords = _build_match_keywords(report)
    if not keywords:
        return []

    # Escape each keyword/phrase for safe regex use, OR them together.
    pattern = "|".join(re.escape(k) for k in keywords)
    # Search across multiple likely fields in the fx_signals docs.
    search_fields = ["text", "search_keyword", "subreddit_or_channel", "username", "post_url"]
    mongo_or = []
    for f in search_fields:
        mongo_or.append({f: {"$regex": pattern, "$options": "i"}})
    mongo_query = {"$or": mongo_or}

    matched_docs: list[dict] = []
    matched_ids: list[ObjectId] = []

    try:
        cursor = (
            fx_signals_collection.find(mongo_query)
            .sort("intent_score", -1)
            .limit(MAX_MATCHED_SIGNALS)
        )
        async for doc in cursor:
            matched_ids.append(doc["_id"])
            matched_docs.append(_serialize_signal(doc))
    except Exception as exc:
        # A missing/misconfigured fx_signals collection shouldn't break the
        # main analysis flow — the site report is still useful on its own.
        logger.error("fx_signals lookup failed for domain=%s: %s", domain, exc)
        return []

    if matched_ids:
        matched_keywords_preview = keywords[:25]
        try:
            # Store the association under both `flintel_web_data` (existing key)
            # and `web_data` (alternate name) so downstream UIs/scripts can
            # reference either field depending on their expectations.
            payload = {
                "domain": domain,
                "url": url,
                "matched_at": datetime.now(timezone.utc),
                "matched_keywords": matched_keywords_preview,
            }
            await fx_signals_collection.update_many(
                {"_id": {"$in": matched_ids}},
                {"$set": {"flintel_web_data": payload, "web_data": payload}},
            )
        except Exception as exc:
            logger.error("fx_signals tagging (update_many) failed for domain=%s: %s", domain, exc)

    return matched_docs


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "web.html", {})


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest):
    try:
        url = normalize_url(payload.url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please enter a valid website address.")

    domain = urlparse(url).netloc.replace("www.", "")

    site_data = await fetch_site_text(url)
    report = await analyze_with_claude(domain, site_data)

    document = {
        "domain": domain,
        "url": url,
        "report": report,
        "created_at": datetime.now(timezone.utc),
    }

    report_id = None
    try:
        result = await reports_collection.insert_one(document)
        report_id = str(result.inserted_id)
    except Exception as exc:
        # Storage is a nice-to-have; never fail the user's analysis because of it,
        # but log it loudly so a broken DB connection doesn't go unnoticed.
        logger.error("MongoDB insert_one failed for domain=%s: %s", domain, exc)

    # Cross-reference against previously scraped fx_signals posts so the user
    # sees which real conversations line up with this specific business.
    matched_signals = await match_and_tag_signals(domain, url, report)

    return JSONResponse(
        {
            "id": report_id,
            "domain": domain,
            "url": url,
            "report": report,
            "matched_signals": matched_signals,
        }
    )


@app.get("/api/reports/{domain}")
async def get_reports(domain: str):
    cursor = reports_collection.find({"domain": domain}).sort("created_at", -1).limit(5)
    reports = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["created_at"] = doc["created_at"].isoformat()
        reports.append(doc)
    return JSONResponse({"domain": domain, "reports": reports})


@app.get("/api/signals/{domain}")
async def get_signals(domain: str):
    """
    Re-fetch whatever fx_signals documents are currently tagged as matching
    this domain (i.e. flintel_web_data.domain == domain), without re-running
    the analysis or the matching query.
    """
    cursor = (
        fx_signals_collection.find({"flintel_web_data.domain": domain})
        .sort("intent_score", -1)
        .limit(MAX_MATCHED_SIGNALS)
    )
    signals = [_serialize_signal(doc) async for doc in cursor]
    return JSONResponse({"domain": domain, "signals": signals})


@app.get("/api/health")
async def health():
    mongo_status = "unknown"
    try:
        await mongo_client.admin.command("ping")
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"error: {exc}"
    return {"status": "ok", "mongodb": mongo_status}
