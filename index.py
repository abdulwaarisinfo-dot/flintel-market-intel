"""
Flintel — Website Market Intelligence + Reddit Signal Tracking
----------------------------------------------------------------
FastAPI backend that:
  1. Accepts a domain/URL from the user.
  2. Fetches and cleans the text of that website (homepage + a few likely
     secondary pages such as /about and /pricing).
  3. Sends that text to Claude with a carefully engineered prompt that
     returns a structured market-intelligence report: what the company
     does, who it's for, what pain points its buyers talk about, who its
     competitors are, and where to reach them.
  4. Using the discovery_keywords from that report, searches Reddit's public
     search endpoint for posts/discussions that mention those keywords (e.g.
     your own product name, or the pain points people discuss), and asks
     Claude to score each result for relevance/intent.
  5. Persists every website report AND every Reddit signal in MongoDB
     (database "fx_signals") so history can be re-fetched without
     re-analyzing.
  6. Serves a single-page UI (templates/web.html) that drives the whole
     flow.

IMPORTANT — what this does NOT do:
  This backend does not generate ready-to-post replies meant to look like
  organic, undisclosed endorsements. For each matched Reddit post it stores
  a set of internal "suggested_talking_points" — notes a human on your team
  can read and use to write their OWN reply, with their affiliation
  disclosed as required by Reddit's rules and by consumer-protection law in
  most jurisdictions. Nothing here posts to Reddit automatically.

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
# NOTE: per your setup, the Mongo *database* is "fx_signals" (you already
# have this database with other collections in it, e.g. your FX-signal
# data). This backend adds its own collections inside that same database:
#   fx_signals.reports    -> website market-intelligence reports
#   fx_signals.flintel_web_data   -> Reddit posts/signals matched to a domain
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "fx_signals")

SECONDARY_PATHS = ["/about", "/about-us", "/product", "/products", "/pricing", "/features"]
MAX_SITE_CHARS = 12000          # cap on scraped text sent to Claude
REQUEST_TIMEOUT_SECONDS = 15.0

# How many keywords (max) from Claude's discovery_keywords to actually
# match against the existing web_data collection, to keep queries bounded.
MAX_KEYWORDS_TO_MATCH = 6

if not ANTHROPIC_API_KEY:
    # Fail loudly at startup rather than on the first request.
    print("WARNING: ANTHROPIC_API_KEY is not set. /api/analyze will fail until it is.")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]

# Two DISTINCT collections inside the same fx_signals database:
#
# 1) flintel_web_data (NEW — this app writes here)
#    Every time a domain is analyzed, Claude's output (the report +
#    discovery_keywords) is appended here with an incrementing "sequence"
#    number. This is the record of "what did Claude find for this domain,
#    and in what order".
#
# 2) web_data (ALREADY EXISTS — this app only READS from here)
#    This is your existing, pre-populated collection of Reddit/Twitter/
#    Telegram posts (post_url, username, subreddit_or_channel, upvotes,
#    comments, search_keyword, google_rank, search_volume, etc.). This app
#    never writes to it or modifies it — it only queries it to find rows
#    whose search_keyword / subreddit_or_channel / text match the keywords
#    Claude just produced for a domain.
flintel_web_data_collection = db["flintel_web_data"]
web_data_collection = db["web_data"]

app = FastAPI(title="Flintel — Website Market Intelligence", version="1.2.0")
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
# Claude analysis (website -> market-intelligence report)
# --------------------------------------------------------------------------

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
   emotional, casual language — not corporate language.
4. Who are the 3-5 most direct competitors this buyer is currently using or \
   considering? For each one, what is the main reason buyers get frustrated with \
   them and start looking for an alternative?
5. What makes this offering genuinely different from those competitors — not just \
   marketing claims, but real functional differences?
6. Where would this buyer realistically be reached online — search, paid ads, \
   industry communities, marketplaces, or content platforms?

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
  language.
- "competitors" must include 3-5 real named competitors, not vague categories.
- "discovery_keywords" should include the product/company name itself plus 4-8 short \
  phrases a buyer or a frustrated competitor-customer would actually type into a \
  search box or into Reddit's search bar (used later to find relevant discussions).
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
# Store Claude's per-domain output, then match it against the EXISTING
# pre-populated web_data collection (no live Reddit/Twitter/Telegram fetch)
# --------------------------------------------------------------------------

async def store_flintel_web_data(domain: str, url: str, report: dict) -> dict:
    """
    Append Claude's analysis (report + discovery_keywords) for this domain
    into flintel_web_data, with an incrementing sequence number so history
    for a domain can be read back in order.
    """
    sequence = await flintel_web_data_collection.count_documents({"domain": domain}) + 1

    doc = {
        "domain": domain,
        "url": url,
        "sequence": sequence,
        "discovery_keywords": report.get("discovery_keywords") or [],
        "company_summary": report.get("company_summary", ""),
        "buyer_pain_points": report.get("buyer_pain_points", []),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = await flintel_web_data_collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
    except Exception as exc:
        logger.error("MongoDB insert_one (flintel_web_data) failed for domain=%s: %s", domain, exc)
        doc["_id"] = None
    return doc


async def match_existing_web_data(keywords: list[str], limit_per_keyword: int = 50) -> list[dict]:
    """
    Read-only query against the EXISTING web_data collection (already
    populated outside this app). For each keyword Claude produced, find
    documents where:
      - search_keyword matches the keyword (case-insensitive), OR
      - subreddit_or_channel matches the keyword, OR
      - the post text contains the keyword
    This app never writes to web_data — only reads/matches.
    """
    if not keywords:
        return []

    matches = []
    seen_ids = set()

    for keyword in keywords:
        pattern = re.escape(keyword)
        query = {
            "$or": [
                {"search_keyword": {"$regex": pattern, "$options": "i"}},
                {"subreddit_or_channel": {"$regex": pattern, "$options": "i"}},
                {"text": {"$regex": pattern, "$options": "i"}},
            ]
        }
        try:
            cursor = web_data_collection.find(query).limit(limit_per_keyword)
            async for doc in cursor:
                doc_id = str(doc["_id"])
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                doc["_id"] = doc_id
                doc["matched_keyword"] = keyword
                if isinstance(doc.get("posted_at"), datetime):
                    doc["posted_at"] = doc["posted_at"].isoformat()
                if isinstance(doc.get("fetched_at"), datetime):
                    doc["fetched_at"] = doc["fetched_at"].isoformat()
                matches.append(doc)
        except Exception as exc:
            logger.warning("web_data match query failed for keyword=%r: %s", keyword, exc)

    return matches


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "web.html", {})


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest):
    """
    Full pipeline for a domain:
      1. Fetch + analyze the website -> Claude produces a report + discovery_keywords.
         (Claude's involvement ends here — no scoring, no live fetching, nothing else.)
      2. Append that output to fx_signals.flintel_web_data (NEW collection,
         written by this app, one row per analysis with an incrementing
         "sequence" per domain).
      3. Using those keywords, run a read-only match against fx_signals.web_data
         (the EXISTING, pre-populated collection this app never writes to) —
         matching on search_keyword, subreddit_or_channel, or post text.
      4. Return the report + the matched rows to the caller.
    """
    try:
        url = normalize_url(payload.url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please enter a valid website address.")

    domain = urlparse(url).netloc.replace("www.", "")

    site_data = await fetch_site_text(url)
    report = await analyze_with_claude(domain, site_data)
    # --- Claude's job ends here. Everything below is plain Python/MongoDB. ---

    flintel_doc = await store_flintel_web_data(domain, url, report)

    keywords = (report.get("discovery_keywords") or [])[:MAX_KEYWORDS_TO_MATCH]
    if not keywords:
        keywords = [domain]

    matches = await match_existing_web_data(keywords)

    return JSONResponse({
        "domain": domain,
        "url": url,
        "report": report,
        "flintel_web_data_sequence": flintel_doc["sequence"],
        "matched_keywords": keywords,
        "matches_found": len(matches),
        "matches": matches,
    })


@app.get("/api/reports/{domain}")
async def get_reports(domain: str):
    cursor = flintel_web_data_collection.find({"domain": domain}).sort("sequence", -1).limit(5)
    reports = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["created_at"] = doc["created_at"].isoformat()
        reports.append(doc)
    return JSONResponse({"domain": domain, "reports": reports})


@app.get("/api/signals/{domain}")
async def get_signals(domain: str, subreddit: str | None = None, limit: int = 50):
    """
    Re-run the match for a domain: read the most recent discovery_keywords
    Claude produced for it from flintel_web_data, then match those keywords
    against the EXISTING web_data collection (read-only) and return the
    matching rows — post_url, username, subreddit_or_channel, upvotes,
    comments, google_rank, search_volume, everything that's already stored
    there. Optionally filter to a single subreddit_or_channel.
    """
    latest = await flintel_web_data_collection.find_one(
        {"domain": domain}, sort=[("sequence", -1)]
    )
    if not latest:
        raise HTTPException(status_code=404, detail=f"No analysis found yet for domain={domain}.")

    keywords = (latest.get("discovery_keywords") or [])[:MAX_KEYWORDS_TO_MATCH]
    if not keywords:
        keywords = [domain]

    matches = await match_existing_web_data(keywords, limit_per_keyword=limit)
    if subreddit:
        matches = [m for m in matches if m.get("subreddit_or_channel") == subreddit]

    return JSONResponse({
        "domain": domain,
        "matched_keywords": keywords,
        "count": len(matches),
        "signals": matches,
    })


@app.get("/api/health")
async def health():
    mongo_status = "unknown"
    try:
        await mongo_client.admin.command("ping")
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"error: {exc}"
    return {"status": "ok", "mongodb": mongo_status, "database": MONGODB_DB_NAME}
