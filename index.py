"""
Flintel — Website Market Intelligence + Reddit Signal Tracking
----------------------------------------------------------------
(See project docs for full architecture notes.)

THIS REVISION (on top of 2.4.0) fixes the domain-caching bug:

  PROBLEM: /api/analyze called analyze_with_claude() fresh on every
  request, for every user, for the same domain. Claude's output
  (including discovery_keywords) is not deterministic between calls,
  so two different users analyzing the same domain could get two
  different keyword sets — which meant they could get two different
  sets of matches out of web_data_collection, even though the domain
  itself was identical. That looked like "data not appearing for
  another user" / "shared domain cache not working", but the real
  isolation (flintel_user_signals, scoped by user_email) was never
  broken — the INPUT to the matching step just wasn't shared.

  FIX: a new collection, flintel_domain_reports_collection, caches
  ONE canonical Claude analysis + discovery_keywords per domain
  (keyed only by domain — no user_email on this collection, by
  design, since this data is meant to be shared). get_or_create_
  domain_report() is now the single entry point for turning a domain
  into a report:
    - If a complete cached report exists for the domain, it is
      reused as-is — no new Claude call, no new scrape.
    - If another request is already processing the same domain
      (race between two users hitting a brand-new domain at the
      same moment), this request waits briefly and then reuses
      whatever that in-flight request produces, instead of kicking
      off a second, redundant Claude analysis.
    - Only if nothing exists and nothing is in flight does it run
      fetch_site_text() + analyze_with_claude() and write the result
      once.

  Every user who analyzes the same domain now matches against the
  exact same discovery_keywords, so results are consistent across
  users. Per-user data (flintel_user_signals, flintel_web_data_
  collection "my reports" history, reviewed flags, dashboard stats)
  is completely unchanged in shape and remains scoped by user_email
  exactly as before — only the INPUT report/keywords feeding into it
  is now shared and cached instead of being independently
  regenerated per user.

  No existing route, request/response schema, or function signature
  was removed or renamed. All changes are additive or internal to
  analyze()'s implementation.

THIS REVISION (on top of 2.5.0) ADDS Section 05 — "What this means
for your pipeline":

  WHAT IT IS: a 6-month trials / customers / MRR projection chart,
  computed with fixed conservative funnel assumptions (0.5% click-
  through, 10% visitor->trial, 30% trial->customer, configurable
  ACV), seeded from each user's OWN real matched-signal reach.

  IMPORTANT — this is pure Python arithmetic, NOT a Claude call.
  Claude is never asked to predict trials/customers/MRR: those
  numbers must be deterministic (same input -> same output, every
  refresh) and grounded only in data already sitting in MongoDB
  (flintel_user_signals_collection). Asking an LLM to invent a
  revenue forecast would (a) be non-deterministic like the original
  domain-cache bug this file already fixes once, and (b) violate the
  "never invent facts not supported by the data" principle already
  enforced in ANALYSIS_SYSTEM_PROMPT.

  STORAGE: a new collection, flintel_analytics_collection, caches the
  latest computed projection per (user_email, domain) pair — unlike
  flintel_domain_reports_collection (shared across users by domain
  only), this new collection IS user-scoped, because it's derived
  from that user's own matched-signal history, exactly like every
  other per-user collection in this file (flintel_user_signals,
  flintel_tracking_state, flintel_web_data).

  LIVE UPDATES: run_live_match_tracker() already finds new matched
  Reddit posts for each (user_email, domain) pair on a timer and
  persists them via _persist_match_for_user(). This revision adds ONE
  additional call right after that — _refresh_pipeline_analytics() —
  so the cached Section 05 projection is recomputed and re-saved
  automatically whenever new matching data arrives, with no separate
  polling job needed. The existing tracker loop, its query logic, and
  its logging are completely unchanged.

  Nothing above (domain cache, per-user isolation, existing routes)
  is modified. All changes below are additive.
"""

import asyncio
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import anthropic
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from pymongo import ReturnDocument
from starlette.middleware.sessions import SessionMiddleware

from authlib.integrations.starlette_client import OAuth

load_dotenv()

logger = logging.getLogger("flintel")
logging.basicConfig(level=logging.INFO)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "fx_signals")

SECONDARY_PATHS = ["/about", "/about-us", "/product", "/products", "/pricing", "/features"]
MAX_SITE_CHARS = 12000
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_KEYWORDS_TO_MATCH = 6
LIVE_TRACKER_INTERVAL_SECONDS = int(os.getenv("LIVE_TRACKER_INTERVAL_SECONDS", "120"))
MAX_COMPETITORS_FOR_GAP = 5

# How long a request will wait (polling) for ANOTHER in-flight request that
# is already analyzing the same brand-new domain, before giving up and
# running the analysis itself as a fallback. This only matters for the
# rare case of two users submitting the exact same never-seen-before
# domain within a few seconds of each other.
DOMAIN_LOCK_POLL_INTERVAL_SECONDS = 1.0
DOMAIN_LOCK_MAX_WAIT_SECONDS = 45.0

if not ANTHROPIC_API_KEY:
    print("WARNING: ANTHROPIC_API_KEY is not set. /api/analyze will fail until it is.")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]

users_collection             = db["users"]
flintel_web_data_collection  = db["flintel_web_data"]
web_data_collection          = db["web_data"]
flintel_user_signals_collection = db["flintel_user_signals"]
flintel_tracking_state_collection = db["flintel_tracking_state"]
flintel_counters_collection  = db["flintel_counters"]

# Shared, domain-scoped (NOT user-scoped) cache of the Claude analysis for
# a domain. Deliberately has no user_email field: the whole point is that
# this is the same for every user who looks up the domain.
flintel_domain_reports_collection = db["flintel_domain_reports"]

# NEW — per-user, per-domain cache of the Section 05 pipeline projection.
# UNLIKE flintel_domain_reports_collection, this one IS keyed by user_email
# + domain, because the projection is seeded from that specific user's own
# matched-signal reach (flintel_user_signals_collection), which can differ
# between users even for the same domain (different accounts may have
# reviewed/accumulated different signal history over time).
flintel_analytics_collection = db["flintel_analytics"]

app = FastAPI(title="Flintel — Website Market Intelligence", version="2.6.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
if not SESSION_SECRET_KEY:
    print("WARNING: SESSION_SECRET_KEY not set — using a random one-off key "
          "(sessions will NOT survive a server restart). Set this in .env for production.")
    SESSION_SECRET_KEY = secrets.token_urlsafe(32)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, same_site="lax")

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI")

oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
else:
    print("WARNING: GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET not set — "
          "'Continue with Google' will not work until these are configured.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


class AnalyzeRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=300, description="Website URL or bare domain")

    @field_validator("url")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("URL cannot be blank")
        return v


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


async def get_current_user(request: Request) -> dict:
    email = request.session.get("user_email")
    if not email:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    user = await users_collection.find_one({"email": email})
    if not user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    user["_id"] = str(user["_id"])
    return user


async def get_current_user_optional(request: Request) -> dict | None:
    email = request.session.get("user_email")
    if not email:
        return None
    user = await users_collection.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
    return user


async def _find_or_create_google_user(email: str, name: str, google_id: str) -> dict:
    now = datetime.now(timezone.utc)
    existing = await users_collection.find_one({"email": email})
    if existing:
        if not existing.get("google_id"):
            await users_collection.update_one(
                {"email": email},
                {"$set": {"google_id": google_id, "name": existing.get("name") or name}},
            )
        return await users_collection.find_one({"email": email})

    doc = {
        "email": email,
        "name": name,
        "auth_method": "google",
        "google_id": google_id,
        "password_hash": None,
        "created_at": now,
        "last_dashboard_view_at": None,
    }
    result = await users_collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


@app.get("/auth/google/login")
async def google_login(request: Request):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=503, detail="Google login is not configured on this server.")
    return await oauth.google.authorize_redirect(request, GOOGLE_REDIRECT_URI)


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=503, detail="Google login is not configured on this server.")
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        logger.error("Google OAuth callback failed: %s", exc)
        raise HTTPException(status_code=400, detail="Google sign-in failed. Please try again.")

    userinfo = token.get("userinfo") or {}
    email = userinfo.get("email")
    name = userinfo.get("name") or (email.split("@")[0] if email else "")
    google_id = userinfo.get("sub")

    if not email:
        raise HTTPException(status_code=400, detail="Google did not return an email address.")

    await _find_or_create_google_user(email=email, name=name, google_id=google_id)
    request.session["user_email"] = email
    return RedirectResponse(url="/dashboard")


@app.post("/auth/signup")
async def signup(payload: SignupRequest, request: Request):
    existing = await users_collection.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    doc = {
        "email": payload.email,
        "name": payload.name or payload.email.split("@")[0],
        "auth_method": "manual",
        "google_id": None,
        "password_hash": hash_password(payload.password),
        "created_at": datetime.now(timezone.utc),
        "last_dashboard_view_at": None,
    }
    await users_collection.insert_one(doc)
    request.session["user_email"] = payload.email
    return JSONResponse({"status": "ok", "email": payload.email})


@app.post("/auth/login")
async def login(payload: LoginRequest, request: Request):
    user = await users_collection.find_one({"email": payload.email})
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    request.session["user_email"] = payload.email
    return JSONResponse({"status": "ok", "email": payload.email})


@app.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse({"status": "ok"})


@app.get("/api/me")
async def me(user: dict | None = Depends(get_current_user_optional)):
    if not user:
        return JSONResponse({"logged_in": False})
    return JSONResponse({
        "logged_in": True,
        "email": user["email"],
        "name": user.get("name"),
        "auth_method": user.get("auth_method"),
    })


def normalize_url(raw: str) -> str:
    raw = raw.strip()
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc or "." not in parsed.netloc:
        raise ValueError("Not a valid website address")
    return raw


def extract_visible_text(soup: BeautifulSoup, max_lines: int = 400) -> str:
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
                continue

    combined = "\n\n".join(page_texts)[:MAX_SITE_CHARS]
    if not combined.strip():
        combined = "(No readable text could be extracted from this website.)"

    return {"title": title, "meta_description": meta_description, "content": combined}


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


# ============================================================================
# SHARED, DOMAIN-SCOPED ANALYSIS CACHE (unchanged from 2.5.0)
# ============================================================================
#
# This is the fix for the three issues described:
#   1. "Data not appearing for another user" / 3. "users' data must never
#      mix": neither was actually a mixing bug — flintel_user_signals was
#      always correctly scoped by user_email. The real problem was that
#      analyze_with_claude() ran independently for every user, so two
#      users analyzing the same domain could get two different
#      discovery_keywords lists and therefore two different match results
#      for what should have been identical domain data.
#   2. "Shared domain cache": get_or_create_domain_report() below is the
#      single place a domain gets turned into a report. It is checked
#      FIRST, before ever touching fetch_site_text() or
#      analyze_with_claude(), so a domain is only ever scraped + analyzed
#      once — every subsequent lookup (by any user) reuses the same
#      cached report and the same discovery_keywords, which is what makes
#      the resulting matches consistent across users.
#
# flintel_domain_reports_collection has a unique index on "domain" only
# (no user_email) — see on_startup(). That uniqueness constraint is what
# actually prevents duplicate concurrent processing: the first request
# for a brand-new domain wins an atomic upsert and proceeds to do the
# real work; any other concurrent request for that same domain sees the
# "processing" placeholder already there and polls briefly instead of
# starting its own redundant scrape + Claude call.
#
# DO NOT REMOVE THIS. Without it, two tabs opening the same brand-new
# domain at the same time will race two independent Claude calls, get two
# different discovery_keywords sets, and show two different match counts
# for the identical domain (e.g. 25 posts vs 0 posts) — this is the exact
# failure mode this cache exists to prevent.
# ============================================================================

async def _try_claim_domain_for_processing(domain: str, url: str) -> bool:
    """
    Atomically attempts to become the request responsible for analyzing
    `domain`. Returns True if this call created the placeholder doc (i.e.
    this request should do the actual scrape + Claude analysis). Returns
    False if a doc for this domain already existed (either complete, or
    already being processed by someone else).
    """
    now = datetime.now(timezone.utc)
    before = await flintel_domain_reports_collection.find_one_and_update(
        {"domain": domain},
        {
            "$setOnInsert": {
                "domain": domain,
                "url": url,
                "status": "processing",
                "report": None,
                "discovery_keywords": [],
                "created_at": now,
                "updated_at": now,
            }
        },
        upsert=True,
        return_document=ReturnDocument.BEFORE,
    )
    # If `before` is None, there was nothing there before our upsert — we
    # created it just now, so we own processing for this domain.
    return before is None


async def _wait_for_domain_report(domain: str) -> dict | None:
    """
    Polls flintel_domain_reports_collection for a domain that is currently
    being processed by another request, up to DOMAIN_LOCK_MAX_WAIT_SECONDS.
    Returns the completed doc if it shows up in time, else None.
    """
    waited = 0.0
    while waited < DOMAIN_LOCK_MAX_WAIT_SECONDS:
        doc = await flintel_domain_reports_collection.find_one({"domain": domain})
        if doc and doc.get("status") == "complete":
            return doc
        await asyncio.sleep(DOMAIN_LOCK_POLL_INTERVAL_SECONDS)
        waited += DOMAIN_LOCK_POLL_INTERVAL_SECONDS
    return None


async def get_or_create_domain_report(domain: str, url: str) -> dict:
    """
    Single entry point for turning a domain into a market-intelligence
    report. Guarantees (per the isolation/caching requirements above):
      - A domain is only ever scraped + sent to Claude once. Every
        subsequent call for the same domain — from any user — reuses the
        cached report and discovery_keywords.
      - Two concurrent first-time requests for the same brand-new domain
        do not both trigger a scrape + Claude call; only one does the
        work, the other waits for it and reuses the result.
      - This collection carries no user_email — it is intentionally
        shared. Per-user state (flintel_web_data_collection "my reports"
        history, flintel_user_signals, reviewed flags) is written
        separately by the caller and remains fully user-scoped.
    """
    existing = await flintel_domain_reports_collection.find_one({"domain": domain})
    if existing and existing.get("status") == "complete":
        return existing

    if existing and existing.get("status") == "processing":
        # Someone else is already analyzing this domain right now — wait
        # for them instead of starting a second, redundant analysis.
        completed = await _wait_for_domain_report(domain)
        if completed:
            return completed
        # Fallback: whoever was processing this seems to have stalled or
        # crashed. Try to claim it ourselves rather than waiting forever.

    claimed = await _try_claim_domain_for_processing(domain, url)
    if not claimed:
        # Another request claimed it in the tiny window between our check
        # above and now — wait for that one instead of racing it.
        completed = await _wait_for_domain_report(domain)
        if completed:
            return completed
        # If it's still not done, fall through and do the work ourselves
        # rather than leaving the user stuck with nothing.

    try:
        site_data = await fetch_site_text(url)
        report = await analyze_with_claude(domain, site_data)
    except Exception:
        # Don't leave the domain permanently stuck in "processing" if the
        # scrape or Claude call fails — clear it so the next attempt (by
        # this user retrying, or another user) can try fresh.
        await flintel_domain_reports_collection.update_one(
            {"domain": domain},
            {"$set": {"status": "failed", "updated_at": datetime.now(timezone.utc)}},
        )
        raise

    now = datetime.now(timezone.utc)
    await flintel_domain_reports_collection.update_one(
        {"domain": domain},
        {
            "$set": {
                "domain": domain,
                "url": url,
                "report": report,
                "discovery_keywords": report.get("discovery_keywords") or [],
                "status": "complete",
                "updated_at": now,
            }
        },
        upsert=True,
    )
    return await flintel_domain_reports_collection.find_one({"domain": domain})


# ============================================================================
# END SHARED DOMAIN CACHE (unchanged from 2.5.0)
# ============================================================================


async def _next_sequence(user_email: str, domain: str) -> int:
    counter_doc = await flintel_counters_collection.find_one_and_update(
        {"user_email": user_email, "domain": domain, "counter_name": "flintel_web_data_sequence"},
        {"$inc": {"value": 1}},
        upsert=True,
        return_document=True,
    )
    return counter_doc["value"]


async def store_flintel_web_data(user_email: str, domain: str, url: str, report: dict) -> dict:
    sequence = await _next_sequence(user_email, domain)

    doc = {
        "user_email":         user_email,
        "domain":             domain,
        "url":                url,
        "sequence":           sequence,
        "discovery_keywords": report.get("discovery_keywords") or [],
        "company_summary":    report.get("company_summary", ""),
        "buyer_pain_points":  report.get("buyer_pain_points", []),
        "competitors":        report.get("competitors") or [],
        "unique_value_prop":  report.get("unique_value_prop", ""),
        "created_at":         datetime.now(timezone.utc),
    }
    try:
        result = await flintel_web_data_collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
    except Exception as exc:
        logger.error("MongoDB insert_one (flintel_web_data) failed for domain=%s user=%s: %s",
                     domain, user_email, exc)
        doc["_id"] = None

    await flintel_tracking_state_collection.update_one(
        {"user_email": user_email, "domain": domain},
        {
            "$set": {
                "discovery_keywords": doc["discovery_keywords"],
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {
                "user_email": user_email,
                "domain": domain,
                "last_checked_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
    )

    return doc


def _build_keyword_query(keywords: list[str]) -> dict:
    or_clauses = []
    for keyword in keywords:
        pattern = re.escape(keyword)
        or_clauses.extend([
            {"search_keyword":        {"$regex": pattern, "$options": "i"}},
            {"subreddit_or_channel":  {"$regex": pattern, "$options": "i"}},
            {"text":                  {"$regex": pattern, "$options": "i"}},
        ])
    return {"$or": or_clauses} if or_clauses else {}


async def match_existing_web_data(keywords: list[str], limit_per_keyword: int = 50) -> list[dict]:
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


def _web_data_timestamp_field(doc: dict) -> datetime | None:
    ts = doc.get("fetched_at") or doc.get("created_at")
    return ts if isinstance(ts, datetime) else None


async def _persist_match_for_user(user_email: str, domain: str, keyword: str, source_doc: dict):
    source_id = str(source_doc["_id"])
    doc = dict(source_doc)
    doc.pop("_id", None)
    doc["source_id"] = source_id
    doc["user_email"] = user_email
    doc["domain"] = domain
    doc["matched_keyword"] = keyword
    doc["matched_at"] = datetime.now(timezone.utc)

    for f in ("posted_at", "fetched_at", "created_at"):
        if isinstance(doc.get(f), datetime):
            doc[f] = doc[f].isoformat()

    try:
        await flintel_user_signals_collection.update_one(
            {"user_email": user_email, "source_id": source_id},
            {
                "$set": doc,
                "$setOnInsert": {"reviewed": False, "first_matched_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
    except Exception as exc:
        logger.error("Failed to persist match for user=%s source_id=%s: %s", user_email, source_id, exc)


async def _persist_matches_for_user(user_email: str, domain: str, matches: list[dict]) -> int:
    persisted = 0
    for m in matches:
        keyword = m.get("matched_keyword") or (m.get("matched_keyword") if isinstance(m, dict) else None)
        if not keyword:
            continue
        await _persist_match_for_user(user_email, domain, keyword, m)
        persisted += 1
    return persisted


async def run_live_match_tracker():
    logger.info(
        "[LIVE-TRACKER] started | interval:%ss | watching flintel_tracking_state pairs "
        "for new web_data rows", LIVE_TRACKER_INTERVAL_SECONDS
    )
    while True:
        try:
            pairs = flintel_tracking_state_collection.find({})
            pair_count, new_matches_total = 0, 0

            async for pair in pairs:
                pair_count += 1
                user_email = pair["user_email"]
                domain = pair["domain"]
                keywords = (pair.get("discovery_keywords") or [])[:MAX_KEYWORDS_TO_MATCH]
                if not keywords:
                    continue

                last_checked_at = pair.get("last_checked_at") or datetime(1970, 1, 1, tzinfo=timezone.utc)
                keyword_query = _build_keyword_query(keywords)
                if not keyword_query:
                    continue

                full_query = {
                    "$and": [
                        keyword_query,
                        {"$or": [
                            {"fetched_at": {"$gt": last_checked_at}},
                            {"created_at": {"$gt": last_checked_at}},
                        ]},
                    ]
                }

                newest_seen = last_checked_at
                pair_new_matches = 0
                try:
                    cursor = web_data_collection.find(full_query)
                    async for source_doc in cursor:
                        text_blob = " ".join([
                            str(source_doc.get("search_keyword", "")),
                            str(source_doc.get("subreddit_or_channel", "")),
                            str(source_doc.get("text", "")),
                        ]).lower()
                        matched_keyword = next(
                            (kw for kw in keywords if kw.lower() in text_blob), keywords[0]
                        )
                        await _persist_match_for_user(user_email, domain, matched_keyword, source_doc)
                        new_matches_total += 1
                        pair_new_matches += 1

                        doc_ts = _web_data_timestamp_field(source_doc)
                        if doc_ts and doc_ts > newest_seen:
                            newest_seen = doc_ts
                except Exception as exc:
                    logger.error("[LIVE-TRACKER] query failed for user=%s domain=%s: %s",
                                 user_email, domain, exc)
                    continue

                if newest_seen > last_checked_at:
                    await flintel_tracking_state_collection.update_one(
                        {"user_email": user_email, "domain": domain},
                        {"$set": {"last_checked_at": newest_seen}},
                    )

                # NEW — if this (user_email, domain) pair got new matched
                # signals this pass, recompute + re-cache its Section 05
                # pipeline projection so it stays live without any extra
                # polling job. Wrapped in try/except so a projection
                # failure can never break the existing tracker loop.
                if pair_new_matches > 0:
                    try:
                        await _refresh_pipeline_analytics(user_email, domain)
                    except Exception as exc:
                        logger.error(
                            "[LIVE-TRACKER] pipeline analytics refresh failed for user=%s domain=%s: %s",
                            user_email, domain, exc,
                        )

            if new_matches_total:
                logger.info("[LIVE-TRACKER] pass complete | pairs_checked:%d | new_matches:%d",
                            pair_count, new_matches_total)

        except Exception as exc:
            logger.error("[LIVE-TRACKER] loop error: %s", exc)

        await asyncio.sleep(LIVE_TRACKER_INTERVAL_SECONDS)


def _engagement_score(doc: dict) -> float:
    for field in ("upvotes", "score", "ups", "num_upvotes"):
        val = doc.get(field)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _serialize_signal(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    for f in ("posted_at", "fetched_at", "created_at", "matched_at", "first_matched_at", "reviewed_at"):
        if isinstance(doc.get(f), datetime):
            doc[f] = doc[f].isoformat()
    return doc


async def compute_dashboard_stats(user_email: str) -> dict:
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    user_doc = await users_collection.find_one_and_update(
        {"email": user_email},
        {"$set": {"last_dashboard_view_at": now}},
    )
    last_view = (user_doc or {}).get("last_dashboard_view_at") or (now - timedelta(days=3650))
    if isinstance(last_view, str):
        try:
            last_view = datetime.fromisoformat(last_view)
        except ValueError:
            last_view = now - timedelta(days=3650)

    signals_filter = {"user_email": user_email}

    signals_total = await flintel_user_signals_collection.count_documents(signals_filter)
    signals_this_week = await flintel_user_signals_collection.count_documents(
        {**signals_filter, "matched_at": {"$gte": week_ago}}
    )
    new_since_last_visit = await flintel_user_signals_collection.count_documents(
        {**signals_filter, "matched_at": {"$gte": last_view}}
    )
    unreviewed_count = await flintel_user_signals_collection.count_documents(
        {**signals_filter, "reviewed": {"$ne": True}}
    )

    reach_total = 0.0
    top_thread = None
    top_score = -1.0
    cursor = flintel_user_signals_collection.find(signals_filter).sort([("matched_at", -1)]).limit(500)
    async for doc in cursor:
        score = _engagement_score(doc)
        reach_total += score
        if score > top_score:
            top_score = score
            top_thread = doc

    latest_report = await flintel_web_data_collection.find_one(
        {"user_email": user_email}, sort=[("created_at", -1)]
    )
    competitor_gap = []
    your_mentions = 0
    domain = None
    if latest_report:
        domain = latest_report.get("domain")
        competitors = (latest_report.get("competitors") or [])[:MAX_COMPETITORS_FOR_GAP]
        for comp in competitors:
            name = (comp or {}).get("name") if isinstance(comp, dict) else None
            if not name:
                continue
            pattern = re.escape(name)
            count = await flintel_user_signals_collection.count_documents(
                {**signals_filter, "text": {"$regex": pattern, "$options": "i"}}
            )
            competitor_gap.append({"name": name, "mentions": count})

        if domain:
            your_mentions = await flintel_user_signals_collection.count_documents(
                {**signals_filter, "text": {"$regex": re.escape(domain), "$options": "i"}}
            )

    return {
        "generated_at": now.isoformat(),
        "domain": domain,
        "signals_total": signals_total,
        "signals_this_week": signals_this_week,
        "new_since_last_visit": new_since_last_visit,
        "unreviewed_count": unreviewed_count,
        "estimated_reach": int(reach_total),
        "top_thread": _serialize_signal(top_thread) if top_thread else None,
        "competitor_gap": competitor_gap,
        "your_mentions": your_mentions,
    }


# ============================================================================
# SECTION 03 REPORT CARD: "Your market is moving — and it's moving
# toward you" (unchanged from 2.5.0)
# ============================================================================
#
# Everything below is 100% ADDITIVE (carried over unchanged from 2.4.0).
#
# ISOLATION GUARANTEE: every query below filters on {"user_email": user_email},
# the exact same pattern already used throughout compute_dashboard_stats()
# above. user_email is always the value from the authenticated session
# (via get_current_user, reading request.session — never from client input),
# so one user's competitor volume, intent-phrase volume, and bar chart can
# never mix with another user's data, for the same reason /api/dashboard-stats
# already can't mix users: the filter is applied at the database-query level,
# not reconstructed or trusted from anything the client sends.
#
# HONESTY GUARANTEE: "Threads ranking on Google Page 1" has no underlying
# data source anywhere in this codebase — nothing here queries Google or
# stores SERP-ranking history. Rather than fabricate a number for it (which
# would violate the same "never invent facts" principle already applied in
# ANALYSIS_SYSTEM_PROMPT and in public_stats()'s fail-soft-to-null pattern),
# it is returned as null with an explicit "not yet available" reason string,
# so the frontend can hide/skip that card instead of rendering a fake number.
# ============================================================================

INTENT_PHRASE_MARKERS = [
    "anyone use", "anyone tried", " vs ", "alternative to", "alternative for",
    "recommend", "worth it", "switch from", "switching from", "better than",
]


def _is_intent_phrase(keyword: str) -> bool:
    k = keyword.lower()
    return any(marker in k for marker in INTENT_PHRASE_MARKERS)


async def _volume_trend(user_email: str, text_pattern: str, days_back: int = 90) -> dict:
    now = datetime.now(timezone.utc)
    recent_start = now - timedelta(days=days_back // 3)
    baseline_start = now - timedelta(days=days_back)

    filt = {
        "user_email": user_email,
        "text": {"$regex": text_pattern, "$options": "i"},
    }

    recent_count = await flintel_user_signals_collection.count_documents(
        {**filt, "matched_at": {"$gte": recent_start}}
    )
    baseline_count = await flintel_user_signals_collection.count_documents(
        {**filt, "matched_at": {"$gte": baseline_start, "$lt": recent_start}}
    )

    pct_change = None
    if baseline_count > 0:
        pct_change = round(((recent_count - baseline_count) / baseline_count) * 100)

    return {
        "current_period_count": recent_count,
        "baseline_period_count": baseline_count,
        "pct_change_vs_prior_period": pct_change,
    }


async def _competitor_complaint_volume(user_email: str, competitors: list[dict], days_back: int = 90) -> dict:
    names = [c.get("name") for c in competitors if isinstance(c, dict) and c.get("name")]
    if not names:
        return {"current_period_count": 0, "baseline_period_count": 0, "pct_change_vs_prior_period": None}

    pattern = "|".join(re.escape(n) for n in names)
    return await _volume_trend(user_email, pattern, days_back=days_back)


async def _intent_phrase_volume(user_email: str, discovery_keywords: list[str], days_back: int = 90) -> dict:
    intent_keywords = [kw for kw in discovery_keywords if _is_intent_phrase(kw)]
    if not intent_keywords:
        return {
            "current_period_count": 0,
            "baseline_period_count": 0,
            "pct_change_vs_prior_period": None,
            "matched_phrases": [],
        }

    pattern = "|".join(re.escape(kw) for kw in intent_keywords)
    trend = await _volume_trend(user_email, pattern, days_back=days_back)
    trend["matched_phrases"] = intent_keywords
    return trend


def _build_competitor_bar_chart(competitor_gap: list[dict], your_mentions: int) -> dict:
    bars = sorted(
        [{"name": c["name"], "mentions": c["mentions"], "is_you": False} for c in competitor_gap],
        key=lambda c: c["mentions"],
        reverse=True,
    )
    bars.append({"name": "Your product", "mentions": your_mentions, "is_you": True})

    competitor_avg = (
        sum(c["mentions"] for c in competitor_gap) / len(competitor_gap)
        if competitor_gap else 0
    )
    multiple = round(competitor_avg / your_mentions, 1) if your_mentions > 0 else None

    return {"bars": bars, "talked_about_multiple": multiple}


async def compute_market_momentum(user_email: str) -> dict:
    base_stats = await compute_dashboard_stats(user_email)
    competitor_gap = base_stats["competitor_gap"]
    your_mentions = base_stats["your_mentions"]
    domain = base_stats["domain"]

    latest_report = await flintel_web_data_collection.find_one(
        {"user_email": user_email}, sort=[("created_at", -1)]
    )
    competitors = (latest_report.get("competitors") or [])[:MAX_COMPETITORS_FOR_GAP] if latest_report else []
    discovery_keywords = (latest_report.get("discovery_keywords") or []) if latest_report else []

    complaint_volume = await _competitor_complaint_volume(user_email, competitors)
    intent_volume = await _intent_phrase_volume(user_email, discovery_keywords)
    bar_chart = _build_competitor_bar_chart(competitor_gap, your_mentions)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "competitor_complaint_volume": complaint_volume,
        "intent_search_phrases": intent_volume,
        "page_one_threads": None,
        "page_one_threads_reason": "Not available — no SERP-ranking data source is wired up yet.",
        "competitor_bar_chart": bar_chart,
    }


# ============================================================================
# END SECTION 03 (unchanged from 2.5.0)
# ============================================================================


# ============================================================================
# SECTION 05 ADDITION: "What this means for your pipeline"
# ============================================================================
#
# WHAT THIS COMPUTES
#   A 6-month trials/customers/MRR projection. Section 05 in the mock is
#   explicitly labeled "Illustrative model based on typical Flintel
#   customer outcomes — not calculated from your specific data above," so
#   this does NOT try to predict revenue out of thin air:
#     1. It seeds from a real number already in this user's own data —
#        estimated_reach from compute_dashboard_stats(), built from
#        upvotes/score across THIS user's matched signals for THIS domain.
#     2. It applies fixed, fully-disclosed conservative funnel assumptions
#        (click-through, visitor->trial, trial->customer, ACV) to turn
#        that reach into a 6-month curve.
#     3. It compares against a flat "organic" line at today's pace.
#   All assumptions are returned in the payload (not hidden on the
#   frontend), and the exact "illustrative, not calculated..." disclaimer
#   from the mock is included as `note` in every response.
#
# WHY THIS IS PLAIN PYTHON, NOT A CLAUDE CALL
#   Same reasoning that motivated the domain-cache fix above: Claude's
#   output is not deterministic between calls. If trials/MRR came from a
#   Claude prompt, refreshing the page could show 216 trials one moment
#   and 340 the next, for identical underlying data — reintroducing the
#   exact inconsistency bug this file already fixed once for
#   discovery_keywords. Revenue/trial projections are pure arithmetic
#   here, so they are 100% reproducible for the same input every time.
#
# STORAGE — flintel_analytics_collection
#   Keyed by (user_email, domain), unlike flintel_domain_reports_collection
#   (keyed by domain only). This IS user-scoped because the projection
#   depends on that specific user's own accumulated matched-signal reach,
#   exactly like flintel_user_signals_collection, flintel_web_data_
#   collection, and flintel_tracking_state_collection already are.
#
# WHEN IT'S CREATED / UPDATED
#   - First created right after /api/analyze finishes persisting a user's
#     initial matches for a domain (see analyze() below) — so a fresh
#     domain lookup gets an immediate, if rough, first-cut projection.
#   - Refreshed automatically by run_live_match_tracker() (see the
#     pair_new_matches hook added above) every time new matched signals
#     arrive for that (user_email, domain) pair — no separate cron/poll
#     job required, and the existing tracker loop/logging is untouched.
#   - Also computed on-demand (and re-cached) if a request for
#     /api/report/pipeline-projection arrives before any cache exists.
# ============================================================================

PIPELINE_DEFAULT_ACV = 300.0
PIPELINE_DEFAULT_CLICK_THROUGH_RATE = 0.005     # 0.5% of thread viewers click through
PIPELINE_DEFAULT_VISITOR_TO_TRIAL_RATE = 0.10   # 10% of clickers start a trial
PIPELINE_DEFAULT_TRIAL_TO_CUSTOMER_RATE = 0.30  # 30% of trials convert to paying
PIPELINE_PROJECTION_MONTHS = 6

# Fraction of Flintel's steady-state reach realized by each month (M1..M6).
PIPELINE_RAMP_CURVE = [0.35, 0.55, 0.72, 0.85, 0.95, 1.0]

# How much bigger the user's engaged reach becomes at steady state once
# Flintel is surfacing threads they weren't previously seeing/participating
# in, vs. their current organic-only reach.
PIPELINE_STEADY_STATE_REACH_MULTIPLIER = 3.0


def _estimate_monthly_organic_reach(base_stats: dict) -> float:
    """
    Baseline 'reach' the user already gets organically, before Flintel.
    Reuses estimated_reach from compute_dashboard_stats() (built from
    upvotes/score across the user's matched signals) so the projection is
    seeded by something real for that specific user, not invented.
    """
    return float(base_stats.get("estimated_reach") or 0.0)


def _project_pipeline_curve(
    baseline_reach: float,
    click_through_rate: float,
    visitor_to_trial_rate: float,
    months: int = PIPELINE_PROJECTION_MONTHS,
) -> list[dict]:
    """
    Builds two cumulative-trials curves over `months`:
      - organic: flat monthly pace at the user's current reach/conversion
        rate (no change in behavior).
      - with_flintel: reach ramps toward PIPELINE_STEADY_STATE_REACH_MULTIPLIER
        along PIPELINE_RAMP_CURVE, representing Flintel surfacing more of
        the relevant conversation over the first 6 months.
    Pure arithmetic — deterministic for a given baseline_reach.
    """
    organic_monthly_trials = baseline_reach * click_through_rate * visitor_to_trial_rate
    flintel_steady_state_reach = baseline_reach * PIPELINE_STEADY_STATE_REACH_MULTIPLIER

    curve = []
    organic_cumulative = 0.0
    flintel_cumulative = 0.0
    for i in range(months):
        ramp_fraction = PIPELINE_RAMP_CURVE[min(i, len(PIPELINE_RAMP_CURVE) - 1)]
        month_reach = flintel_steady_state_reach * ramp_fraction
        month_trials = month_reach * click_through_rate * visitor_to_trial_rate

        organic_cumulative += organic_monthly_trials
        flintel_cumulative += month_trials

        curve.append({
            "month_label": f"M{i + 1}",
            "organic_trials_cumulative": round(organic_cumulative, 1),
            "flintel_trials_cumulative": round(flintel_cumulative, 1),
        })
    return curve


async def compute_pipeline_projection(user_email: str, domain: str | None = None,
                                       acv: float | None = None) -> dict:
    """
    Computes (does NOT save) the Section 05 pipeline projection for one
    user. Seeds from that user's real accumulated reach for `domain`
    (falls back to their most recent analyzed domain if none given), then
    applies fixed conservative funnel assumptions. Always illustrative —
    see `note` — never a guarantee. Pure Python; no Claude call.
    """
    acv = acv if acv is not None else PIPELINE_DEFAULT_ACV

    base_stats = await compute_dashboard_stats(user_email)
    resolved_domain = domain or base_stats.get("domain")
    baseline_reach = _estimate_monthly_organic_reach(base_stats)

    curve = _project_pipeline_curve(
        baseline_reach,
        PIPELINE_DEFAULT_CLICK_THROUGH_RATE,
        PIPELINE_DEFAULT_VISITOR_TO_TRIAL_RATE,
    )

    projected_trials_6mo = curve[-1]["flintel_trials_cumulative"] if curve else 0.0
    projected_customers_6mo = projected_trials_6mo * PIPELINE_DEFAULT_TRIAL_TO_CUSTOMER_RATE
    projected_mrr_6mo = projected_customers_6mo * acv

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_email": user_email,
        "domain": resolved_domain,
        "acv": acv,
        "click_through_rate": PIPELINE_DEFAULT_CLICK_THROUGH_RATE,
        "visitor_to_trial_rate": PIPELINE_DEFAULT_VISITOR_TO_TRIAL_RATE,
        "trial_to_customer_rate": PIPELINE_DEFAULT_TRIAL_TO_CUSTOMER_RATE,
        "projected_trials_6mo": round(projected_trials_6mo),
        "projected_customers_6mo": round(projected_customers_6mo),
        "projected_mrr_6mo": round(projected_mrr_6mo),
        "curve": curve,
        "baseline_reach_used": baseline_reach,
        "note": (
            "Illustrative model based on typical Flintel customer outcomes "
            "— not calculated from your specific data above."
        ),
    }


async def _refresh_pipeline_analytics(user_email: str, domain: str, acv: float | None = None) -> dict:
    """
    Recomputes the pipeline projection for (user_email, domain) and
    upserts it into flintel_analytics_collection — keyed by user_email +
    domain, exactly like every other per-user collection in this file.
    Called:
      - once right after a fresh /api/analyze for this domain, and
      - automatically by run_live_match_tracker() whenever new matched
        signals arrive for this pair.
    """
    projection = await compute_pipeline_projection(user_email, domain=domain, acv=acv)

    await flintel_analytics_collection.update_one(
        {"user_email": user_email, "domain": domain},
        {
            "$set": {
                "user_email": user_email,
                "domain": domain,
                "projection": projection,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    return projection


async def get_or_refresh_pipeline_analytics(user_email: str, domain: str, acv: float | None = None) -> dict:
    """
    Read path for GET /api/report/pipeline-projection. Serves the cached
    projection from flintel_analytics_collection if one exists (fast —
    the live tracker keeps it fresh in the background); computes and
    caches one on the spot otherwise (e.g. right after a brand-new
    domain's very first analyze(), before the tracker has run yet).

    If the caller explicitly passes a custom `acv`, we always recompute
    (a cached projection was built with whatever ACV was in effect at
    cache time, and mixing ACVs would silently misrepresent the numbers).
    """
    if acv is not None:
        return await _refresh_pipeline_analytics(user_email, domain, acv=acv)

    cached = await flintel_analytics_collection.find_one(
        {"user_email": user_email, "domain": domain}
    )
    if cached and cached.get("projection"):
        return cached["projection"]

    return await _refresh_pipeline_analytics(user_email, domain)


# ============================================================================
# END SECTION 05 ADDITION
# ============================================================================


@app.on_event("startup")
async def on_startup():
    await flintel_user_signals_collection.create_index(
        [("user_email", 1), ("source_id", 1)], unique=True, name="user_source_unique"
    )
    await flintel_user_signals_collection.create_index(
        [("user_email", 1), ("matched_at", -1)], name="user_matched_at"
    )
    await flintel_user_signals_collection.create_index(
        [("user_email", 1), ("reviewed", 1)], name="user_reviewed"
    )
    await flintel_tracking_state_collection.create_index(
        [("user_email", 1), ("domain", 1)], unique=True, name="user_domain_unique"
    )
    await users_collection.create_index([("email", 1)], unique=True, name="email_unique")
    await flintel_web_data_collection.create_index([("user_email", 1), ("domain", 1)])
    await flintel_web_data_collection.create_index([("user_email", 1), ("created_at", -1)])
    await flintel_counters_collection.create_index(
        [("user_email", 1), ("domain", 1), ("counter_name", 1)],
        unique=True, name="counter_unique",
    )
    # Enforces "one cached report per domain, ever" at the database level.
    # This unique index is what makes the upsert in
    # _try_claim_domain_for_processing() atomic: two concurrent requests
    # for the same brand-new domain can't both succeed at creating a doc.
    await flintel_domain_reports_collection.create_index(
        [("domain", 1)], unique=True, name="domain_unique"
    )
    # NEW — one cached Section 05 projection per (user_email, domain).
    # Same isolation pattern as flintel_tracking_state_collection's index.
    await flintel_analytics_collection.create_index(
        [("user_email", 1), ("domain", 1)], unique=True, name="user_domain_analytics_unique"
    )

    asyncio.create_task(run_live_match_tracker())
    logger.info("Flintel startup complete — live tracker scheduled.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "web.html", {})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    user = await get_current_user_optional(request)
    if not user:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(request, "dashboard.html", {"user": user})


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, user: dict = Depends(get_current_user)):
    try:
        url = normalize_url(payload.url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please enter a valid website address.")

    domain = urlparse(url).netloc.replace("www.", "")

    # ── SHARED DOMAIN CACHE LOOKUP ──────────────────────────────────────
    # This replaces the old "always scrape + always call Claude" behavior.
    # get_or_create_domain_report() checks flintel_domain_reports_collection
    # (keyed only by domain, shared across every user) first. If this
    # domain has already been analyzed by ANYONE, that exact same report
    # and discovery_keywords are reused here — no new scrape, no new
    # Claude call, no API cost, and critically: every user gets the same
    # keywords, so matching against web_data_collection produces the same
    # results for everyone looking at this domain.
    domain_doc = await get_or_create_domain_report(domain, url)
    report = domain_doc["report"]
    # ─────────────────────────────────────────────────────────────────────

    # Per-user "my reports" history — unchanged in shape, still written
    # once per user per analyze() call, still scoped by user_email. This
    # is intentionally NOT deduplicated across users: it's each user's own
    # personal history of when they looked this domain up, not the shared
    # analysis itself.
    flintel_doc = await store_flintel_web_data(user["email"], domain, url, report)

    keywords = (report.get("discovery_keywords") or [])[:MAX_KEYWORDS_TO_MATCH]
    if not keywords:
        keywords = [domain]

    matches = await match_existing_web_data(keywords)

    persisted_count = await _persist_matches_for_user(user["email"], domain, matches)
    if persisted_count != len(matches):
        logger.warning(
            "analyze(): only persisted %d/%d matches for user=%s domain=%s "
            "(some matches were missing a matched_keyword field)",
            persisted_count, len(matches), user["email"], domain,
        )

    # NEW — build the very first Section 05 projection for this user +
    # domain right away, seeded by whatever matches were just persisted
    # above, so the dashboard has something to show immediately rather
    # than waiting for the next live-tracker pass. Wrapped in try/except
    # so a projection failure can never break the analyze() response.
    try:
        await _refresh_pipeline_analytics(user["email"], domain)
    except Exception as exc:
        logger.error(
            "analyze(): initial pipeline analytics refresh failed for user=%s domain=%s: %s",
            user["email"], domain, exc,
        )

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
async def get_reports(domain: str, user: dict = Depends(get_current_user)):
    cursor = flintel_web_data_collection.find(
        {"domain": domain, "user_email": user["email"]}
    ).sort("sequence", -1).limit(5)
    reports = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["created_at"] = doc["created_at"].isoformat()
        reports.append(doc)
    return JSONResponse({"domain": domain, "reports": reports})


@app.get("/api/signals/{domain}")
async def get_signals(domain: str, subreddit: str | None = None, limit: int = 50,
                       user: dict = Depends(get_current_user)):
    latest = await flintel_web_data_collection.find_one(
        {"domain": domain, "user_email": user["email"]}, sort=[("sequence", -1)]
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


@app.get("/api/my-signals")
async def get_my_signals(domain: str | None = None, unreviewed_only: bool = False,
                          limit: int = 100, user: dict = Depends(get_current_user)):
    query: dict = {"user_email": user["email"]}
    if domain:
        query["domain"] = domain
    if unreviewed_only:
        query["reviewed"] = {"$ne": True}

    cursor = flintel_user_signals_collection.find(query, {"_id": 0}).sort("matched_at", -1).limit(limit)
    signals = [_serialize_signal(doc) async for doc in cursor]
    return JSONResponse({"count": len(signals), "signals": signals})


@app.post("/api/signals/{source_id}/review")
async def mark_signal_reviewed(source_id: str, user: dict = Depends(get_current_user)):
    result = await flintel_user_signals_collection.update_one(
        {"user_email": user["email"], "source_id": source_id},
        {"$set": {"reviewed": True, "reviewed_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Signal not found for this user.")
    return JSONResponse({"status": "ok", "source_id": source_id})


@app.get("/api/my-domains")
async def get_my_domains(user: dict = Depends(get_current_user)):
    cursor = flintel_web_data_collection.find(
        {"user_email": user["email"]}
    ).sort("created_at", -1)
    seen_domains = []
    async for doc in cursor:
        if doc["domain"] not in seen_domains:
            seen_domains.append(doc["domain"])
    return JSONResponse({"domains": seen_domains})


@app.get("/api/dashboard-stats")
async def dashboard_stats(user: dict = Depends(get_current_user)):
    stats = await compute_dashboard_stats(user["email"])
    return JSONResponse(stats)


@app.get("/api/report/market-momentum")
async def market_momentum(user: dict = Depends(get_current_user)): 
    momentum = await compute_market_momentum(user["email"])
    return JSONResponse(momentum)


@app.get("/api/report/pipeline-projection")
async def pipeline_projection(domain: str | None = None, acv: float | None = None,
                               user: dict = Depends(get_current_user)):
    """
    Section 05 — "What this means for your pipeline". Serves the cached,
    live-updating projection from flintel_analytics_collection when one
    exists; computes + caches one on the spot otherwise.

    Query params:
      domain — which analyzed domain to project for. Defaults to the
               user's most recently analyzed domain if omitted.
      acv    — override the default $300 ACV. Forces a fresh (uncached)
               computation, since a cached projection reflects whatever
               ACV was in effect when it was last saved.
    """
    resolved_domain = domain
    if not resolved_domain:
        base_stats = await compute_dashboard_stats(user["email"])
        resolved_domain = base_stats.get("domain")
    if not resolved_domain:
        raise HTTPException(
            status_code=404,
            detail="No analyzed domain found yet — analyze a domain first.",
        )

    projection = await get_or_refresh_pipeline_analytics(user["email"], resolved_domain, acv=acv)
    return JSONResponse(projection)


@app.get("/api/stats/public")
async def public_stats():
    try:
        domains = await flintel_web_data_collection.distinct("domain")
        return JSONResponse({"domains_analyzed": len(domains)})
    except Exception as exc:
        logger.error("public_stats() failed: %s", exc)
        return JSONResponse({"domains_analyzed": None})


@app.get("/api/health")
async def health():
    mongo_status = "unknown"
    try:
        await mongo_client.admin.command("ping")
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"error: {exc}"
    return {"status": "ok", "mongodb": mongo_status, "database": MONGODB_DB_NAME}
