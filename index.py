"""
Flintel — Website Market Intelligence + Reddit Signal Tracking
----------------------------------------------------------------
FastAPI backend that:
  1. Lets a user sign in — either "Continue with Google" (OAuth 2.0) or a
     manual email + password account. Both paths land the user in the
     exact same logged-in session; downstream code never cares which one
     was used.
  2. Accepts a domain/URL from a LOGGED-IN user.
  3. Fetches and cleans the text of that website (homepage + a few likely
     secondary pages such as /about and /pricing).
  4. Sends that text to Claude with a carefully engineered prompt that
     returns a structured market-intelligence report: what the company
     does, who it's for, what pain points its buyers talk about, who its
     competitors are, and where to reach them.
  5. Using the discovery_keywords from that report, matches against the
     EXISTING, pre-populated `web_data` collection (Reddit/Twitter signals
     collected by the separate background worker service) — read-only,
     on-demand, exactly as before.
  6. Persists every website report AND every matched signal, PERMANENTLY
     TIED TO THE LOGGED-IN USER'S EMAIL — so if the user logs back in 10
     days later, all of their domains and all of their matches are still
     there, nothing is lost, nothing needs to be re-analyzed.
  7. NEW — runs a continuous background loop (independent of any user
     being online) that watches the `web_data` collection for genuinely
     NEW rows and, for every logged-in user's saved keyword set, checks
     if the new row matches. Any match is written into that user's own
     permanent `flintel_user_signals` collection immediately — so by the
     time a user logs back in, their dashboard is already up to date,
     with zero live querying needed at page-load time.
  8. Serves a single-page UI (templates/web.html) that drives the whole
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

Required environment variables (new, for auth):
    GOOGLE_CLIENT_ID       - from Google Cloud Console OAuth credentials
    GOOGLE_CLIENT_SECRET   - from Google Cloud Console OAuth credentials
    GOOGLE_REDIRECT_URI    - e.g. http://localhost:8000/auth/google/callback
    SESSION_SECRET_KEY     - any long random string, signs the session cookie
"""

import asyncio
import json
import logging
import os
import re
import secrets
from datetime import datetime, timezone
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
from starlette.middleware.sessions import SessionMiddleware

from authlib.integrations.starlette_client import OAuth

load_dotenv()

logger = logging.getLogger("flintel")
logging.basicConfig(level=logging.INFO)

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
#   fx_signals.users                -> login accounts (Google + manual)
#   fx_signals.flintel_web_data     -> Claude reports, one row per analysis,
#                                       tied to the user who ran it
#   fx_signals.flintel_user_signals -> permanently-saved matches per user,
#                                       kept up to date by the live tracker
#   fx_signals.flintel_tracking_state -> internal bookkeeping: per
#                                       (user_email, domain), what's the
#                                       newest web_data row we've already
#                                       checked, so the tracker never
#                                       re-scans the whole collection
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "fx_signals")

SECONDARY_PATHS = ["/about", "/about-us", "/product", "/products", "/pricing", "/features"]
MAX_SITE_CHARS = 12000          # cap on scraped text sent to Claude
REQUEST_TIMEOUT_SECONDS = 15.0

# How many keywords (max) from Claude's discovery_keywords to actually
# match against the existing web_data collection, to keep queries bounded.
MAX_KEYWORDS_TO_MATCH = 6

# How often the live background tracker wakes up to look for new web_data
# rows and match them against every user's saved keywords.
LIVE_TRACKER_INTERVAL_SECONDS = int(os.getenv("LIVE_TRACKER_INTERVAL_SECONDS", "120"))

if not ANTHROPIC_API_KEY:
    # Fail loudly at startup rather than on the first request.
    print("WARNING: ANTHROPIC_API_KEY is not set. /api/analyze will fail until it is.")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]

users_collection             = db["users"]
flintel_web_data_collection  = db["flintel_web_data"]
web_data_collection          = db["web_data"]                # existing, read-only
flintel_user_signals_collection = db["flintel_user_signals"] # NEW — permanent, per-user
flintel_tracking_state_collection = db["flintel_tracking_state"]  # NEW — internal bookkeeping

app = FastAPI(title="Flintel — Website Market Intelligence", version="2.0.0")
templates = Jinja2Templates(directory="templates")

# Serve static assets (favicon.ico, logo, css/js if ever split out) from a
# top-level "static" folder, sibling to "templates". Mounted as "static" so
# templates can reference files via {{ request.url_for('static', path=...) }}.
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------------
# Session cookie middleware — this is what makes "logged in" possible.
# Starlette signs the cookie with SESSION_SECRET_KEY so the user can't
# tamper with it; nothing sensitive (like a password) is ever stored in it,
# only the user's email once they've successfully authenticated.
# --------------------------------------------------------------------------

SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
if not SESSION_SECRET_KEY:
    print("WARNING: SESSION_SECRET_KEY not set — using a random one-off key "
          "(sessions will NOT survive a server restart). Set this in .env for production.")
    SESSION_SECRET_KEY = secrets.token_urlsafe(32)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, same_site="lax")

# --------------------------------------------------------------------------
# Google OAuth client (authlib) — "Continue with Google"
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# Manual (email + password) auth — bcrypt hashing via passlib
# --------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


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


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


# --------------------------------------------------------------------------
# Auth helpers — everything downstream just asks "who is logged in?"
# It never cares whether they came in through Google or manually.
# --------------------------------------------------------------------------

async def get_current_user(request: Request) -> dict:
    """
    Reads the signed session cookie for a user_email, then loads the full
    user document from MongoDB. Raises 401 if nobody is logged in. Use as
    a FastAPI dependency on any route that requires a logged-in user.
    """
    email = request.session.get("user_email")
    if not email:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    user = await users_collection.find_one({"email": email})
    if not user:
        # Session cookie refers to a user that no longer exists — clear it.
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    user["_id"] = str(user["_id"])
    return user


async def get_current_user_optional(request: Request) -> dict | None:
    """Same as get_current_user, but returns None instead of raising."""
    email = request.session.get("user_email")
    if not email:
        return None
    user = await users_collection.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
    return user


async def _find_or_create_google_user(email: str, name: str, google_id: str) -> dict:
    """
    $setOnInsert-style upsert: if this Google account has signed in before
    (matched by email), we NEVER overwrite their existing document (so a
    manual-signup user who later also uses "Continue with Google" with the
    same email keeps their original account, just gains a linked google_id).
    A brand-new email gets a fresh account created with auth_method="google".
    """
    now = datetime.now(timezone.utc)
    existing = await users_collection.find_one({"email": email})
    if existing:
        # Link the google_id onto the existing account if not already linked.
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
    }
    result = await users_collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------

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

    # This is the entire "login" step: put the email in the signed session
    # cookie. Every other route just reads this back via get_current_user().
    request.session["user_email"] = email
    return RedirectResponse(url="/")


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


# --------------------------------------------------------------------------
# Helpers: fetching & cleaning website content  (unchanged)
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
# Claude analysis (website -> market-intelligence report)  (unchanged)
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
# Store Claude's per-domain output — NOW TIED TO THE LOGGED-IN USER — then
# match it against the EXISTING pre-populated web_data collection exactly
# as before (on-demand, read-only, unchanged logic).
# --------------------------------------------------------------------------

async def store_flintel_web_data(user_email: str, domain: str, url: str, report: dict) -> dict:
    """
    Append Claude's analysis (report + discovery_keywords) for this domain,
    tied to user_email, into flintel_web_data, with an incrementing
    sequence number PER (user_email, domain) — so history for a domain,
    for THIS user, can be read back in order. Two different users
    analyzing the same domain each get their own independent sequence.
    """
    sequence = await flintel_web_data_collection.count_documents(
        {"domain": domain, "user_email": user_email}
    ) + 1

    doc = {
        "user_email":         user_email,
        "domain":             domain,
        "url":                url,
        "sequence":           sequence,
        "discovery_keywords": report.get("discovery_keywords") or [],
        "company_summary":    report.get("company_summary", ""),
        "buyer_pain_points":  report.get("buyer_pain_points", []),
        "created_at":         datetime.now(timezone.utc),
    }
    try:
        result = await flintel_web_data_collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
    except Exception as exc:
        logger.error("MongoDB insert_one (flintel_web_data) failed for domain=%s user=%s: %s",
                     domain, user_email, exc)
        doc["_id"] = None

    # Register/refresh this (user, domain) pair in the tracking-state
    # collection so the live background tracker immediately picks up the
    # newest keyword set for this user on its very next pass. We deliberately
    # do NOT reset last_checked_at here if it already exists — re-analyzing
    # a domain should never cause the tracker to re-scan already-seen posts.
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
    """Same $or regex match used everywhere: search_keyword / subreddit_or_channel / text."""
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
    """
    Read-only query against the EXISTING web_data collection (already
    populated outside this app, by the separate background worker service).
    For each keyword Claude produced, find documents where:
      - search_keyword matches the keyword (case-insensitive), OR
      - subreddit_or_channel matches the keyword, OR
      - the post text contains the keyword
    This app never writes to web_data — only reads/matches. Used for the
    on-demand "analyze right now" flow; the live tracker below uses its
    own narrower, timestamp-bounded version of this same query.
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
# NEW — LIVE BACKGROUND TRACKER
#
# Runs forever, independent of any user being logged in or any page being
# open. Every LIVE_TRACKER_INTERVAL_SECONDS it:
#
#   1. Reads flintel_tracking_state — one document per (user_email, domain)
#      pair that has ever been analyzed, holding that pair's current
#      discovery_keywords and the timestamp of the newest web_data row
#      already checked for it (last_checked_at).
#
#   2. For each pair, queries web_data for rows that are BOTH:
#        - newer than last_checked_at (genuinely new since the last pass)
#        - a keyword match (same $or regex logic used everywhere else)
#
#   3. Any match found is upserted into flintel_user_signals, keyed by
#      (user_email, source web_data _id) so the SAME underlying post is
#      never duplicated for the same user even if it matches multiple
#      keywords or the tracker somehow sees it twice.
#
#   4. Advances last_checked_at for that pair to "now" once done, so the
#      NEXT pass only ever looks at what's newly arrived since this pass —
#      this fully replaces having to re-scan the whole web_data collection
#      every time, exactly the same fetch-once-forward principle used by
#      the SERP discovery cache in the separate worker service.
#
# Net effect: a user can close their laptop entirely, and by the time they
# log back in — an hour, a day, or 10 days later — flintel_user_signals
# already has every match that arrived in the meantime, ready to display
# with zero live querying needed.
# --------------------------------------------------------------------------

def _web_data_timestamp_field(doc: dict) -> datetime | None:
    """web_data rows use fetched_at if present, else created_at, else None."""
    ts = doc.get("fetched_at") or doc.get("created_at")
    return ts if isinstance(ts, datetime) else None


async def _persist_match_for_user(user_email: str, domain: str, keyword: str, source_doc: dict):
    """
    Upserts ONE matched web_data row into the user's permanent
    flintel_user_signals collection. Uniqueness is on (user_email,
    source_id) — see index creation in on_startup() — so re-matching the
    same post for the same user (e.g. it matches two of their keywords)
    never creates a duplicate row; it just updates matched_keyword info.
    """
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
            {"$set": doc},
            upsert=True,
        )
    except Exception as exc:
        logger.error("Failed to persist match for user=%s source_id=%s: %s", user_email, source_id, exc)


async def run_live_match_tracker():
    """
    The main loop. Started once at app startup, runs forever in the
    background alongside FastAPI's own request handling — it does not
    block or slow down any HTTP route.
    """
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

                # Only rows strictly newer than last_checked_at — this is
                # what makes each pass cheap regardless of how big web_data
                # has grown, and guarantees no post is ever matched twice.
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
                try:
                    cursor = web_data_collection.find(full_query)
                    async for source_doc in cursor:
                        # Figure out which specific keyword(s) actually matched,
                        # so matched_keyword is meaningful (not just "some keyword").
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

            if new_matches_total:
                logger.info("[LIVE-TRACKER] pass complete | pairs_checked:%d | new_matches:%d",
                            pair_count, new_matches_total)

        except Exception as exc:
            logger.error("[LIVE-TRACKER] loop error: %s", exc)

        await asyncio.sleep(LIVE_TRACKER_INTERVAL_SECONDS)


@app.on_event("startup")
async def on_startup():
    # Uniqueness index: one row per (user_email, source_id) in
    # flintel_user_signals — this is the hard guarantee against duplicates.
    await flintel_user_signals_collection.create_index(
        [("user_email", 1), ("source_id", 1)], unique=True, name="user_source_unique"
    )
    await flintel_tracking_state_collection.create_index(
        [("user_email", 1), ("domain", 1)], unique=True, name="user_domain_unique"
    )
    await users_collection.create_index([("email", 1)], unique=True, name="email_unique")
    await flintel_web_data_collection.create_index([("user_email", 1), ("domain", 1)])

    # Launch the live tracker as a detached background task — it runs for
    # the lifetime of the process, independent of any individual request.
    asyncio.create_task(run_live_match_tracker())
    logger.info("Flintel startup complete — live tracker scheduled.")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "web.html", {})


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, user: dict = Depends(get_current_user)):
    """
    Full pipeline for a domain, for the LOGGED-IN user:
      1. Fetch + analyze the website -> Claude produces a report + discovery_keywords.
         (Claude's involvement ends here — no scoring, no live fetching, nothing else.)
      2. Append that output to fx_signals.flintel_web_data, tied to this
         user's email, with an incrementing "sequence" per (user, domain).
      3. Using those keywords, run a read-only match against fx_signals.web_data
         (the EXISTING, pre-populated collection this app never writes to) —
         matching on search_keyword, subreddit_or_channel, or post text.
         (The live tracker, separately, keeps finding NEW matches for this
         same keyword set going forward — this step just gives the user an
         immediate first look.)
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

    flintel_doc = await store_flintel_web_data(user["email"], domain, url, report)

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
async def get_reports(domain: str, user: dict = Depends(get_current_user)):
    """Only this user's own analyses of this domain — never another user's."""
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
    """
    Re-run the on-demand match for a domain (same as before) for THIS user's
    most recent discovery_keywords for that domain.
    """
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
async def get_my_signals(domain: str | None = None, limit: int = 100,
                          user: dict = Depends(get_current_user)):
    """
    NEW — the dashboard endpoint. Returns matches already collected by the
    LIVE TRACKER and stored in flintel_user_signals — no live query against
    web_data happens here at all. This is what makes "log back in 10 days
    later and everything is already there" work: the data was written
    continuously in the background while the user was away.
    """
    query: dict = {"user_email": user["email"]}
    if domain:
        query["domain"] = domain

    cursor = flintel_user_signals_collection.find(query, {"_id": 0}).sort("matched_at", -1).limit(limit)
    signals = [doc async for doc in cursor]
    return JSONResponse({"count": len(signals), "signals": signals})


@app.get("/api/my-domains")
async def get_my_domains(user: dict = Depends(get_current_user)):
    """All domains this user has ever analyzed, most recent first."""
    cursor = flintel_web_data_collection.find(
        {"user_email": user["email"]}
    ).sort("created_at", -1)
    seen_domains = []
    async for doc in cursor:
        if doc["domain"] not in seen_domains:
            seen_domains.append(doc["domain"])
    return JSONResponse({"domains": seen_domains})


@app.get("/api/health")
async def health():
    mongo_status = "unknown"
    try:
        await mongo_client.admin.command("ping")
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"error: {exc}"
    return {"status": "ok", "mongodb": mongo_status, "database": MONGODB_DB_NAME}
