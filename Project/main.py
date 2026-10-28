import asyncio
import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import auth_db


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# ----- user feature models -----
class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    token: str
    username: str


class MovieRef(BaseModel):
    tmdb_id: int
    title: str = Field(..., min_length=1)
    poster_url: Optional[str] = None


class RatingIn(MovieRef):
    rating: int = Field(..., ge=1, le=5)


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bulletproof TMDB GET:
    - Automatically retries up to 3 times if network errors/timeouts occur.
    - Extended timeout configuration for stability.
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    timeout = httpx.Timeout(30.0, connect=15.0)

    last_error = None
    for attempt in range(3):  # Try up to 3 times
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{TMDB_BASE}{path}", params=q)
                
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 429:  # Rate limit hit, wait longer
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                else:
                    raise HTTPException(
                        status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
                    )
        except httpx.RequestError as e:
            last_error = e
            await asyncio.sleep(1 * (attempt + 1))  # Backoff delay before retry
        except HTTPException:
            raise

    # If all retries fail, return an empty results dictionary instead of crashing 502
    return {"results": []}
    """
    Safe TMDB GET with retry mechanism and extended timeout.
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    # Increase timeout to 30 seconds
    timeout = httpx.Timeout(30.0, connect=15.0)

    for attempt in range(2):  # Try twice if connect error occurs
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{TMDB_BASE}{path}", params=q)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code != 200 and attempt == 1:
                    raise HTTPException(
                        status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
                    )
        except httpx.RequestError as e:
            if attempt == 1:
                raise HTTPException(
                    status_code=502,
                    detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
                )
            await asyncio.sleep(0.5)  # wait briefly before retrying

async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


def tfidf_scores_for_title(title: str) -> Optional[Tuple[int, np.ndarray]]:
    """
    Similarity of every local movie to `title`.
    Returns (local_idx, scores) or None when the title isn't in the local dataset.
    """
    if df is None or tfidf_matrix is None or TITLE_TO_IDX is None:
        return None
    idx = TITLE_TO_IDX.get(_norm_title(title))
    if idx is None:
        return None
    qv = tfidf_matrix[idx]
    return idx, (tfidf_matrix @ qv.T).toarray().ravel()


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES + INIT USER DB
# =========================
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    auth_db.init_db()
    try:
        load_pickles()
        print("TF-IDF resources loaded.")
    except FileNotFoundError as e:
        # Accounts/lists still work; only TF-IDF endpoints will return errors.
        print(f"WARNING: could not load pickles ({e}). TF-IDF features disabled.")
    yield
    # SHUTDOWN (nothing to clean up)


# =========================
# FASTAPI APP (single instance)
# =========================
app = FastAPI(title="Movie Recommender API", version="4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# AUTH DEPENDENCY
# =========================
def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth_db.get_user_by_token(authorization[7:].strip())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------

@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit with safe fallbacks for any category.
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        params = {"language": "en-US", "page": 1}
        if category in {"now_playing", "upcoming"}:
            params["region"] = "US"

        data = await tmdb_get(f"/movie/{category}", params)
        cards = await tmdb_cards_from_results(data.get("results", []), limit=limit)
        
        # If TMDB is temporarily down, fallback to popular or return empty gracefully
        if not cards and category != "popular":
            fallback_data = await tmdb_get("/movie/popular", {"language": "en-US", "page": 1})
            cards = await tmdb_cards_from_results(fallback_data.get("results", []), limit=limit)

        return cards

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")

# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    # fetch posters concurrently instead of one-by-one
    cards = await asyncio.gather(*(attach_tmdb_card_by_title(t) for t, _ in recs))
    tfidf_items: List[TFIDFRecItem] = [
        TFIDFRecItem(title=t, score=s, tmdb=card)
        for (t, s), card in zip(recs, cards)
    ]

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        genre_cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in genre_cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


# ==========================================================
# USER FEATURES
# ==========================================================

# ---------- AUTH ----------
@app.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(body: Credentials):
    username = body.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=422, detail="Username too short")
    user_id = auth_db.create_user(username, body.password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="Username already taken")
    return AuthResponse(token=auth_db.create_session(user_id), username=username)


@app.post("/auth/login", response_model=AuthResponse)
def login(body: Credentials):
    user = auth_db.verify_user(body.username.strip(), body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Wrong username or password")
    return AuthResponse(
        token=auth_db.create_session(user["id"]), username=user["username"]
    )


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.lower().startswith("bearer "):
        auth_db.delete_session(authorization[7:].strip())
    return {"status": "logged_out"}


@app.get("/auth/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"id": user["id"], "username": user["username"]}


# ---------- WATCHLIST / FAVORITES / HISTORY ----------
def _check_list_name(list_name: str) -> str:
    if list_name not in auth_db.LIST_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown list '{list_name}'. Use one of: {sorted(auth_db.LIST_NAMES)}",
        )
    return list_name


@app.get("/user/lists/{list_name}")
def get_list(
    list_name: str,
    limit: int = Query(100, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Returns movie cards: [{tmdb_id, title, poster_url, added_at}]"""
    _check_list_name(list_name)
    return auth_db.list_get(user["id"], list_name, limit)


@app.post("/user/lists/{list_name}", status_code=201)
def add_to_list(
    list_name: str,
    movie: MovieRef,
    user: Dict[str, Any] = Depends(get_current_user),
):
    _check_list_name(list_name)
    auth_db.list_add(user["id"], list_name, movie.tmdb_id, movie.title, movie.poster_url)
    return {"status": "added", "list": list_name, "tmdb_id": movie.tmdb_id}


@app.delete("/user/lists/{list_name}/{tmdb_id}")
def remove_from_list(
    list_name: str,
    tmdb_id: int,
    user: Dict[str, Any] = Depends(get_current_user),
):
    _check_list_name(list_name)
    auth_db.list_remove(user["id"], list_name, tmdb_id)
    return {"status": "removed", "list": list_name, "tmdb_id": tmdb_id}


@app.delete("/user/lists/{list_name}")
def clear_list(list_name: str, user: Dict[str, Any] = Depends(get_current_user)):
    _check_list_name(list_name)
    auth_db.list_clear(user["id"], list_name)
    return {"status": "cleared", "list": list_name}


# ---------- RATINGS ----------
@app.get("/user/ratings")
def get_my_ratings(user: Dict[str, Any] = Depends(get_current_user)):
    return auth_db.get_ratings(user["id"])


@app.put("/user/ratings")
def rate_movie(body: RatingIn, user: Dict[str, Any] = Depends(get_current_user)):
    auth_db.set_rating(
        user["id"], body.tmdb_id, body.rating, body.title, body.poster_url
    )
    return {"status": "rated", "tmdb_id": body.tmdb_id, "rating": body.rating}


@app.delete("/user/ratings/{tmdb_id}")
def remove_rating(tmdb_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    auth_db.delete_rating(user["id"], tmdb_id)
    return {"status": "removed", "tmdb_id": tmdb_id}


# ---------- STATUS OF ONE MOVIE (for the details page buttons) ----------
@app.get("/user/status/{tmdb_id}")
def movie_status(tmdb_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    return auth_db.get_status(user["id"], tmdb_id)


# ---------- PERSONALIZED RECOMMENDATIONS ----------
@app.get("/user/recommendations", response_model=List[TMDBMovieCard])
async def personalized_recommendations(
    limit: int = Query(12, ge=1, le=30),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    'Recommended for you': blends TF-IDF similarity to the movies the user
    favorited (weight 1.0), rated 4-5 stars (rating/5) or put on the watchlist (0.4).
    Movies the user already favorited, rated, saved or viewed are excluded.
    """
    uid = user["id"]
    favorites = auth_db.list_get(uid, "favorites", 100)
    watchlist = auth_db.list_get(uid, "watchlist", 100)
    history = auth_db.list_get(uid, "history", auth_db.HISTORY_LIMIT)
    ratings = auth_db.get_ratings(uid)

    # seed title -> weight (keep the highest weight per title)
    seeds: Dict[str, float] = {}

    def add_seed(title: str, weight: float):
        seeds[title] = max(seeds.get(title, 0.0), weight)

    for m in favorites:
        add_seed(m["title"], 1.0)
    for r in ratings:
        if r["rating"] >= 4:
            add_seed(r["title"], r["rating"] / 5.0)
    for m in watchlist:
        add_seed(m["title"], 0.4)

    if not seeds or df is None:
        return []

    # everything the user has already interacted with
    seen_ids = {m["tmdb_id"] for m in favorites + watchlist + history}
    seen_ids |= {r["tmdb_id"] for r in ratings}
    seen_titles = {
        _norm_title(x["title"]) for x in favorites + watchlist + history + ratings
    }

    total = np.zeros(len(df), dtype=float)
    used = 0
    for title, weight in seeds.items():
        res = tfidf_scores_for_title(title)
        if res is None:
            continue
        idx, scores = res
        total += weight * scores
        total[idx] = -1.0  # never recommend a seed itself
        used += 1

    if used == 0:
        return []

    candidates: List[str] = []
    for i in np.argsort(-total):
        if total[int(i)] <= 0:
            break
        try:
            t = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        if _norm_title(t) in seen_titles:
            continue
        candidates.append(t)
        if len(candidates) >= limit + 8:  # extra, since some won't match on TMDB
            break

    cards = await asyncio.gather(*(attach_tmdb_card_by_title(t) for t in candidates))

    out: List[TMDBMovieCard] = []
    added_ids = set()
    for c in cards:
        if c is None or c.tmdb_id in seen_ids or c.tmdb_id in added_ids:
            continue
        added_ids.add(c.tmdb_id)
        out.append(c)
        if len(out) >= limit:
            break
    return out
