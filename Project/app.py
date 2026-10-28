import os

import requests
import streamlit as st

# =============================
# CONFIG
# =============================
# Local default; set API_BASE to your deployed backend URL (e.g. your Render URL)
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# =============================
# STYLES (minimal modern)
# =============================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
.small-muted { color:#6b7280; font-size: 0.92rem; }
.movie-title { font-size: 0.9rem; line-height: 1.15rem; height: 2.3rem; overflow: hidden; }
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE + ROUTING (single-file pages)
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details | library
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "last_logged_id" not in st.session_state:
    st.session_state.last_logged_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view in ("home", "details", "library"):
    st.session_state.view = qp_view
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except Exception:
        pass


def logged_in() -> bool:
    return bool(st.session_state.token)


def goto_home():
    st.session_state.view = "home"
    st.session_state.last_logged_id = None
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_library():
    st.session_state.view = "library"
    st.query_params["view"] = "library"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# API HELPERS
# =============================
@st.cache_data(ttl=30)  # short cache for public (non-user) endpoints only
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"


def api_request(method: str, path: str, params=None, json=None, timeout: int = 25):
    """Uncached request that sends the login token. Use for all /auth and /user calls."""
    headers = {}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.request(
            method,
            f"{API_BASE}{path}",
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )
    except Exception as e:
        return None, f"Request failed: {e}"

    if r.status_code == 401 and token:
        st.session_state.token = None
        st.session_state.username = None
        return None, "Session expired. Please log in again."
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        if isinstance(detail, list):  # pydantic validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        return None, str(detail)
    if not r.content:
        return {}, None
    return r.json(), None


def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")

            with colset[c]:
                if poster:
                    st.image(poster, width="stretch")
                else:
                    st.markdown("🖼️ *No poster*")

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True
                )
                if m.get("caption"):
                    st.caption(m["caption"])


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title") or x.get("title") or "Untitled",
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return cards


# =============================
# IMPORTANT: Robust TMDB search parsing
# Supports BOTH API shapes:
# 1) raw TMDB: {"results":[{id,title,poster_path,...}]}
# 2) list cards: [{tmdb_id,title,poster_url,...}]
# =============================
def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    """
    Returns:
      suggestions: list[(label, tmdb_id)]
      cards: list[{tmdb_id,title,poster_url}]
    """
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            title = (m.get("title") or "").strip()
            tmdb_id = m.get("id")
            poster_path = m.get("poster_path")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": f"{TMDB_IMG}{poster_path}" if poster_path else None,
                    "release_date": m.get("release_date", ""),
                }
            )

    elif isinstance(data, list):
        raw_items = []
        for m in data:
            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = (m.get("title") or "").strip()
            poster_url = m.get("poster_url")
            if not title or not tmdb_id:
                continue
            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": m.get("release_date", ""),
                }
            )
    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:10]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [
        {"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
        for x in final_list[:limit]
    ]
    return suggestions, cards


# =============================
# SIDEBAR: menu + account
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")
    if st.button("🏠 Home"):
        goto_home()
    if logged_in() and st.button("📚 My Library"):
        goto_library()

    st.markdown("---")

    # ----- Account box -----
    if logged_in():
        st.markdown(f"👤 **{st.session_state.username}**")
        if st.button("Log out"):
            api_request("POST", "/auth/logout")
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.last_logged_id = None
            goto_home()
    else:
        st.markdown("### 🔐 Account")
        tab_login, tab_register = st.tabs(["Log in", "Register"])

        with tab_login:
            with st.form("login_form"):
                lu = st.text_input("Username", key="login_user")
                lp = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Log in"):
                    res, err = api_request(
                        "POST", "/auth/login", json={"username": lu, "password": lp}
                    )
                    if err:
                        st.error(err)
                    else:
                        st.session_state.token = res["token"]
                        st.session_state.username = res["username"]
                        st.rerun()

        with tab_register:
            with st.form("register_form"):
                ru = st.text_input("Username (3-32 chars)", key="reg_user")
                rp = st.text_input(
                    "Password (min 6 chars)", type="password", key="reg_pass"
                )
                if st.form_submit_button("Create account"):
                    res, err = api_request(
                        "POST", "/auth/register", json={"username": ru, "password": rp}
                    )
                    if err:
                        st.error(err)
                    else:
                        st.session_state.token = res["token"]
                        st.session_state.username = res["username"]
                        st.rerun()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed (only home)")
    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )
    grid_cols = st.slider("Grid columns", 4, 8, 6)

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Type keyword → dropdown suggestions + matching results → open → details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# VIEW: HOME
# ==========================================================
if st.session_state.view == "home":
    typed = st.text_input(
        "Search by movie title (keyword)", placeholder="Type: avenger, batman, love..."
    )

    st.divider()

    # SEARCH MODE (Autocomplete + word-match results)
    if typed.strip():
        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters for suggestions.")
        else:
            data, err = api_get_json("/tmdb/search", params={"query": typed.strip()})

            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_tmdb_search_to_cards(
                    data, typed.strip(), limit=24
                )

                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    selected = st.selectbox("Suggestions", labels, index=0)

                    if selected != "-- Select a movie --":
                        label_to_id = {s[0]: s[1] for s in suggestions}
                        goto_details(label_to_id[selected])
                else:
                    st.info("No suggestions found. Try another keyword.")

                st.markdown("### Results")
                poster_grid(cards, cols=grid_cols, key_prefix="search_results")

        st.stop()

    # PERSONALIZED ROW (only when logged in and has something to base it on)
    if logged_in():
        for_you, err_fy = api_request(
            "GET", "/user/recommendations", params={"limit": 12}
        )
        if for_you:
            st.markdown("### ✨ Recommended for you")
            poster_grid(for_you, cols=grid_cols, key_prefix="home_for_you")
            st.divider()

    # HOME FEED MODE
    st.markdown(f"### 🏠 Home — {home_category.replace('_',' ').title()}")

    home_cards, err = api_get_json(
        "/home", params={"category": home_category, "limit": 24}
    )
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()

    poster_grid(home_cards, cols=grid_cols, key_prefix="home_feed")

# ==========================================================
# VIEW: LIBRARY (watchlist / favorites / history / ratings / for you)
# ==========================================================
elif st.session_state.view == "library":
    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📚 My Library")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    if not logged_in():
        st.info("Log in from the sidebar to see your library.")
        st.stop()

    t_fy, t_wl, t_fav, t_hist, t_rat = st.tabs(
        ["✨ For You", "📌 Watchlist", "❤️ Favorites", "🕘 History", "⭐ My Ratings"]
    )

    with t_fy:
        recs, err = api_request("GET", "/user/recommendations", params={"limit": 18})
        if err:
            st.error(err)
        elif not recs:
            st.info(
                "Favorite or rate (4-5 ⭐) a few movies and your personalized "
                "recommendations will show up here."
            )
        else:
            poster_grid(recs, cols=grid_cols, key_prefix="lib_for_you")

    for tab, name, label in (
        (t_wl, "watchlist", "watchlist"),
        (t_fav, "favorites", "favorites"),
        (t_hist, "history", "history"),
    ):
        with tab:
            items, err = api_request("GET", f"/user/lists/{name}")
            if err:
                st.error(err)
            elif not items:
                st.info(f"Your {label} is empty.")
            else:
                poster_grid(items, cols=grid_cols, key_prefix=f"lib_{name}")
                if name == "history" and st.button("Clear history"):
                    api_request("DELETE", "/user/lists/history")
                    st.rerun()

    with t_rat:
        ratings, err = api_request("GET", "/user/ratings")
        if err:
            st.error(err)
        elif not ratings:
            st.info("You haven't rated any movies yet.")
        else:
            cards = [
                {**r, "caption": "⭐" * int(r["rating"]) + f"  ({r['rating']}/5)"}
                for r in ratings
            ]
            poster_grid(cards, cols=grid_cols, key_prefix="lib_ratings")

# ==========================================================
# VIEW: DETAILS
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    # Top bar
    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 📄 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")
    if err or not data:
        st.error(f"Could not load details: {err or 'Unknown error'}")
        st.stop()

    movie_ref = {
        "tmdb_id": int(tmdb_id),
        "title": data.get("title") or "Untitled",
        "poster_url": data.get("poster_url"),
    }

    # Record in viewing history (once per visit)
    if logged_in() and st.session_state.last_logged_id != tmdb_id:
        api_request("POST", "/user/lists/history", json=movie_ref)
        st.session_state.last_logged_id = tmdb_id

    # Layout: Poster LEFT, Details RIGHT
    left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], width="stretch")
        else:
            st.write("🖼️ No poster")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        release = data.get("release_date") or "-"
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Release: {release}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>", unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- User actions: favorite / watchlist / rating -----
    st.markdown("#### Your actions")
    if not logged_in():
        st.caption("🔐 Log in from the sidebar to save favorites, a watchlist and ratings.")
    else:
        status, serr = api_request("GET", f"/user/status/{tmdb_id}")
        if serr or status is None:
            st.warning(f"Could not load your status: {serr}")
        else:
            c1, c2, c3 = st.columns([1, 1, 2])

            with c1:
                if status["is_favorite"]:
                    if st.button("💔 Remove favorite", key="fav_rm"):
                        api_request("DELETE", f"/user/lists/favorites/{tmdb_id}")
                        st.rerun()
                else:
                    if st.button("❤️ Add to favorites", key="fav_add"):
                        api_request("POST", "/user/lists/favorites", json=movie_ref)
                        st.rerun()

            with c2:
                if status["in_watchlist"]:
                    if st.button("✖️ Remove from watchlist", key="wl_rm"):
                        api_request("DELETE", f"/user/lists/watchlist/{tmdb_id}")
                        st.rerun()
                else:
                    if st.button("📌 Add to watchlist", key="wl_add"):
                        api_request("POST", "/user/lists/watchlist", json=movie_ref)
                        st.rerun()

            with c3:
                current = status.get("rating")
                options = [1, 2, 3, 4, 5]
                new_rating = st.radio(
                    "Your rating",
                    options,
                    index=options.index(current) if current else None,
                    format_func=lambda n: "⭐" * n,
                    horizontal=True,
                    key=f"rating_radio_{tmdb_id}",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Save rating", key="rating_save"):
                        if new_rating is None:
                            st.warning("Pick a rating first.")
                        else:
                            _, rerr = api_request(
                                "PUT",
                                "/user/ratings",
                                json={**movie_ref, "rating": int(new_rating)},
                            )
                            if rerr:
                                st.error(rerr)
                            else:
                                st.rerun()
                with rc2:
                    if current and st.button("Remove rating", key="rating_rm"):
                        api_request("DELETE", f"/user/ratings/{tmdb_id}")
                        st.rerun()
                if current:
                    st.caption(f"You rated this {current}/5.")

    if data.get("backdrop_url"):
        st.markdown("#### Backdrop")
        st.image(data["backdrop_url"], width="stretch")

    st.divider()
    st.markdown("### ✅ Recommendations")

    title = (data.get("title") or "").strip()
    if title:
        bundle, err2 = api_get_json(
            "/movie/search",
            params={"query": title, "tfidf_top_n": 12, "genre_limit": 12},
        )

        if not err2 and bundle:
            st.markdown("#### 🔎 Similar Movies (TF-IDF)")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown("#### 🎭 More Like This (Genre)")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="details_genre",
            )
        else:
            st.info("Showing Genre recommendations (fallback).")
            genre_only, err3 = api_get_json(
                "/recommend/genre", params={"tmdb_id": tmdb_id, "limit": 18}
            )
            if not err3 and genre_only:
                poster_grid(
                    genre_only, cols=grid_cols, key_prefix="details_genre_fallback"
                )
            else:
                st.warning("No recommendations available right now.")
    else:
        st.warning("No title available to compute recommendations.")
