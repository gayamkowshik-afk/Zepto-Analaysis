"""
Module 1 -- Data Pipeline (/data_pipeline)

Scrapes book catalog data from books.toscrape.com (a public scraping-practice
site, no login/API key/paid tier required), cleans it, converts price from
GBP to INR using a fixed project baseline rate, loads it into a normalized
two-table SQLite schema, and runs SQL + pandas queries against it.

Run:
    python scrape_clean_load.py

Outputs (written next to this script):
    books_raw.csv       -- raw scraped rows before cleaning
    books_clean.csv      -- cleaned rows with price_gbp / price_inr / rating / in_stock
    books.db             -- SQLite database (categories, books tables)
    queries_output.txt   -- the 5+ SQL queries and their printed output
"""

import os
import re
import sqlite3
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/"
HERE = os.path.dirname(os.path.abspath(__file__))

GBP_TO_INR = 105.50  # fixed, project-defined baseline conversion rate (required, keyless)

RATING_WORD_TO_INT = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (data-pipeline-assignment)"})


def get_soup(url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def discover_categories() -> list[dict]:
    """Return list of {name, url} for every book category on the site."""
    soup = get_soup(BASE_URL)
    cats = []
    for a in soup.select("div.side_categories ul li ul li a"):
        name = a.get_text(strip=True)
        url = BASE_URL + a["href"]
        cats.append({"name": name, "url": url})
    return cats


def scrape_category(name: str, url: str) -> list[dict]:
    """Scrape every book across every paginated listing page of one category."""
    rows = []
    page_url = url
    while page_url:
        soup = get_soup(page_url)
        for art in soup.select("article.product_pod"):
            title = art.h3.a["title"]
            price_text = art.select_one("p.price_color").get_text(strip=True)
            availability_text = art.select_one("p.instock.availability").get_text(strip=True)
            star_classes = art.select_one("p.star-rating")["class"]
            # star_classes looks like ["star-rating", "Three"]
            star_rating_word = [c for c in star_classes if c != "star-rating"][0]
            rows.append({
                "title": title,
                "price": price_text,
                "star_rating": star_rating_word,
                "availability": availability_text,
                "category": name,
            })
        next_link = soup.select_one("li.next a")
        if next_link:
            page_url = page_url.rsplit("/", 1)[0] + "/" + next_link["href"]
        else:
            page_url = None
        time.sleep(0.2)  # be polite to the practice server
    return rows


def scrape_all(min_categories: int = 3, min_books: int = 60) -> pd.DataFrame:
    categories = discover_categories()
    all_rows = []
    used = 0
    for cat in categories:
        cat_rows = scrape_category(cat["name"], cat["url"])
        if not cat_rows:
            continue
        all_rows.extend(cat_rows)
        used += 1
        # keep going until we have both >=3 categories AND >=60 books,
        # then stop once both minimums are comfortably satisfied
        if used >= min_categories and len(all_rows) >= min_books:
            break
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_price(price_text: str) -> float | None:
    """'£51.77' -> 51.77. Returns None if it can't be parsed."""
    match = re.search(r"[\d.]+", price_text.replace(",", ""))
    return float(match.group()) if match else None


def clean_rating(word: str) -> int | None:
    return RATING_WORD_TO_INT.get(word)


def clean_availability(text: str) -> bool | None:
    """'In stock (22 available)' -> True, 'Out of stock' -> False."""
    low = text.lower()
    if "out of stock" in low:
        return False
    if "in stock" in low:
        return True
    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(clean_rating)
    df["in_stock"] = df["availability"].apply(clean_availability)

    # Rows where a field failed to parse: price_gbp and rating are numeric ->
    # median-impute; in_stock is boolean/categorical with no sensible median,
    # so a row with an unparseable availability string is dropped instead
    # (documented choice, not a silent crash).
    n_before = len(df)

    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    if df["rating"].isna().any():
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating).round().astype(int)

    dropped = df["in_stock"].isna().sum()
    df = df[df["in_stock"].notna()].copy()
    df["in_stock"] = df["in_stock"].astype(bool)

    print(f"Cleaning: {n_before} rows in, {dropped} dropped (unparseable availability), "
          f"{len(df)} rows out.")

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    df["rating"] = df["rating"].astype(int)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


# ---------------------------------------------------------------------------
# Database load
# ---------------------------------------------------------------------------

def load_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        );
    """)

    categories = sorted(df["category"].unique())
    cat_id_map = {}
    for name in categories:
        cur.execute("INSERT INTO categories (category_name) VALUES (?)", (name,))
        cat_id_map[name] = cur.lastrowid

    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["title"], row["price_gbp"], row["price_inr"], int(row["rating"]),
             int(row["in_stock"]), cat_id_map[row["category"]]),
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

QUERIES = [
    (
        "Q1 - SELECT/WHERE: in-stock books priced above 20 GBP",
        """SELECT title, price_gbp, rating
           FROM books
           WHERE in_stock = 1 AND price_gbp > 20
           LIMIT 10;""",
    ),
    (
        "Q2 - ORDER BY/LIMIT: 10 most expensive books (INR)",
        """SELECT title, price_inr
           FROM books
           ORDER BY price_inr DESC
           LIMIT 10;""",
    ),
    (
        "Q3 - DISTINCT: distinct category names present",
        """SELECT DISTINCT category_name FROM categories;""",
    ),
    (
        "Q4 - IN/BETWEEN: books rated 4 or 5 stars priced between 10 and 40 GBP",
        """SELECT title, rating, price_gbp
           FROM books
           WHERE rating IN (4, 5) AND price_gbp BETWEEN 10 AND 40
           ORDER BY price_gbp;""",
    ),
    (
        "Q5 - JOIN + ORDER BY/LIMIT: top 10 highest-rated books per category listing",
        """SELECT b.title, c.category_name, b.rating, b.price_inr
           FROM books b
           JOIN categories c ON b.category_id = c.category_id
           ORDER BY b.rating DESC, b.price_inr DESC
           LIMIT 10;""",
    ),
    (
        "Q6 - JOIN + aggregate: average price and book count per category",
        """SELECT c.category_name, COUNT(*) AS n_books, ROUND(AVG(b.price_inr), 2) AS avg_price_inr
           FROM books b
           JOIN categories c ON b.category_id = c.category_id
           GROUP BY c.category_name
           ORDER BY n_books DESC;""",
    ),
]


def run_queries(db_path: str, out_path: str) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    results = {}
    with open(out_path, "w") as f:
        for title, sql in QUERIES:
            df = pd.read_sql(sql, conn)
            results[title] = df
            f.write(f"{'=' * 70}\n{title}\n{'=' * 70}\n{sql.strip()}\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
    conn.close()
    return results


def cross_check_with_pandas(clean_df: pd.DataFrame, sql_results: dict) -> str:
    """Reproduce the Q5/Q6-style join purely with pandas (no SQL) and show it
    matches. We rebuild categories/books frames the same way load_to_sqlite did,
    then use pd.merge, and compare against the Q6 SQL aggregate result."""
    categories_df = pd.DataFrame({
        "category_name": sorted(clean_df["category"].unique()),
    })
    categories_df["category_id"] = range(1, len(categories_df) + 1)

    books_df = clean_df.merge(
        categories_df, left_on="category", right_on="category_name", how="left"
    )

    merged = pd.merge(
        books_df, categories_df, on=["category_id", "category_name"], how="inner"
    )

    pandas_agg = (
        merged.groupby("category_name")
        .agg(n_books=("title", "count"), avg_price_inr=("price_inr", "mean"))
        .reset_index()
        .sort_values("n_books", ascending=False)
    )
    pandas_agg["avg_price_inr"] = pandas_agg["avg_price_inr"].round(2)

    sql_agg = sql_results["Q6 - JOIN + aggregate: average price and book count per category"]
    sql_agg_sorted = sql_agg.sort_values("category_name").reset_index(drop=True)
    pandas_agg_sorted = pandas_agg.sort_values("category_name").reset_index(drop=True)[
        ["category_name", "n_books", "avg_price_inr"]
    ]

    match = sql_agg_sorted.equals(pandas_agg_sorted)
    summary = (
        f"pd.merge reproduction of the JOIN query "
        f"(category-level count + avg price_inr):\n\n{pandas_agg_sorted.to_string(index=False)}\n\n"
        f"Matches SQL/pd.read_sql output exactly: {match}\n"
    )
    return summary


def main():
    print("Discovering categories and scraping books.toscrape.com ...")
    raw_df = scrape_all(min_categories=3, min_books=60)
    raw_path = os.path.join(HERE, "books_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Scraped {len(raw_df)} raw rows across {raw_df['category'].nunique()} categories "
          f"-> {raw_path}")

    print("Cleaning and converting currency (1 GBP = {} INR) ...".format(GBP_TO_INR))
    clean_df = clean_dataframe(raw_df)
    clean_path = os.path.join(HERE, "books_clean.csv")
    clean_df.to_csv(clean_path, index=False)
    print(f"Cleaned data -> {clean_path} ({len(clean_df)} rows)")

    db_path = os.path.join(HERE, "books.db")
    print("Loading into normalized SQLite schema ...")
    load_to_sqlite(clean_df, db_path)
    print(f"Loaded -> {db_path}")

    out_path = os.path.join(HERE, "queries_output.txt")
    print("Running SQL queries ...")
    sql_results = run_queries(db_path, out_path)
    print(f"Query output -> {out_path}")

    print("Cross-checking JOIN result with pd.merge (no SQL) ...")
    cross_check_summary = cross_check_with_pandas(clean_df, sql_results)
    with open(out_path, "a") as f:
        f.write("=" * 70 + "\n")
        f.write("pandas cross-check (pd.read_sql vs pd.merge)\n")
        f.write("=" * 70 + "\n")
        f.write(cross_check_summary)
    print(cross_check_summary)

    print("\nDone. See books_raw.csv, books_clean.csv, books.db, queries_output.txt")


if __name__ == "__main__":
    main()
