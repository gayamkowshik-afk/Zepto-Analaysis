# Module 1 — Data Pipeline (`/data_pipeline`)

## What this does

Scrapes book catalog data from [books.toscrape.com](https://books.toscrape.com)
(a public scraping-practice site — no login, no API key, no paid tier),
cleans it, converts price from GBP to INR using a **fixed baseline rate**,
loads it into a normalized two-table SQLite database, and runs SQL + pandas
queries against it.

## Install & run

```bash
pip install requests beautifulsoup4 pandas
python scrape_clean_load.py
```

This produces, in this folder:

| File | Description |
|---|---|
| `books_raw.csv` | Raw scraped rows (title, raw price string, raw star-rating class, raw availability text, category) |
| `books_clean.csv` | Cleaned/typed rows: `price_gbp`, `price_inr`, `rating` (int), `in_stock` (bool) |
| `books.db` | SQLite database with the two-table normalized schema below |
| `queries_output.txt` | All 6 SQL queries with their printed output, plus the pandas cross-check |

## Design decisions

**Category/book coverage.** The scraper walks the site's category list and
paginates through each category until it has covered at least 3 categories
*and* at least 60 total books (it typically ends up with more, since a
category's full page count is scraped before moving on — categories aren't
split mid-page).

**Currency conversion.** `price_inr = price_gbp * 105.50`, a fixed,
project-defined constant stated here as required. This is **not** a live or
historical market rate — no network call or date reference is used for the
conversion itself, only for the initial scrape.

**Cleaning / handling malformed rows.**
- `price_gbp` and `rating` are numeric fields. If either fails to parse for
  a given row (e.g. an unexpected price string), that field is
  **median-imputed** rather than dropping the row — a single malformed
  numeric field doesn't necessarily mean the whole row is untrustworthy, and
  median imputation is robust to outliers.
- `in_stock` is derived from free-text availability. If the availability
  text doesn't contain a recognizable "in stock" / "out of stock" phrase,
  there's no sensible "median" for a boolean, so that **row is dropped**
  instead. This is logged with a count at cleaning time.

**Schema.**
```sql
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```

**Queries.** Six SQL queries are run (one more than the required five),
collectively covering `SELECT`/`WHERE`, `ORDER BY`/`LIMIT`, `DISTINCT`,
`IN`/`BETWEEN`, and two `JOIN` queries (a top-N join and a `GROUP BY`
aggregate join).

**pandas cross-check.** The category-level aggregate JOIN query (Q6 — book
count and average `price_inr` per category) is independently reproduced
using only `pd.merge` on in-memory DataFrames (no SQL), and the script
asserts the two results match exactly.

## A note on running this

This script was developed and its cleaning/DB/query/cross-check logic was
verified against a synthetic dataset shaped like the real scrape output
(same columns, same messy-row cases). The live scrape against
books.toscrape.com itself needs to be run on a machine with normal internet
access — it wasn't executed against the live site during development here.
