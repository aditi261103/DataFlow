import io, json, re, sqlite3, time
from pathlib import Path

import pandas as pd
from groq import Groq
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB, MODEL = "dataflow.db", "openai/gpt-oss-120b"
llm = Groq()  # reads ANTHROPIC_API_KEY
app = FastAPI(title="DataFlow")
META = ("datasets", "charts", "chunks")


# ---------- database ----------
def db(readonly=False):
    if not readonly:
        c = sqlite3.connect(DB)
    else:  # AI-written SQL runs here: read-only, no access to metadata tables, 5s limit
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        ok = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE)
        def auth(action, a, *_):
            if action == sqlite3.SQLITE_READ:
                return sqlite3.SQLITE_DENY if a in META else sqlite3.SQLITE_OK
            return sqlite3.SQLITE_OK if action in ok else sqlite3.SQLITE_DENY
        c.set_authorizer(auth)
        t0 = time.time()
        c.set_progress_handler(lambda: time.time() - t0 > 5, 10000)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY, name TEXT, kind TEXT, tbl TEXT, rows INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS charts(id INTEGER PRIMARY KEY, dataset_id INTEGER, title TEXT, config TEXT);
    CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, dataset_id INTEGER, source TEXT, page INTEGER, text TEXT);
    """)


# ---------- parsing ----------
def clean(df):
    seen, cols = {}, []
    for i, col in enumerate(df.columns):
        n = re.sub(r"\W+", "_", str(col).strip().lower()).strip("_") or f"col{i}"
        seen[n] = seen.get(n, 0) + 1
        cols.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    df.columns = cols
    return df.infer_objects()


def parse(name, data):
    ext, tables, texts = Path(name).suffix.lower(), [], []
    if ext == ".csv":
        tables = [pd.read_csv(io.BytesIO(data))]
    elif ext == ".json":
        obj = json.loads(data)
        tables = [pd.json_normalize(obj if isinstance(obj, list) else [obj])]
    elif ext in (".xlsx", ".xls"):
        tables = list(pd.read_excel(io.BytesIO(data), sheet_name=None).values())
    elif ext == ".txt":
        texts = [(1, data.decode("utf-8", "ignore"))]
    elif ext == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        texts = [(1, "\n".join(p.text for p in d.paragraphs))]
        for t in d.tables:
            rows = [[c.text for c in r.cells] for r in t.rows]
            if len(rows) > 1:
                tables.append(pd.DataFrame(rows[1:], columns=rows[0]))
    elif ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, p in enumerate(pdf.pages, 1):
                texts.append((i, p.extract_text() or ""))
                for t in p.extract_tables():
                    if len(t) > 1:
                        tables.append(pd.DataFrame(t[1:], columns=t[0]))
    else:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    return tables, texts


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        tables, texts = parse(file.filename, await file.read())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read {file.filename}: {e}")
    with db() as c:
        for i, df in enumerate(tables, 1):
            df = clean(df)
            label = f"{file.filename} (table {i})" if len(tables) > 1 else file.filename
            cur = c.execute("INSERT INTO datasets(name,kind,rows) VALUES(?,?,?)", (label, "table", len(df)))
            tbl = f"t{cur.lastrowid}"
            c.execute("UPDATE datasets SET tbl=? WHERE id=?", (tbl, cur.lastrowid))
            df.to_sql(tbl, c, index=False)
        if any(t.strip() for _, t in texts):
            cur = c.execute("INSERT INTO datasets(name,kind,rows) VALUES(?,?,0)", (f"{file.filename} (text)", "document"))
            for pg, t in texts:
                for k in range(0, len(t), 1500):
                    part = t[k:k + 1500]
                    if part.strip():
                        c.execute("INSERT INTO chunks(dataset_id,source,page,text) VALUES(?,?,?,?)",
                                  (cur.lastrowid, file.filename, pg, part))
    return datasets()


# ---------- datasets: list / delete ----------
@app.get("/api/datasets")
def datasets():
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM datasets ORDER BY id DESC")]
        for r in rows:
            r["columns"] = [x["name"] for x in c.execute(f"PRAGMA table_info({r['tbl']})")] if r["tbl"] else []
    return rows


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: int):
    with db() as c:
        r = c.execute("SELECT tbl FROM datasets WHERE id=?", (ds_id,)).fetchone()
        if not r:
            raise HTTPException(404, "Dataset not found")
        if r["tbl"]:
            c.execute(f"DROP TABLE IF EXISTS {r['tbl']}")
        for t in ("chunks", "charts"):
            c.execute(f"DELETE FROM {t} WHERE dataset_id=?", (ds_id,))
        c.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
    return {"ok": True}


# ---------- LLM helpers ----------
def ask_llm(system, user, max_tokens=900):
    r = llm.chat.completions.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content


def as_json(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group())


def schema():
    out = []
    with db() as c:
        for d in c.execute("SELECT * FROM datasets"):
            if d["tbl"]:
                cols = [x["name"] for x in c.execute(f"PRAGMA table_info({d['tbl']})")]
                sample = c.execute(f"SELECT * FROM {d['tbl']} LIMIT 1").fetchone()
                out.append(f"TABLE {d['tbl']} (from {d['name']}, {d['rows']} rows): {', '.join(cols)}\n  sample: {dict(sample) if sample else {}}")
            else:
                out.append(f"DOCUMENT {d['name']} (searchable text)")
    return "\n".join(out) or "(no data uploaded)"


# ---------- charts: build / AI suggest / save / delete ----------
class ChartReq(BaseModel):
    dataset_id: int
    type: str = "bar"
    x: str
    y: str | None = None
    agg: str = "sum"


@app.post("/api/chart")
def chart(q: ChartReq):
    with db() as c:
        r = c.execute("SELECT tbl FROM datasets WHERE id=?", (q.dataset_id,)).fetchone()
        if not r or not r["tbl"]:
            raise HTTPException(404, "Table not found")
        cols = [x["name"] for x in c.execute(f"PRAGMA table_info({r['tbl']})")]
        if q.x not in cols or (q.y and q.y not in cols) or q.agg not in ("sum", "avg", "count", "min", "max"):
            raise HTTPException(400, "Invalid columns or aggregation")
        val = "COUNT(*)" if q.agg == "count" or not q.y else f'{q.agg.upper()}(CAST("{q.y}" AS REAL))'
        rows = c.execute(f'SELECT "{q.x}" AS label, {val} AS value FROM {r["tbl"]} GROUP BY 1 ORDER BY 1 LIMIT 100').fetchall()
    return {"labels": [str(x["label"]) for x in rows], "values": [x["value"] for x in rows]}


class SuggestReq(BaseModel):
    dataset_id: int
    prompt: str


@app.post("/api/chart/suggest")
def suggest(q: SuggestReq):
    ds = next((d for d in datasets() if d["id"] == q.dataset_id and d["tbl"]), None)
    if not ds:
        raise HTTPException(404, "Table not found")
    text = ask_llm(
        "Return ONLY JSON: {\"type\":\"bar|line|pie\",\"x\":col,\"y\":col or null,\"agg\":\"sum|avg|count|min|max\",\"title\":str}. Use only the listed columns.",
        f"Columns: {ds['columns']}\nRequest: {q.prompt}", 300)
    return as_json(text)


class SaveReq(BaseModel):
    dataset_id: int
    title: str
    config: dict


@app.post("/api/charts")
def save_chart(s: SaveReq):
    with db() as c:
        c.execute("INSERT INTO charts(dataset_id,title,config) VALUES(?,?,?)", (s.dataset_id, s.title, json.dumps(s.config)))
    return list_charts()


@app.get("/api/charts")
def list_charts():
    with db() as c:
        return [{**dict(r), "config": json.loads(r["config"])} for r in c.execute("SELECT * FROM charts ORDER BY id DESC")]


@app.delete("/api/charts/{cid}")
def delete_chart(cid: int):
    with db() as c:
        c.execute("DELETE FROM charts WHERE id=?", (cid,))
    return list_charts()


# ---------- ask: text-to-SQL + RAG ----------
class AskReq(BaseModel):
    question: str


SQL_OK = re.compile(r"^\s*(select|with)\b", re.I)
SQL_BAD = re.compile(r";|\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def run_sql(sql):
    sql = sql.strip().rstrip(";")
    if not SQL_OK.match(sql) or SQL_BAD.search(sql):
        raise ValueError("Only a single SELECT query is allowed")
    c = db(readonly=True)
    try:
        cur = c.execute(f"SELECT * FROM ({sql}) LIMIT 200")
        return [d[0] for d in cur.description], [list(r) for r in cur.fetchall()]
    finally:
        c.close()


def retrieve(q, k=4):
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT source,page,text FROM chunks")]
    if not rows:
        return []
    vec = TfidfVectorizer(stop_words="english").fit([r["text"] for r in rows] + [q])
    sims = cosine_similarity(vec.transform([q]), vec.transform([r["text"] for r in rows]))[0]
    return [rows[i] for i in sims.argsort()[::-1][:k] if sims[i] > 0]


@app.post("/api/ask")
def ask(req: AskReq):
    try:
        plan = as_json(ask_llm(
            "You route questions over uploaded data. Return ONLY JSON: {\"sql\": a SQLite SELECT using only the listed TABLEs, or null, \"use_docs\": true if the answer needs DOCUMENT text}.",
            f"{schema()}\n\nQuestion: {req.question}", 500))
        sql, result, error = plan.get("sql"), None, None
        if sql:
            try:
                cols, rows = run_sql(sql)
                result = {"columns": cols, "rows": rows}
            except Exception as e:
                error = str(e)
        chunks = retrieve(req.question) if plan.get("use_docs") else []
        context = f"SQL result: {json.dumps(result, default=str)[:6000]}\nSQL error: {error}\n\nDocument excerpts:\n" + \
            "\n".join(f"[{c['source']} p.{c['page']}] {c['text']}" for c in chunks)
        answer = ask_llm("Answer the question using only the context. Be concise. Cite documents as [file p.N]. If the context is insufficient, say so.",
                         f"Question: {req.question}\n\n{context}")
        return {"answer": answer, "sql": sql, "result": result, "error": error,
                "sources": sorted({f"{c['source']} p.{c['page']}" for c in chunks})}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
