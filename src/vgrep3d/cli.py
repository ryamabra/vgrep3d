"""vgrep CLI.

    vgrep index ~/Pictures      build or update the index
    vgrep "golden gate bridge"  search
    vgrep status                what is indexed
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from rich.console import Console

from . import index as faiss_index
from .config import SETTINGS
from .db import Db
from .scan import iter_images

app = typer.Typer(
    add_completion=False,
    help="Semantic grep for local files.",
    # Let `vgrep "some query"` fall through to search instead of being parsed
    # as an unknown subcommand.
    no_args_is_help=False,
)
console = Console()
err = Console(stderr=True)


def _open_db() -> Db:
    db = Db(SETTINGS.db_path)
    db.check_model_compatibility(SETTINGS.model, SETTINGS.dim)
    return db


@app.command("index")
def index_cmd(
    root: Path = typer.Argument(..., help="Directory to index"),
    batch: int = typer.Option(SETTINGS.batch_size, "--batch", "-b"),
) -> None:
    """Scan a directory and encode anything new or changed."""
    from .encoder import Encoder, load_image

    db = _open_db()

    with console.status("Scanning..."):
        found = queued = 0
        for p in iter_images(root):
            found += 1
            st = p.stat()
            if db.upsert_file(str(p), st.st_mtime, st.st_size):
                queued += 1
        db.conn.commit()
        removed = db.drop_missing()

    console.print(f"Found {found} images. {queued} need encoding. {removed} stale entries removed.")

    def rebuild() -> int:
        """Regenerate the search index from the embeddings in SQLite."""
        vecs, ids = db.all_embeddings(SETTINGS.dim)
        faiss_index.build(vecs, ids, SETTINGS.index_path, SETTINGS.dim)
        return len(ids)

    if queued == 0:
        # Still rebuild: the index file may be missing, stale, or from an older
        # format even when every file is already encoded.
        with console.status("Rebuilding index..."):
            n = rebuild()
        console.print(f"[dim]Index is up to date ({n} images).[/dim]")
        return

    enc = Encoder()
    console.print(f"[dim]Encoding on {enc.device}...[/dim]")

    todo = [(r["id"], r["path"]) for r in db.pending()]
    done = 0
    t0 = time.time()

    # Decode in threads (PIL releases the GIL), encode on the main thread.
    # Decode is usually the bottleneck, so this keeps the model fed.
    with ThreadPoolExecutor(max_workers=SETTINGS.loader_workers) as pool:
        for start in range(0, len(todo), batch):
            chunk = todo[start : start + batch]
            images = list(pool.map(lambda t: load_image(t[1]), chunk))

            pairs = [(fid, img) for (fid, _), img in zip(chunk, images) if img is not None]
            if not pairs:
                continue

            vecs = enc.encode_images([img for _, img in pairs])
            db.save_embeddings(zip((fid for fid, _ in pairs), vecs), time.time())

            done += len(pairs)
            rate = done / max(time.time() - t0, 1e-6)
            console.print(f"  {done}/{len(todo)}  ({rate:.1f} img/s)", end="\r")

    console.print()
    with console.status("Building index..."):
        rebuild()

    console.print(f"[green]Indexed {done} images in {time.time() - t0:.1f}s.[/green]")
    db.close()


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="What you are looking for, in plain language"),
    k: int = typer.Option(SETTINGS.top_k, "--k", "-k", help="Number of results"),
    threshold: float = typer.Option(
        0.0, "--min", help="Hide results below this similarity (0-1)"
    ),
    paths_only: bool = typer.Option(False, "--paths", help="Bare paths, for piping"),
    open_top: bool = typer.Option(
        False, "--open", "-o", help="Open the best match in Preview"
    ),
) -> None:
    """Search the index by meaning."""
    from .encoder import Encoder

    db = _open_db()
    idx = faiss_index.load(SETTINGS.index_path)
    if idx is None or idx.ntotal == 0:
        err.print("[yellow]Nothing indexed yet. Run `vgrep index <dir>` first.[/yellow]")
        raise typer.Exit(1)

    qvec = Encoder().encode_text([query])[0]
    hits = [(i, s) for i, s in faiss_index.search(idx, qvec, k) if s >= threshold]

    if not hits:
        err.print("[dim]No matches above threshold.[/dim]")
        raise typer.Exit(1)

    paths = db.paths_for(i for i, _ in hits)
    for fid, score in hits:
        p = paths.get(fid)
        if not p:
            continue
        if paths_only:
            print(p)
        else:
            # Similarity is shown because it is genuinely informative: a distinctive
            # subject scores far higher than a vague concept, and the user deserves
            # to see the difference rather than getting ten results that look equally
            # confident. Thresholds come from measurement, not guesswork: on a
            # 22-category corpus the noise floor sat around 8% and correct matches
            # landed at 12-18%, so those are the boundaries.
            colour = "green" if score > 0.14 else "yellow" if score > 0.10 else "dim"
            console.print(f"[{colour}]{score:6.1%}[/{colour}]  {p}")

    if open_top:
        import subprocess

        best = paths.get(hits[0][0])
        if best:
            subprocess.run(["open", best], check=False)

    db.close()


@app.command("status")
def status_cmd() -> None:
    """Show what is indexed."""
    db = _open_db()
    s = db.stats()
    console.print(f"Model:   {db.get_meta('model')}")
    console.print(f"Files:   {s['total']}")
    console.print(f"Encoded: {s['encoded']}")
    console.print(f"Pending: {s['total'] - s['encoded']}")
    console.print(f"[dim]{SETTINGS.db_path}[/dim]")
    db.close()


@app.command("reset")
def reset_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete the index and start over."""
    if not yes:
        typer.confirm("Delete all vgrep data?", abort=True)
    for p in (SETTINGS.db_path, SETTINGS.index_path):
        p.unlink(missing_ok=True)
    console.print("[green]Reset.[/green]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Allow `vgrep "some query"` as shorthand for `vgrep search "some query"`."""
    if ctx.invoked_subcommand is None and not ctx.args:
        console.print(ctx.get_help())


def run() -> None:
    """Entry point.

    Typer resolves the first argument as a subcommand name, so a bare query like
    `vgrep "a person"` would fail as an unknown command. Rewrite argv to insert
    the implicit `search` before handing off.
    """
    import sys

    known = {"index", "search", "status", "reset"}
    argv = sys.argv[1:]
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        sys.argv.insert(1, "search")
    app()


if __name__ == "__main__":
    run()
