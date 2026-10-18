# openoutnews/cli.py
"""The `outnews` console script.

    outnews find 10          build the next issue's 10 picks → CSV on stdout
    outnews label ID read    record how the reader reacted to a sent article
    outnews label ID skip

No wizard, no daemon: `find` builds the next issue — it fetches the days
since the last issue (see ``openoutnews.issue``), embeds what it hasn't seen,
puts the articles the qualifier chooses to the model against the newsletter's
brief (``openoutnews.qualify``), and prints the fits it ranks highest as CSV.
An issue is finished only once its verdicts are in, so a run that stops — a
crash, or ``--agent-qualify`` waiting on a verdict — is resumed by the next
`find` without fetching again. Run again on the day an issue was finished, it
prints that issue again. Labelling what the reader actually read is a
separate, explicit step, stored apart from the verdicts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

from openoutnews import anchors, brief, conf, fetch, issue, llm, qualify, store

OVERVIEW = """\
OpenOutNews — news picks for a newsletter, judged against its brief.

  outnews find 10                 the next issue's ten picks -> CSV on stdout
  outnews find 10 --brief PATH    ...judged by the rules in PATH, kept for later issues
  outnews find 10 --agent-qualify [--verdict fit|no-fit --reason MSG]
                                  the calling agent judges each article instead of the model
  outnews label ID read           record that the reader engaged with an article
  outnews label ID skip           record that they didn't

Configure OPENOUTNEWS_TOPICS (comma-separated), OPENOUTNEWS_BRIEF (who reads the
newsletter and what it covers), OPENOUTNEWS_AI_MODEL and OPENOUTNEWS_LLM_API_KEY
before the first `find`.
"""

CSV_FIELDS = ["issue", "id", "title", "url", "published_at", "reason", "mode", "score"]


def _write_issue(issue_row, picks) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in picks:
        writer.writerow({
            "issue": issue_row["id"],
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "published_at": row["published_at"],
            "reason": row["verdict_reason"],
            "mode": issue_row["mode"],
            "score": f"{row['issue_score']:.3f}" if row["issue_score"] is not None else "",
        })


# ── find ──

def _open_issue(conn, topics: list[str], today: date):
    """Fetch the next window and record its issue, unfinished; ``None`` when
    nothing was published in it."""
    start, end = issue.next_window(conn, today)
    fetches = issue.fetch_window(conn, topics, start, end)
    issue.embed_pending(conn)
    if not issue.pool(conn, start, end):
        print(f"No candidates published between {start} and {end} to pick from.",
              file=sys.stderr)
        return None

    issue_id = store.create_issue(conn, window_start=start.isoformat(), window_end=end.isoformat())
    for f in fetches:
        store.record_fetch(conn, issue_id, topic=f.topic, day=f.day.isoformat(),
                           status=f.status, returned=f.returned, error=f.error)
    conn.commit()
    return store.last_issue(conn)


def _summarise(conn, issue_row, pool, picks, modes) -> None:
    fetches = store.issue_fetches(conn, issue_row["id"])
    judged = store.issue_verdict_count(conn, issue_row["id"])
    fits = sum(store.get_row(conn, r["id"])["verdict"] == 1 for r in pool)
    chosen = ", ".join(f"{count} {mode}" for mode, count in modes.most_common())
    print(f"issue {issue_row['id']} ({issue_row['window_start']} to {issue_row['window_end']}): "
          f"judged {judged} of {len(pool)} candidates (budget {conf.verdicts_per_issue()}), "
          f"{fits} fit, picked {len(picks)} by {issue_row['mode']}; "
          + (f"chosen by {chosen}; " if chosen else "")
          + f"{sum(f['status'] == 'saturated' for f in fetches)} saturated and "
          f"{sum(f['status'] == 'failed' for f in fetches)} failed day queries",
          file=sys.stderr)


def _run_issue(conn, last, topics, today: date, n: int, *, agent: bool,
               answer: qualify.AgentAnswer | None) -> int:
    brief.require(conn)
    if not agent and (unset := llm.missing_settings()):
        print(f"error: set {', '.join(unset)} — the model judges every article "
              f"(or pass --agent-qualify to judge them yourself).", file=sys.stderr)
        return 1
    rules = brief.ensure_rules(conn, agent=agent)

    issue_row = last if last is not None and last["finished_at"] is None else None
    if issue_row is None:
        issue_row = _open_issue(conn, topics, today)
        if issue_row is None:
            return 0
    else:
        issue.embed_pending(conn)
    pool = issue.pool(conn, date.fromisoformat(issue_row["window_start"]),
                      date.fromisoformat(issue_row["window_end"]))

    qualifier = qualify.qualifier_for(conn, anchors.ensure(conn, rules=rules, agent=agent))
    modes = qualify.judge_pool(conn, issue_row, pool, qualifier, rules=rules,
                               agent=agent, answer=answer)
    mode, picks = qualify.rank_fits(conn, pool, qualifier, n)
    store.finish_issue(conn, issue_row["id"], mode=mode, picks=picks)

    finished = store.last_issue(conn)
    _write_issue(finished, store.issue_picks(conn, finished["id"]))
    _summarise(conn, finished, pool, picks, modes)
    return 0


def _print_pending(pending: qualify.QualifyPending, n: int) -> None:
    print(f"error: qualify_pending: {pending}", file=sys.stderr)
    print(json.dumps(pending.payload), file=sys.stderr)
    print(f"answer with: outnews find {n} --agent-qualify --verdict fit|no-fit --reason MSG",
          file=sys.stderr)


def cmd_find(n: int, *, brief_path: str | None = None, agent: bool = False,
             answer: qualify.AgentAnswer | None = None) -> int:
    topics = conf.topics()
    if not topics:
        print("OPENOUTNEWS_TOPICS is not set — nothing to fetch.", file=sys.stderr)
        return 0
    try:
        conf.edition()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    today = fetch.google_today()
    with store.connect() as conn:
        if brief_path is not None:
            brief.adopt(conn, Path(brief_path).read_text())
        last = store.last_issue(conn)
        if last is not None and last["finished_at"] and last["window_end"] == today.isoformat():
            _write_issue(last, store.issue_picks(conn, last["id"]))
            print(f"issue {last['id']} was already built today — printed again", file=sys.stderr)
            return 0
        try:
            return _run_issue(conn, last, topics, today, n, agent=agent, answer=answer)
        except (brief.MissingBrief, qualify.NothingPending) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except qualify.QualifyPending as pending:
            _print_pending(pending, n)
            return 1


# ── label ──

def cmd_label(article_id: str, verdict: str) -> int:
    if verdict not in ("read", "skip"):
        print(f"unknown label {verdict!r} — use 'read' or 'skip'", file=sys.stderr)
        return 1
    label = 1 if verdict == "read" else 0
    with store.connect() as conn:
        row = store.get_row(conn, article_id)
        if row is None:
            print(f"no article with id {article_id}", file=sys.stderr)
            return 1
        store.label_article(conn, article_id, label)
    return 0


# ── main ──

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="outnews")
    commands = parser.add_subparsers(dest="command", required=True)

    find = commands.add_parser("find")
    find.add_argument("n", nargs="?", type=int, default=10)
    find.add_argument("--brief", metavar="PATH")
    find.add_argument("--agent-qualify", action="store_true")
    find.add_argument("--verdict", choices=["fit", "no-fit"])
    find.add_argument("--reason")

    label = commands.add_parser("label")
    label.add_argument("id")
    label.add_argument("verdict")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(OVERVIEW)
        return 0

    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command == "label":
        return cmd_label(args.id, args.verdict)

    answer = None
    if args.verdict or args.reason:
        if not (args.agent_qualify and args.verdict and args.reason):
            print("error: --verdict and --reason go together, with --agent-qualify",
                  file=sys.stderr)
            return 1
        answer = qualify.AgentAnswer(fit=args.verdict == "fit", reason=args.reason)
    return cmd_find(args.n, brief_path=args.brief, agent=args.agent_qualify, answer=answer)
