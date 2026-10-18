![OpenOutNews Logo](docs/logo.png)

# OpenOutNews

Explore/exploit news curation for a newsletter: fetch candidate articles,
have a model judge them against what the newsletter covers, print a CSV of
what to send next.

The scoring mechanism — a GP regressor over article embeddings, ranked by
posterior fit probability once enough verdicts exist and by BALD uncertainty
before that — is [OpenOutLearn](https://github.com/eracle/OpenOutLearn)'s
`GPBaldQualifier`, the same engine [OpenOutFind](https://github.com/eracle/OpenOutFind)'s
lead qualifier subclasses. As there, the label is an LLM's verdict — here,
whether an article fits the newsletter's brief — see `openoutnews/ml/qualifier.py`.

A CSV out, no wizard, no send step.

## Install

```bash
pip install -e .
```

## Configure

```bash
export OPENOUTNEWS_TOPICS="open source AI,local LLMs"
export OPENOUTNEWS_BRIEF="Engineers who self-host open-weight models. Releases, benchmarks,
tooling, licences. Not funding rounds, not opinion pieces, not consumer chatbot news."
export OPENOUTNEWS_AI_MODEL="anthropic:claude-haiku-5-5"   # or openai:<model>
export OPENOUTNEWS_LLM_API_KEY="..."
export OPENOUTNEWS_EDITION="US:en"          # default Google News edition, e.g. "IT:it";
                                            # a topic names its own as "Italian wine@GB:en"
export OPENOUTNEWS_VERDICTS_PER_ISSUE=150   # articles an issue may put to the model
export OPENOUTNEWS_ISSUE_DAYS=7             # how far back the first issue reaches
export OPENOUTNEWS_DEDUP_LOOKBACK_DAYS=28   # how far back "already sent" reaches
```

## Use

```bash
outnews find 10 > issue.csv
outnews find 10 --brief rules.txt > issue.csv     # judge by rules you wrote
# ... send the newsletter, see what the reader engaged with ...
outnews label <id> read
outnews label <id> skip
```

Each `find` builds the next **issue**. Its window runs from where the last
issue ended (or `OPENOUTNEWS_ISSUE_DAYS` back, for the first) to today, in
Google News' Pacific-time days. Each topic is fetched one day at a time,
since a Google News query returns at most 100 articles; a day that returns
100 is recorded as saturated, and a day whose query fails is recorded as a
gap without stopping the issue. Only articles published inside the window
are candidates. One that repeats an article sent in the last
`OPENOUTNEWS_DEDUP_LOOKBACK_DAYS` is absorbed into it, and syndicated copies
within the window collapse to the earliest.

**The brief.** The first `find` has the model turn `OPENOUTNEWS_BRIEF` into
judging rules — *A fit is* / *Not a fit* — and stores them; every verdict
reads both. `--brief PATH` hands rules in instead (written by a stronger
model, or by you) and keeps them for later issues. With neither, `find` stops
with an error rather than build an empty issue.

**The verdicts.** The model reads each article's headline, outlet and date —
a Google News search item carries nothing more — and says whether it fits,
with a one-line reason. It does not read every candidate: the qualifier picks
which ones, one at a time, up to `OPENOUTNEWS_VERDICTS_PER_ISSUE`. On a cold
start the model invents three ideal headlines (the anchors) and the first
verdicts go to the articles nearest them; while the anchors outnumber real
fits it picks by P(fit); after that it exploits once rejections outnumber
fits and explores by BALD while they are even. Each verdict is stored as it
lands, so a run that dies resumes the same issue without fetching again.

Each verdict also sees the headlines the issue has accepted so far and says
whether a fit reports one of those stories again. Whether two headlines are
one story is the model's call: no similarity threshold separates rewrites
from different stories. A repeat is never picked, and it teaches the
qualifier to move on rather than spend more verdicts on that story.

The issue's picks are the accepted stories, one article each, ranked by the
GP, with the reason in the CSV's `reason` column. Running `find` again on the day an issue
was finished prints that issue again.

**Reader labels** (`outnews label`) are stored in their own column and train
nothing yet: what the reader did is a different question from whether an
article fits.

### Judging from an agent

`--agent-qualify` calls no model. `find` stops at the article the qualifier
picks, prints `error: qualify_pending: …` and a JSON line with everything a
verdict reads, and exits 1; the agent answers with

```bash
outnews find 10 --agent-qualify --verdict fit --reason "Three new open-weight models from Nvidia"
```

(adding `--repeats ID` for a fit that reports a story in the payload's
`accepted` list), which records the verdict and stops at the next article, until the budget is
spent and the issue prints. Nothing writes a brief or anchors in this mode,
so hand rules in with `--brief`; without anchors the first verdicts go to the
newest articles.
