# OpenOutNews

Explore/exploit news curation for a newsletter: fetch candidate articles,
learn what a reader engages with, print a CSV of what to send next.

The scoring mechanism — a GP regressor over article embeddings, ranked by
posterior fit probability once enough labels exist and by BALD uncertainty
before that — is [OpenOutLearn](https://github.com/eracle/OpenOutLearn)'s
`GPBaldQualifier`, the same engine [OpenOutFind](https://github.com/eracle/OpenOutFind)'s
lead qualifier subclasses. Here the label is reader engagement (read/skip)
instead of an LLM's ICP-fit verdict — see `openoutnews/ml/qualifier.py`.

This is a first version: a CSV in, a CSV out, no wizard, no send step.

## Install

```bash
pip install -e .
```

## Configure

```bash
export OPENOUTNEWS_TOPICS="open source AI,B2B sales tools"
```

## Use

```bash
outnews find 10 > picks.csv
# ... send the newsletter, see what the reader engaged with ...
outnews label <id> read
outnews label <id> skip
```

Each `find` run fetches fresh candidates for the configured topics, embeds
the ones it hasn't seen, and picks by whichever side of explore/exploit the
label balance currently favors — newest-first until at least one "read" and
one "skip" exist.
