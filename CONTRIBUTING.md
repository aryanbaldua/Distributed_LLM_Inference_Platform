# Working on this repo

This is a solo project run with a team's process on purpose. `main` is protected:
you cannot push to it directly, and nothing lands without a green CI run and a
deliberate review pass. The point is the habit, not the ceremony.

## One unit of work = one PR

Size a PR to one checklist section of `plan.md` — small enough that you can hold
the whole diff in your head, big enough to be demonstrable against that section's
**Done when:** line. If a PR has two "and"s in its title, it's two PRs.

## The loop

```bash
git checkout main && git pull
git checkout -b feat/worker-registration      # see naming below

# ... work, committing as you go ...

git push -u origin feat/worker-registration
gh pr create --fill                            # opens the PR template
```

Then, on the PR:

1. **Walk away from it.** Even ten minutes. Reviewing code you just wrote with the
   authoring context still hot is how bugs get waved through. This is the single
   highest-value part of the ritual and the easiest to skip.
2. **Review it in the GitHub UI**, in the *Files changed* tab — never in your
   editor. The unfamiliar rendering is doing real work: it strips the mental model
   you had while writing and forces you to read what's actually there.
3. **Leave real line comments.** Click a line, write the comment, *Start a review*.
   Write them addressed to someone else: "why is this dict access not under the
   lock?" — not "TODO fix." If you can't justify a line in a sentence, that's the
   finding.
4. **Submit the review as "Comment."** GitHub does not permit approving your own
   PR, and that's fine — a Comment review is recorded on the PR permanently, which
   is all you need. The approval gate here is CI plus your own judgment.
5. **Act on your own findings** — push fixes as new commits so the review thread
   stays readable, and reply to each comment with what you did.
6. **Merge with Squash and merge.** One commit per PR on `main` keeps `git log`
   readable as a project narrative. Delete the branch.

### When Claude wrote the code

The review step matters more, not less. Read for:

- **Invented requirements** — behavior the plan never asked for, quietly added.
- **Plausible-but-wrong concurrency** — code that looks like it locks correctly.
- **Tests that assert the implementation** rather than the behavior, so they'd
  pass even if the logic were wrong.
- **Error paths you can't trace** — if you can't say what happens when the
  downstream call times out, that's a comment, not a merge.

Do not approve a diff you could not have written yourself. Ask for the explanation
until you could.

## Branch naming

| Prefix | For |
| --- | --- |
| `feat/` | new behavior |
| `fix/` | bug fixes |
| `refactor/` | no behavior change |
| `test/` | tests only |
| `chore/` | tooling, CI, deps, docs |

## Commits

Imperative mood, no trailing period, explain *why* in the body when the *what*
isn't self-evident.

```
Expire workers whose heartbeat is older than the timeout

The registry previously kept returning stale workers to the scheduler
because expiry only ran on write. Check freshness on read instead.
```

## Running things locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

pytest -q                # what CI runs
ruff check . && ruff format --check .
./scripts/run_local.sh   # controller + workers
```

Run those before you push. CI is a backstop, not your first signal.
