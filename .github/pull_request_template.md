## What & why

<!-- One paragraph. What changed, and what problem it solves. Link the plan.md section, e.g. "§2 Registration & heartbeat". -->

## How it works

<!-- The design decision a reviewer needs to hold in their head to read the diff. Not a file-by-file summary — the diff already says that. -->

## How I verified it

<!-- Commands run, output observed. "Done when:" from the plan section, demonstrated. -->

```
$ pytest -q

```

## Review notes

<!-- Where you want the reviewer to look hardest. Anything you're unsure about. Anything deliberately deferred. -->

---

### Self-review checklist

Do this pass in the **Files changed** tab before merging, not here.

- [ ] I read the full diff top to bottom in the GitHub UI, not just my editor
- [ ] Every new branch/error path is either tested or consciously accepted
- [ ] Concurrency: shared state is behind a lock; no `await` while holding one
- [ ] No secrets, hardcoded hosts/ports, or debug prints left behind
- [ ] Names match the vocabulary already used in `common/schemas.py`
- [ ] The PR does one thing — nothing snuck in that belongs in its own PR
- [ ] CI is green
