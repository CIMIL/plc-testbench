# Using the listening Study Hub

The **Study Hub** is the operator's workspace for collecting and reviewing
MUSHRA listening ratings. It is separate from the participant listening test.
Open it through **Perform listening test** in run progress or the run history,
or visit `/listening-studies/<run-id>` in OpenPLC Studio.

## Quick workflow

1. Configure a run with one **HumanCalculator**, one loss simulator, and your PLC conditions.
2. Execute it and wait for **Manual action required** / `awaiting_input`.
3. Open the hub and review **Instructions & consent**.
4. Create individual invitations and copy the links.
5. Listeners open their links, consent, rate every trial, and submit.
6. Review participant sessions and, if needed, explicitly exclude responses.
7. **Close collection** to freeze responses and start finalization.
8. Inspect the final results and export the data.

!!! warning "Protect the management workspace"
    Owner management is not authenticated by the platform. Protect general and
    management APIs at the deployment boundary before recruiting publicly.
    Participant invitations provide session-scoped access, not owner authentication.
    Study-specific ethical approval and appropriate consent remain your responsibility.

## 1. Prepare the run

A platform run permits **at most one HumanCalculator**. When it is present,
**exactly one loss-simulator instance** is required. Automated-only runs can
still use multiple simulators.

The resulting study covers all selected tracks and PLC conditions. Each listener
completes the full set of trials; invitations do not assign subsets. A trial
compares PLC versions of the same excerpt, with a hidden reference and an anchor
as quality-control items. The original excerpt is also available as the explicit
playback reference.

After automated processing finishes, the run waits for human input without
holding a computation worker. If the hub says **No prepared listening studies**,
check run progress: processing must finish successfully first. Preparation warnings
can explain why fewer eligible excerpts were available than requested.

See [Human listening tests](human-listening.md) for excerpt selection, audio
preparation, settings, and standalone export/import details.

## 2. Understand the hub

**Listening study** identifies the MUSHRA test and its collection state:

| State | Meaning |
| --- | --- |
| **Collecting responses** | Invitations, participant saves, and owner inclusion decisions are allowed. |
| **Collection closed** | Responses and inclusion decisions are frozen; finalization may still be pending. |
| **Results finalized** | Final human artifacts have been written. |

Under **Manage collection**, summary cards show trials per listener, PLC condition
count, and the hidden-reference/anchor items. Every listener rates every assigned
item in every trial. **Run progress** and **Run analysis** return to the associated
run views. Ready automated results can be inspected while collection is open.

Older runs may contain multiple studies. Their hub retains a study selector;
manage each separately and close all of them before finalizing the run. New runs
use the single-study workflow above.

## 3. Review instructions and consent

Before generating any invitations, expand **Review instructions & consent**.

- Explain what listeners should do, including the quality scale: **0 = bad,
  100 = excellent**.
- Supply appropriate study-specific contact, retention, and consent information.
- Enter plain text, paste formatted text, or paste HTML. Use the previews to check
  how it will appear to participants.
- Basic headings, paragraphs, emphasis, and lists are supported. Scripts, styles,
  attributes, links, and media are removed; link text remains readable.
- Each field is limited to 10,000 characters, including markup, and must contain
  visible text.
- Click **Save instructions and consent** before creating invitations.

!!! important "The first invitation locks the protocol"
    Instructions and consent cannot be edited after the first invitation is
    created, even if nobody has opened it yet. Each session records the consent
    version it acknowledges. A changed published study definition requires a new run.

## 4. Create a participant session

Set **Individual invitations** to a number from **1 to 100**, then click
**Create invitations**. Each generated link creates a separate pseudonymous
participant session immediately; the participant table therefore includes
invitations that have not been opened yet.

1. Copy the links using **Copy links**, or copy them from the displayed text box.
2. Send **one unique link per listener** through your own communication channel.
3. Ask each listener to retain their link for resuming later.

Links are private and **cannot be retrieved again** after they leave the current
view. The platform does not email invitations. A link belongs to one resumable
session: sharing it between listeners would share that session, not create new
participants.

### Participate yourself

Click **Create my listening link**. This creates a normal participant invitation;
it does not bypass consent or completion requirements. With one newly generated
link, click **Open participant listening test** to open it in a separate tab.

## 5. What the listener does

1. Open the private invitation link in a desktop browser. Chrome/Edge are recommended;
   use headphones and a quiet environment.
2. Read the instructions and consent, select the agreement checkbox, and click
   **Agree and continue**.
3. Play the explicit reference and anonymous rated items. Switch between items,
   adjust the common playback volume, and rate audio quality from 0 to 100.
4. Rate every item before moving to the next trial. **Zero is a valid score**;
   an untouched rating is unanswered, not zero.
5. Use **Previous trial** to review earlier answers and **Save checkpoint** to
   explicitly save progress. The player also saves during normal interaction.
6. On the last trial, click **Finish and submit**. Completion is confirmed only
   after the server acknowledges persistence.

Allow roughly **1–2 minutes per trial**. Breaks are recommended every ten trials.
Listeners can reopen the same invitation to resume saved progress with the same
trial order, condition order, and ratings. Do not use several tabs for the same
session: conflicting revisions require an explicit reload rather than overwriting
another tab's answers.

If a save fails, follow the player's retry/error instructions and wait for an
acknowledgement before leaving. Local draft recovery helps with interrupted saves,
but is not a substitute for server confirmation. Completed sessions are read-only.
Revocation or study closure also stops further edits.

## 6. Review participants

**Participants** shows the number of sessions. With none, an empty-state message
replaces the table. Once invitations exist, the table shows:

| Column | What to look for |
| --- | --- |
| **Pseudonymous session** | Session identifier, not a participant's real name. |
| **State** | Invited, draft, completed, or revoked access. |
| **Quality flags** | Advisory checks such as coverage or reference/anchor anomalies. |
| **Inclusion** | Included completed sessions, explicit exclusions, or unfinished sessions that are not analyzed. |
| **Owner actions** | Change inclusion with a reason, or revoke access while collecting. |

Flags are **advisory**: they neither automatically exclude a session nor prove
that its answers are invalid. Review them before deciding what to include.

### Exclude or re-include answers

Enter a reason and click **Exclude**. Raw responses remain retained, while the
session is omitted from analysis. For an excluded session, enter a reason and
click **Include again** to reverse that decision while collection is open.
Re-inclusion does not make an unfinished session eligible: it must also be completed.

### Delete an unused invitation

While collecting, **Delete invitation** permanently removes an untouched
invitation and its row after confirmation. Its link immediately stops working.
Deletion is allowed only before consent or any saved progress and only without
exclusion history. Draft and completed sessions cannot be deleted this way;
use revocation or exclusion instead. Deleting every unused invitation does not
unlock instructions or consent. Pagination updates to reflect the remaining rows.

### Revoke a link

Click **Revoke access** and confirm to stop further access through that invitation.
Existing answers remain retained. **Revocation is not exclusion**: use **Exclude**
if completed responses should not contribute to analysis.

## 7. Read results and download data

During collection, results are labeled **Provisional** and can change as listeners
finish or inclusion decisions change. The results view includes participant counts,
condition summaries, individual-rating distributions, confidence intervals, and
participant-aware comparisons when the data supports them.

Missing or unestimable values are shown as unavailable, not invented zeros.
Reference and anchor ratings are quality controls rather than primary PLC
comparisons. Too few participants or an unidentifiable statistical model can leave
inference unavailable even when descriptive results are present.

### Statistical comparisons: LMM + Tukey

New analyses fit a **linear mixed-effects model**, for both balanced and incomplete
selected data. PLC condition is a fixed effect; source track and condition-by-track
effects are included when there are multiple tracks. Random intercepts account for
listeners, each listener's trial context (including repetitions), and shared audio
excerpts. Participant-condition random deviations allow listener preferences to
vary; if they cannot be fitted, a documented random-intercept fallback is attempted.
The same excerpt is shared across listeners, not treated as a new independent
piece for each listener.

The overall test is a maximum-likelihood likelihood-ratio test of condition and
condition-by-track effects jointly. Post-hoc comparisons use **estimated marginal
condition means**, averaging selected tracks equally, with **Tukey-adjusted
p-values and simultaneous 95% intervals** over all PLC pairs. Five PLC conditions
produce ten comparisons. Reference and anchor are outside this **primary** family.

The current Python implementation uses an **asymptotic studentized-range
approximation** (infinite degrees of freedom), not finite-sample
Satterthwaite/Kenward–Roger inference. Multiplicity calibration is approximate
when contrast variances differ; small studies need particular caution. Model
inference requires at least five listeners, three distinct excerpts and two PLC
conditions, but these thresholds do not ensure adequate power. The model assumes
approximately continuous scores and ignorable missingness, and does not repair
non-random missingness or establish generalization to all source tracks.
Descriptive listener-bootstrap intervals are separate from these model contrast
intervals. If the joint observed-information covariance is ill-conditioned but
fixed effects remain estimable, a documented asymptotic GLS covariance conditional
on the fitted variance components is used; unestimable models remain unavailable.
Existing development test studies were reanalyzed once with this
method; the application no longer renders legacy ANOVA results.

### Finalized-study charts

After finalization, the Hub presents the full report: text, two summary/comparison
tables, a descriptive **Boxplot / Violin** chart, and four statistical/listener
plots. The shared **Show reference & anchor** toggle is **on by default**, applying
to **both tables and all five plots** in this order: **Anchor → PLC conditions
(configured order) → Hidden reference**.

With QC shown, the three inferential plots and pairwise table use a **separate, clearly labeled exploratory
QC-inclusive LMM and Tukey comparison family**. Two PLC conditions plus reference
and anchor produce six comparisons, rather than the primary family's one. Even
PLC-to-PLC intervals and p-values may change because the model and family change.
Controls remain quality controls, not PLC performance conditions; their endpoint
scores merit extra caution about continuous-score model assumptions. Switching
the toggle off restores the frozen PLC-only model results. The descriptive table
adds frozen participant-bootstrap QC summaries; its PLC summary cells are unchanged.
The primary frozen analysis itself is never overwritten by the display selection.

The plots cover all source tracks (the distribution plot also offers a per-track
aggregation):

- **Listener score distributions:** choose Boxplot or Violin and overall or per-track
  aggregation. Each observation is a completed, included listener's track-balanced
  mean. These are descriptive distributions, not confidence intervals.

- **Estimated marginal means:** LMM-estimated quality scores with pointwise 95%
  normal-Wald confidence intervals. These are not Tukey-adjusted intervals;
  overlapping mean intervals do not determine pairwise significance.
- **Tukey-adjusted pairwise differences:** a horizontal forest plot of first-minus-
  second effects with simultaneous 95% intervals and a dashed zero line. Filled
  circles indicate adjusted p < .05; hollow diamonds indicate nonsignificance.
- **Pairwise significance heatmap:** adjusted p-values over the full comparison
  family. Symbols identify p < .001 (***), .01 (**), or .05 (*); `ns` means
  nonsignificant, not equivalent. Diagonal cells are not tests, and `?` means
  unavailable. Tooltips show signed row-minus-column effects and intervals.
- **Participant-level scores:** one descriptive line per included listener,
  preserving genuine zero and missing-score gaps. Repetitions are averaged within
  excerpts, excerpts within tracks, and tracks equally within each listener.
  These are listener averages, not model predictions or independent trial samples.

All charts support tooltips, light/dark themes and responsive layouts, with
collapsible data tables for nonvisual inspection. Full condition identities remain
in tooltips and tables when long axis labels are shortened. QC labels are explicit;
PLC colors stay consistent across toggle changes. The toggle reuses the frozen
ratings download without additional requests. If the selected inference is
unavailable, the model charts explain why while the
participant chart can still display valid scores. A failed ratings download has
its own retry button and does not hide available model results. Distribution and
listener plots share one frozen ratings download; changing QC scope preserves the
selected plot style and aggregation.

In the analyzer, a selected finalized **HumanCalculator** shows **only these five
plots**, their controls, legends, tooltips and a compact primary/exploratory scope
label. Report tables, explanatory sections and model diagnostics are reserved for
the Study Hub. Unavailable/error notices remain visible in either view.

Two export controls sit below the participant section:

- **Export ratings → CSV:** raw ratings for spreadsheets.
- **Export ratings → JSON:** raw ratings for data processing.
- **Session details:** JSON with consent records, exact presentation orders, and
  session metadata.

Raw exports retain auditable session/response information, including excluded
and unfinished data where available. They are not simply the final inclusion set;
use the inclusion/completion metadata when performing your own analysis. Participants
do not see other listeners' ratings or the identities of the anonymous conditions.

## 8. Close collection and finalize

When you are ready, click **Close collection** and review the confirmation.

!!! warning "Closure is permanent"
    Closing freezes invitations, submissions, unfinished drafts, and inclusion
    decisions. Links stop accepting edits. Only **completed, non-excluded** sessions
    contribute to the final analysis; unfinished sessions remain auditable but are
    not analyzed. There is no reopening or automatic participant-count cutoff.

If no completed, included sessions exist, you must explicitly confirm empty results.
Their statistics remain unavailable rather than zero.

For a new single-study run, closure queues finalization. The run transitions through
queued/running to completed while importing frozen ratings and writing the human
artifacts. **PLC reconstruction and automated metrics are not rerun.** Historical
multi-study runs wait until every study is closed.

If finalization fails, use **Finalize / retry finalization** when offered. It retains
frozen responses. The generic adjust-and-retry workflow instead creates a new run
and does not inherit invitations or participant responses.

After finalization, inspect the frozen results in the hub or **Run analysis**.
The hub stays available for read-only review and exports.
