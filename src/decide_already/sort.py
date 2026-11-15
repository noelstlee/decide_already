"""Sorting: the placement rules, the model's judgment of one file, and the run over the dump.

The model only gives probabilities. Code decides what they mean (`decide`), so Unsorted and
close calls are rules with cutoffs from the config, never an option the model can pick.
"""

from collections.abc import Callable
from pathlib import Path

from . import vault
from .backends.base import (
    Backend,
    BackendError,
    BudgetExceeded,
    RunStopped,
    SortRequest,
    SortResult,
)
from .card import VARIANTS, Card, build_card, hash_file, render_card
from .config import JEV_LOG_FILE, Config, decide_path
from .log import total_input_tokens
from .types import Group, GroupSet, Placement, StrictModel, UnsortedReason

NOT_REACHED = "not sorted: the spend cap stopped the run before this file"
STOPPED_BEFORE = "not sorted: the run was stopped before this file"
NO_NOTE = (
    "not sorted: you moved or froze this file, but its card note is gone or can no longer be "
    "read. Put the note back, or run `p0 unfreeze` with this file's id to have it sorted "
    "from scratch"
)
CHANGED_BY_HAND = (
    "not sorted: its card note was changed by hand while the sort was running, so the note "
    "was left as it is. The next sync picks the change up"
)


class Decision(StrictModel):
    """What the placement rules make of one answer."""

    group_id: str | None  # None means Unsorted
    close_call: bool  # placed, but the top two groups are near each other
    reason: UnsortedReason | None  # why it is Unsorted; None when placed


class SortReport(StrictModel):
    """What one run of sort_vault did."""

    set_version: int  # the set version every answer in this run was computed under
    asked: int  # files sent to the model, answered or not
    placed: int  # files put in a group
    unsorted: int  # files put in Unsorted, by the rules or for having no text
    close_calls: int  # placed files flagged as close calls
    frozen_skipped: int  # frozen files left as they were
    errors: dict[str, str]  # file id to message, for each file that could not be sorted
    spent_usd: float | None  # this vault's total so far, from its Jev log; None without a log


def decide(probs: dict[str, float], fits_any: float, config: Config) -> Decision:
    """Apply the placement rules to one answer.

    Unsorted when fits_any is under its cutoff (or the backend made no pick), then when the top
    probability is under its cutoff. Otherwise the file goes to the top group, flagged as a
    close call when the top two are nearer than the margin. A value exactly at a cutoff counts
    as at or over. Ties go to the lower group id. With one group, the second probability is 0.
    """
    if not probs or fits_any < config.unsorted_fits_any_below:
        return Decision(group_id=None, close_call=False, reason="no_group_fits")
    ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    top_id, top_prob = ranked[0]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_prob < config.unsorted_top_below:
        return Decision(group_id=None, close_call=False, reason="weak_top")
    close_call = top_prob - second_prob < config.close_call_margin
    return Decision(group_id=top_id, close_call=close_call, reason=None)


def _total(values: list[float | None]) -> float | None:
    """Add up what the backend reported per request. None when it reported nothing."""
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None


def judge(
    card: Card,
    groups: list[Group],
    set_version: int,
    backend: Backend,
    variant: str,
    per_state: list[SortResult] | None = None,
) -> SortResult:
    """Ask the backend about one file: one request per state of its card, answers averaged.

    Only the window vote has several states. Probabilities and fits_any are averaged over
    them. Tokens and latency are added up, and stay None when the backend reports none. The
    option order, model and request id are those of the first request. The names-only variant
    sends the groups without their descriptions. When `per_state` is a list, each state's own
    answer is appended to it. A card with no states has nothing to ask about: ValueError.
    """
    if not card.states:
        raise ValueError(f"{card.file.id} has no text, so there is nothing to ask the model")
    results = []
    for state in card.states:
        request = SortRequest(
            file_id=card.file.id,
            state=state,
            groups=groups,
            set_version=set_version,
            variant=variant,
            with_descriptions=variant != "names_only",
        )
        results.append(backend.sort(request))
        if per_state is not None:
            per_state.append(results[-1])
    first = results[0]
    # A backend with no pick answers with no probabilities; a group it left out counts as 0.
    group_ids = list(dict.fromkeys(group_id for result in results for group_id in result.probs))
    return SortResult(
        file_id=card.file.id,
        probs={
            group_id: sum(result.probs.get(group_id, 0.0) for result in results) / len(results)
            for group_id in group_ids
        },
        fits_any=sum(result.fits_any for result in results) / len(results),
        model=first.model,
        option_order=first.option_order,
        input_tokens=_total([result.input_tokens for result in results]),
        output_tokens=_total([result.output_tokens for result in results]),
        request_id=first.request_id,
        latency_ms=_total([result.latency_ms for result in results]),
    )


def _place(card: Card, group_set: GroupSet, backend: Backend, config: Config) -> Placement:
    """Work out one file's placement: from the model's answer and the rules, or by code alone.

    A card with no states (a file with no text) goes to Unsorted without asking the model.
    """
    shared = {
        "source": vault.source_link(card.file.id, config),
        "file_id": card.file.id,
        "content_hash": card.file.content_hash,
        "file_type": card.file.file_type,
        "set_version": group_set.version,
        "variant": config.variant,
    }
    if not card.states:
        return Placement(**shared, group_id=None, placed_by="code", reason="no_text")
    result = judge(card, group_set.groups, group_set.version, backend, config.variant)
    decision = decide(result.probs, result.fits_any, config)
    return Placement(
        **shared,
        group_id=decision.group_id,
        placed_by="jev",
        close_call=decision.close_call,
        reason=decision.reason,
        probs=result.probs,
        fits_any=result.fits_any,
        model=result.model,
    )


def sort_vault(
    root: Path, backend: Backend, on_progress: Callable[[int, int], None] | None = None
) -> SortReport:
    """Sort the dump against the current groups and write a card note for each file.

    Edits made by hand are picked up first (corrections.sync), so a note that was dragged or
    frozen in Obsidian is never overwritten. A frozen file whose content has not changed is
    skipped. A frozen file whose content did change loses its freeze (label "unfrozen").
    Every other file is carded with the config's variant, judged, decided and written, so a
    rerun asks about every unfrozen file again, against the full current set. A file that
    cannot be read or written, or that the backend fails on, keeps its previous note (if any)
    and is listed in the report's errors. So is a file the user labeled whose card note is
    gone: it is not sent to the model. And so is a file whose card note was changed by hand
    while the run was under way: that note is left as it is, for the next sync to read.
    When the backend stops the run (the spend cap is reached, or the call log cannot be
    written), that file and every file after it are listed too. `on_progress(done, total)`
    is called after each file that was handled.

    Raises ValueError when the vault has no groups or the config names an unknown variant,
    and whatever corrections.sync raises (FileExistsError when a note cannot be moved).
    """
    # Imported here and not at the top: corrections needs judge and decide from this module.
    from . import corrections

    corrections.sync(root)
    config = Config.load(root)
    if config.variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {config.variant!r} in the vault config; "
            f"pick one of: {', '.join(VARIANTS)}"
        )
    group_set = vault.sync_group_set(root)
    if not group_set.groups:
        raise ValueError("there are no groups yet: add one with `p0 group add` before sorting")
    notes = vault.read_card_notes(root)
    note_of = {note.placement.file_id: note for note in notes}
    labeled = corrections.active_labels(root)
    files = vault.dump_files(root, config)
    report = SortReport(
        set_version=group_set.version,
        asked=0,
        placed=0,
        unsorted=0,
        close_calls=0,
        frozen_skipped=0,
        errors={},
        spent_usd=None,
    )
    for done, (file_id, path) in enumerate(files, start=1):
        note = note_of.get(file_id)
        try:
            frozen = note is not None and note.placement.frozen
            if note is None and file_id in labeled:
                # The user's label stands, so the model is not asked. The user decides.
                report.errors[file_id] = NO_NOTE
            elif frozen and note.placement.content_hash == hash_file(path):
                report.frozen_skipped += 1
            elif note is not None and not vault.unchanged(note):
                report.errors[file_id] = CHANGED_BY_HAND
            else:
                if frozen:
                    # New content, which the freeze did not cover: the freeze comes off first.
                    corrections.unfreeze_note(root, note, group_set.version)
                    note = vault.restamp(note)  # the tool's own rewrite is not an edit by hand
                card = build_card(path, file_id, config)
                placement = _place(card, group_set, backend, config)
                if placement.placed_by == "jev":
                    report.asked += 1
                if note is not None and not vault.unchanged(note):
                    # Changed while the model was asked. The answer is dropped, not the edit.
                    report.errors[file_id] = CHANGED_BY_HAND
                else:
                    # The notes read at the start are passed along, so that writing one
                    # note does not mean reading all of them again.
                    vault.write_card_note(root, placement, render_card(card), notes)
                    if placement.group_id is None:
                        report.unsorted += 1
                    else:
                        report.placed += 1
                        if placement.close_call:
                            report.close_calls += 1
        except RunStopped as error:
            report.errors[file_id] = str(error)
            later = NOT_REACHED if isinstance(error, BudgetExceeded) else STOPPED_BEFORE
            for later_id, _ in files[done:]:
                report.errors[later_id] = later
            break
        except BackendError as error:
            report.asked += 1  # the request went out, even though no usable answer came back
            report.errors[file_id] = str(error)
        except OSError as error:  # the original could not be read, or its note not written
            report.errors[file_id] = str(error)
        if on_progress is not None:
            on_progress(done, len(files))
    log_path = decide_path(root, JEV_LOG_FILE)
    if log_path.exists():
        report.spent_usd = total_input_tokens(log_path) / 1e6 * config.usd_per_million_tokens
    return report
