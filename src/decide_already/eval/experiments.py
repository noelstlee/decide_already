"""The Phase 0 experiments (build step 7): the extra sorting passes they need, and their numbers.

1. Group added mid-run. Every file is sorted with one group held out
   (eval.groups.HELD_OUT_GROUP_ID), under descriptions that do not name it
   (eval.groups.HELD_OUT_EXCLUDES). A user then adds the group, freezes each file that sits in
   its gold group, and reruns the rest against the full set.
2. Wording of the fits-any question. The same pass with group names only, to show what the
   group descriptions add when a file fits no group.
3. Description edits. Each edit in <vault>/description_edits.yaml gives one group a new
   description. Every file is sorted again under it and compared with its answer before.

Nothing here asks a model. The runner (run.py) puts each pass through the same cache and the
same Jev backend as every other configuration, and the report (report.py) calls the functions
below on what was answered. A pass is named after the configuration it copies:
"jev:card@without-exams", "jev:card@edit-<name>".
"""

import re
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import Field, ValidationError, field_validator

from ..types import Group, StrictModel
from .groups import HELD_OUT_EXCLUDES, HELD_OUT_GROUP_ID
from .metrics import Cutoffs, Prediction, outcome, top_group

EDITS_FILE = "description_edits.yaml"
HELD_OUT_BASES = ("jev:card", "jev:names_only")  # experiments 1 and 2
EDIT_BASE = "jev:card"  # experiment 3
# A pass changes the groups in memory only. The vault's own set stays as it is, so the requests
# of a pass are logged under set version 0, which no vault set ever has.
SET_VERSION = 0


class Edit(StrictModel):
    """One entry of description_edits.yaml: a new description for one group."""

    name: str  # short and plain: it is part of the pass name and of its cache file name
    group_id: str
    covers: str
    excludes: str | None = None  # None leaves the group's excludes as it is; "" removes it
    target_file_ids: list[str] = Field(default_factory=list)  # the files it is meant to fix
    note: str | None = None  # free text for the report: who wrote the edit, when and why

    @field_validator("name")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        """Keep names to lowercase letters, digits, dashes and underscores."""
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
            raise ValueError(
                f"{value!r} cannot name an edit: use lowercase letters, digits, - and _"
            )
        return value

    @field_validator("covers")
    @classmethod
    def _has_covers(cls, value: str) -> str:
        """Every group needs a covers description, so an edit must give one."""
        if not value.strip():
            raise ValueError("an edit needs a covers description")
        return value


class Pass(StrictModel):
    """One extra sorting pass: a configuration run again under a changed list of groups."""

    name: str  # the configuration name its answers are cached under
    base: str  # the configuration it copies, such as "jev:card"
    groups: list[Group]


def held_out_name(base: str) -> str:
    """The name of a configuration's pass without the held-out group."""
    return f"{base}@without-{HELD_OUT_GROUP_ID}"


def edit_name(edit: Edit) -> str:
    """The name of the pass that sorts under one description edit."""
    return f"{EDIT_BASE}@edit-{edit.name}"


def read_edits(root: Path) -> list[Edit] | None:
    """Read a vault's description_edits.yaml. None when the vault has no such file.

    Raises ValueError when the file is not a list of edits, or two edits share a name.
    """
    path = root / EDITS_FILE
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError as error:
        raise ValueError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(data, list):
        # Bad file content, so a ValueError like every other problem with the file.
        raise ValueError(f"{path} must hold a list of edits")  # noqa: TRY004
    try:
        edits = [Edit.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"{path} has an edit that cannot be read: {error}") from error
    names = [edit.name for edit in edits]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{path} has two edits named {', '.join(repeated)}")
    return edits


def without_group(groups: Sequence[Group], group_id: str) -> list[Group]:
    """The groups minus one."""
    return [group for group in groups if group.id != group_id]


def reduced_groups(groups: Sequence[Group]) -> list[Group]:
    """The groups of the held-out passes: the held-out group dropped, the others reworded.

    Each remaining group listed in HELD_OUT_EXCLUDES gets the excludes text given there, which
    does not name the held-out group. The other groups stay as they are (naming_held_out tells
    whether one of them still names it).
    """
    return [
        group.model_copy(update={"excludes": HELD_OUT_EXCLUDES[group.id]})
        if group.id in HELD_OUT_EXCLUDES
        else group
        for group in without_group(groups, HELD_OUT_GROUP_ID)
    ]


def naming_held_out(groups: Sequence[Group]) -> list[str]:
    """The names of the reduced set's groups whose description still names the held-out group.

    `groups` is the full set. A description names the group when its covers or excludes text
    holds the group's name as whole words, with or without a final s, capitals ignored:
    "exams" and "exam" name Exams, "example" does not. Empty when no description names it, or
    when the set has no such group.
    """
    held_out = next((group for group in groups if group.id == HELD_OUT_GROUP_ID), None)
    if held_out is None:
        return []
    stem = held_out.name.strip().lower().removesuffix("s")
    names = re.compile(rf"\b{re.escape(stem)}s?\b", re.IGNORECASE)
    return [
        group.name
        for group in reduced_groups(groups)
        if names.search(f"{group.covers} {group.excludes or ''}")
    ]


def with_edit(groups: Sequence[Group], edit: Edit) -> list[Group]:
    """The groups with one description edit applied. Raises ValueError for an unknown group."""
    if edit.group_id not in {group.id for group in groups}:
        raise ValueError(
            f"edit {edit.name!r} is for group {edit.group_id!r}, which this vault does not have"
        )
    changed: dict[str, str | None] = {"covers": edit.covers.strip()}
    if edit.excludes is not None:
        changed["excludes"] = edit.excludes.strip() or None
    return [
        group.model_copy(update=changed) if group.id == edit.group_id else group for group in groups
    ]


def passes(groups: Sequence[Group], edits: Sequence[Edit] | None) -> list[Pass]:
    """The extra sorting passes the experiments need under a vault's groups.

    Experiments 1 and 2 need the card and the names-only configurations under the reduced set
    (see reduced_groups). The names-only pass sends no descriptions, but it runs under the same
    groups, so both passes share one fingerprint in the cache. They are left out when the
    vault does not have the held-out group, or has no other one.
    Experiment 3 needs the card configuration once per description edit. An edit for a group
    the vault does not have raises ValueError.
    """
    extra: list[Pass] = []
    reduced = reduced_groups(groups)
    if reduced and len(reduced) < len(groups):
        for base in HELD_OUT_BASES:
            extra.append(Pass(name=held_out_name(base), base=base, groups=reduced))
    for edit in edits or []:
        extra.append(Pass(name=edit_name(edit), base=EDIT_BASE, groups=with_edit(groups, edit)))
    return extra


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """The area under the ROC curve of a score, from the scores of the two kinds of item.

    It is the chance that a positive item scores above a negative one, with a tie counted as
    half: 1.0 when every positive is above every negative, 0.5 when the score says nothing.
    None when either kind is missing.
    """
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def group_added(
    reduced: Sequence[Prediction], full: Sequence[Prediction], cutoffs: Cutoffs
) -> dict:
    """The numbers of experiments 1 and 2, from one pass without the held-out group.

    `reduced` holds a configuration's answers under the groups minus the held-out one, and
    `full` its answers under every group. The gold labels are the usual ones, so a file of the
    held-out group has a gold group that the reduced pass could not pick.

    - held_out_n, held_out_unsorted_share, held_out_mean_fits_any: the held-out group's files,
      which fit no group of the reduced set. The share the cutoffs send to Unsorted (the right
      outcome for them), and their mean fits_any.
    - others_n, others_top1_accuracy, others_unsorted_share: the files of the other groups.
      Their top-1 accuracy, and the share the cutoffs wrongly send to Unsorted.
    - fits_any_auroc: how well fits_any tells the other groups' files (positives) from the
      held-out group's files (negatives). See auroc.
    - frozen_n, rerun_n, rerun_accuracy: the user freezes every file the reduced pass placed
      in its gold group and reruns the rest against the full set. rerun_accuracy is the share
      of the rerun files that `full` places right: in their gold group, or in Unsorted for a
      file with no gold group. A rerun file that `full` has no answer for is left out.

    A file with no gold group is neither held out nor one of the others. It counts in n and
    in the rerun only.
    """
    held = [p for p in reduced if p.gold == HELD_OUT_GROUP_ID]
    others = [p for p in reduced if p.gold not in (None, HELD_OUT_GROUP_ID)]
    frozen = {p.file_id for p in reduced if p.gold is not None and _placed(p, cutoffs) == p.gold}
    full_by_id = {p.file_id: p for p in full}
    rerun = [
        full_by_id[p.file_id]
        for p in reduced
        if p.file_id not in frozen and p.file_id in full_by_id
    ]
    return {
        "n": len(reduced),
        "held_out_n": len(held),
        "held_out_unsorted_share": _share(
            sum(_placed(p, cutoffs) is None for p in held), len(held)
        ),
        "held_out_mean_fits_any": _share(sum(p.fits_any for p in held), len(held)),
        "others_n": len(others),
        "others_top1_accuracy": _share(
            sum(top_group(p.probs) == p.gold for p in others), len(others)
        ),
        "others_unsorted_share": _share(
            sum(_placed(p, cutoffs) is None for p in others), len(others)
        ),
        "fits_any_auroc": auroc([p.fits_any for p in others], [p.fits_any for p in held]),
        "frozen_n": len(frozen),
        "rerun_n": len(rerun),
        "rerun_accuracy": _share(sum(_placed(p, cutoffs) == p.gold for p in rerun), len(rerun)),
    }


def edit_effect(
    base: Sequence[Prediction], edited: Sequence[Prediction], targets: Sequence[str]
) -> dict:
    """Compare a configuration's answers before and after one description edit, file by file.

    A file is right when its top group is its gold group, as in the headline accuracy (the
    cutoffs play no part). Only files answered both times are compared.

    - n, top1_before, top1_after: the files compared, and the share right each time.
    - targets_n, targets_fixed, targets_missing: how many files the edit was meant to fix,
      the ones that were wrong and are now right, and the ones that could not be compared.
    - wrong_to_right, right_to_wrong, wrong_to_other_wrong: every other file whose answer
      changed. The last list holds files that moved from one wrong group to another.

    The four lists hold file ids, in the order of `edited`.
    """
    before = {p.file_id: p for p in base}
    pairs = [(before[p.file_id], p) for p in edited if p.file_id in before]
    compared = {new.file_id for _, new in pairs}
    wanted = set(targets)
    changes: dict[str, list[str]] = {
        "targets_fixed": [],
        "wrong_to_right": [],
        "right_to_wrong": [],
        "wrong_to_other_wrong": [],
    }
    rights_before = 0
    rights_after = 0
    for old, new in pairs:
        was, now = top_group(old.probs), top_group(new.probs)
        was_right, now_right = was == new.gold, now == new.gold
        rights_before += was_right
        rights_after += now_right
        if new.file_id in wanted:
            if now_right and not was_right:
                changes["targets_fixed"].append(new.file_id)
        elif now_right and not was_right:
            changes["wrong_to_right"].append(new.file_id)
        elif was_right and not now_right:
            changes["right_to_wrong"].append(new.file_id)
        elif not was_right and was != now:
            changes["wrong_to_other_wrong"].append(new.file_id)
    return {
        "n": len(pairs),
        "top1_before": _share(rights_before, len(pairs)),
        "top1_after": _share(rights_after, len(pairs)),
        "targets_n": len(targets),
        "targets_missing": [file_id for file_id in targets if file_id not in compared],
        **changes,
    }


def _placed(prediction: Prediction, cutoffs: Cutoffs) -> str | None:
    """Where the placement rules put a file: a group id, or None for Unsorted."""
    return outcome(prediction, cutoffs)[0]


def _share(part: float, whole: int) -> float | None:
    """part / whole, or None when whole is 0."""
    return part / whole if whole else None
