"""Corrections: what the user does to the sorter's picks, and picking up edits made by hand.

Moving a file or freezing it turns its group into a label: one line in .decide/labels.jsonl.
A file's active label is its last line there, when that line is "moved" or "confirmed". A last
line of "unfrozen" means the user withdrew the label. When an action changes a card note, the
note is written first and the label second, so a note that could not be written leaves no
label behind.

`move_file`, `freeze`, `unfreeze` and `suggest` each start with `sync`, so they work on the
vault as it is now, with whatever was dragged or edited in Obsidian since the last command.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from . import vault
from .backends.base import Backend, BackendError, RunStopped
from .card import build_card, hash_file, render_card
from .config import LABELS_FILE, SUGGESTIONS_FILE, UNSORTED_DIR, Config, decide_path
from .log import append_jsonl, read_jsonl
from .sort import decide, judge
from .types import Group, Label, Placement, StrictModel, Suggestion
from .vault import CardNote

# What a freeze or unfreeze reference names, when the caller knows (the page always does).
RefKind = Literal["file", "group"]


class SyncReport(StrictModel):
    """What one sync picked up. Each list holds file ids.

    A file is in at most one of moved, confirmed, unfrozen, orphaned and missing_sources.
    A file in rehomed can also be in confirmed or unfrozen: putting a note back says
    nothing about its frozen flag, which is checked as for any other note.
    """

    moved: list[str]  # the note was dragged to another group's folder, or to Unsorted
    confirmed: list[str]  # frozen was set to true by hand
    unfrozen: list[str]  # frozen was set back to false by hand
    orphaned: list[str]  # sent to Unsorted because the note's group was deleted
    rehomed: list[str]  # put back in its group's folder from a folder that is no group's
    missing_sources: list[str]  # the original is gone from the dump; the note was left alone
    set_version: int  # the current set version, after any edit to the group notes


def read_labels(root: Path) -> list[Label]:
    """Every line of .decide/labels.jsonl, oldest first. A vault without the file has none."""
    return [Label.model_validate(record) for record in read_jsonl(decide_path(root, LABELS_FILE))]


def active_labels(root: Path) -> dict[str, Label]:
    """Each labeled file's active label, by file id.

    That is the file's last line in labels.jsonl, when the line is "moved" or "confirmed".
    A file whose last line is "unfrozen" has no active label.
    """
    last = {label.file_id: label for label in read_labels(root)}
    return {file_id: label for file_id, label in last.items() if label.kind != "unfrozen"}


def _now() -> datetime:
    """The time right now in UTC, to the second, like the times in the Jev log."""
    return datetime.now(UTC).replace(microsecond=0)


def _write_label(
    root: Path, note: CardNote, kind: str, to_group: str | None, set_version: int
) -> None:
    """Add one line to labels.jsonl about a file, taken from its card note as it was.

    The note gives the group the file was in, what the model said about it, and the set
    version the model said it under. A file no model was asked about (no text) has neither.
    `set_version` is the current one: the groups the user had in front of them.
    """
    placement = note.placement
    label = Label(
        file_id=placement.file_id,
        kind=kind,
        from_group=placement.group_id,
        to_group=to_group,
        jev_probs=placement.probs,
        jev_set_version=placement.set_version if placement.probs is not None else None,
        set_version=set_version,
        at=_now(),
    )
    append_jsonl(decide_path(root, LABELS_FILE), label.model_dump(mode="json"))


def _changed(placement: Placement, **changes: object) -> Placement:
    """A copy of a placement with some fields changed, checked like any other record."""
    return Placement.model_validate({**placement.model_dump(), **changes})


def _rewrite(note: CardNote, placement: Placement, body: str) -> None:
    """Rewrite a card note where it is. The keys the user added to its frontmatter stay."""
    vault.write_note(note.path, vault.card_frontmatter(placement, note.extras), body)


def _fresh(root: Path, note: CardNote, original: Path | None) -> tuple[Placement, str]:
    """A note's placement and card, brought up to date with its original as it is now.

    A freeze is about the content the user has in front of them. When the original changed
    after its card was made, the note takes the new hash, file type and card (built with
    the config's variant), so that the freeze covers this content and only a later change
    takes it off. `original` is the file in the dump, or None when it is gone. A file that
    is gone or cannot be read leaves the note's own values in place. Raises ValueError when
    a new card is needed and the config names an unknown variant.
    """
    if original is None:
        return note.placement, note.body
    try:
        if hash_file(original) == note.placement.content_hash:
            return note.placement, note.body
        card = build_card(original, note.placement.file_id, Config.load(root))
    except OSError:
        return note.placement, note.body
    placement = _changed(
        note.placement, content_hash=card.file.content_hash, file_type=card.file.file_type
    )
    return placement, render_card(card)


def _originals(root: Path) -> dict[str, Path]:
    """The path of each file in the dump, by file id."""
    return dict(vault.dump_files(root, Config.load(root)))


def _move(
    root: Path,
    note: CardNote,
    notes: list[CardNote],
    target: str | None,
    set_version: int,
    original: Path | None,
) -> Placement:
    """Carry out a move by the user: put the note in the target group, frozen, and label it.

    The note goes to the group's folder (Unsorted for None) and now says the user placed it.
    What the model said (probs, fits_any, set_version, model) stays in the note as it was.
    The freeze is for the original as it is now (see _fresh).
    """
    placement, body = _fresh(root, note, original)
    placement = _changed(
        placement,
        group_id=target,
        placed_by="you",
        frozen=True,
        close_call=False,
        reason=None,
    )
    vault.write_card_note(root, placement, body, notes)
    _write_label(root, note, "moved", target, set_version)
    return placement


def _freeze_note(
    root: Path, note: CardNote, has_label: bool, set_version: int, original: Path | None
) -> Placement | None:
    """Freeze one card note where it is and confirm its group. None when nothing was left to do.

    The note gets frozen: true, for the original as it is now (see _fresh). The label
    "confirmed" is written unless the file already has an active label, which stands.
    Returns the note's frontmatter as it is afterwards.
    """
    if note.placement.frozen and has_label:
        return None
    placement = note.placement
    if not placement.frozen:
        placement, body = _fresh(root, note, original)
        placement = _changed(placement, frozen=True)
        _rewrite(note, placement, body)
    if not has_label:
        _write_label(root, note, "confirmed", note.placement.group_id, set_version)
    return placement


def unfreeze_note(root: Path, note: CardNote, set_version: int) -> Placement:
    """Take the freeze off one card note: frozen becomes false, and "unfrozen" is written.

    `unfreeze` does this for each file it is asked about. The sorter does it for a frozen
    file whose content changed, since the freeze was for the old content.
    """
    placement = _changed(note.placement, frozen=False)
    _rewrite(note, placement, note.body)
    _write_label(root, note, "unfrozen", note.placement.group_id, set_version)
    return placement


def _notes_by_file(notes: list[CardNote]) -> dict[str, CardNote]:
    """Card notes by file id, in order of file id.

    A file can have several notes when one was copied by hand. The last one counts, as in
    sort_vault, and the others are passed over until the next write of that file removes them.
    """
    return {note.placement.file_id: note for note in notes}


def _group_refs(groups: list[Group]) -> dict[str, str | None]:
    """Every way to name a group, mapped to its id: its name, its id, and Unsorted for None."""
    refs: dict[str, str | None] = {group.name: group.id for group in groups}
    refs.update({group.id: group.id for group in groups})  # an id wins over another group's name
    refs[UNSORTED_DIR] = None
    return refs


def _find_note(notes: dict[str, CardNote], ref: str, what: str = "sorted file") -> CardNote:
    """The card note a file reference names: by file id, or else by a name only one file has.

    The name is the last part of the file id. ValueError when nothing matches, or when
    several files share the name; the message then lists their ids.
    """
    if ref in notes:
        return notes[ref]
    named = [file_id for file_id in notes if file_id.rsplit("/", 1)[-1] == ref]
    if len(named) == 1:
        return notes[named[0]]
    if named:
        raise ValueError(
            f"{len(named)} files are named {ref!r}: {', '.join(named)}; use the full id of one"
        )
    raise ValueError(
        f"no {what} with the id or name {ref!r} (a new file gets its card note from `p0 sort`)"
    )


def _selected(root: Path, ref: str, kind: RefKind | None = None) -> list[CardNote]:
    """The card notes a freeze or unfreeze is about: every note of a group, or one file's note.

    `kind` says what the reference names. "file": a file id, matched exactly and never read
    as a group. "group": a group id or name, or Unsorted, and never read as a file. None is
    the command line's reading: a group, or Unsorted, wins over a file of the same name, and
    a file may be given by a name that only one file has.
    """
    notes = _notes_by_file(vault.read_card_notes(root))
    groups = _group_refs(vault.read_groups(root))
    if kind == "file":
        if ref not in notes:
            raise ValueError(f"no sorted file with the id {ref!r}")
        return [notes[ref]]
    if ref in groups:
        return [note for note in notes.values() if note.placement.group_id == groups[ref]]
    if kind == "group":
        raise ValueError(f"no group with the id or name {ref!r}")
    return [_find_note(notes, ref, "group or sorted file")]


def _label_without_note(root: Path, ref: str, kind: RefKind | None) -> Label | None:
    """The active label of a file that has no card note, when `ref` is exactly that file's id.

    None in every other case. Such a file's id wins over another file that only shares its
    name. As in `_selected`, a group wins over a file of the same name on the command line.
    """
    label = None if kind == "group" else active_labels(root).get(ref)
    if label is None:
        return None
    if kind is None and ref in _group_refs(vault.read_groups(root)):
        return None
    if any(note.placement.file_id == ref for note in vault.read_card_notes(root)):
        return None
    return label


def sync(root: Path) -> SyncReport:
    """Pick up what was changed by hand in the vault, and record it.

    Group notes come first: an edit there becomes a new set version. Then every card note
    whose original is still in the dump is checked against four rules, in this order, and
    the first one that applies is carried out:

    1. The note sits in a folder that is neither a current group's nor Unsorted. A folder
       says nothing when it is no group's, so the note's own group decides. When that group
       is gone (it was deleted), the note moves to Unsorted and is no longer frozen, and an
       active label is withdrawn with "unfrozen". Otherwise the note is put back in its
       group's folder as it is (rehomed): no label, and the freeze stays. That is what
       happens after a group note is renamed by hand, and to a note dragged into a folder
       that is no group. Rules 3 and 4 then apply to a rehomed note as to any other.
    2. The note sits in another group's folder than the group in its frontmatter (it was
       dragged): that is a move by the user. The note is frozen there and "moved" is written.
    3. The note says frozen and the file has no active label (frozen was set by hand):
       "confirmed" is written.
    4. The note says not frozen and the file has an active label (frozen was cleared by
       hand): "unfrozen" is written.

    A folder is matched to its group by key, so HW's folder is HW's under any spelling the
    file system treats as the same. A note that becomes frozen (rules 2 and 3) takes the
    hash and card of its original as it is now, so the freeze covers that content.

    A card note whose original is gone from the dump is only listed, under missing_sources:
    no rule is applied to it and nothing about it changes. Every rule leaves its note
    settled, so a second sync right after the first has nothing more to record. Last, the
    folders that were set aside for a renamed or deleted group are removed once they are
    empty. Raises ValueError for a broken group note, and FileExistsError when a file of the
    user's sits where a card note has to go.
    """
    group_set = vault.sync_group_set(root)
    folders = vault.folder_ids(group_set.groups)
    current = {None, *(group.id for group in group_set.groups)}  # None: Unsorted is always there
    originals = _originals(root)
    active = active_labels(root)
    notes = vault.read_card_notes(root)
    report = SyncReport(
        moved=[],
        confirmed=[],
        unfrozen=[],
        orphaned=[],
        rehomed=[],
        missing_sources=[],
        set_version=group_set.version,
    )
    for file_id, note in _notes_by_file(notes).items():
        placement = note.placement
        folder = vault.folder_key(note.folder)
        if file_id not in originals:
            report.missing_sources.append(file_id)
            continue
        if folder not in folders:
            if placement.group_id not in current:
                orphan = _changed(
                    placement, group_id=None, placed_by="code", frozen=False, close_call=False
                )
                vault.write_card_note(root, orphan, note.body, notes)
                if file_id in active:
                    _write_label(root, note, "unfrozen", None, group_set.version)
                report.orphaned.append(file_id)
                continue
            path = vault.write_card_note(root, placement, note.body, notes)
            note = note.model_copy(update={"path": path})
            report.rehomed.append(file_id)
        elif folders[folder] != placement.group_id:
            _move(root, note, notes, folders[folder], group_set.version, originals[file_id])
            report.moved.append(file_id)
            continue
        if placement.frozen and file_id not in active:
            fresh, body = _fresh(root, note, originals[file_id])
            if fresh != placement:
                _rewrite(note, fresh, body)
            _write_label(root, note, "confirmed", placement.group_id, group_set.version)
            report.confirmed.append(file_id)
        elif not placement.frozen and file_id in active:
            _write_label(root, note, "unfrozen", placement.group_id, group_set.version)
            report.unfrozen.append(file_id)
    vault.remove_empty_set_asides(root)
    return report


def move_file(
    root: Path, file_ref: str, group_ref: str | None, *, exact: bool = False
) -> Placement:
    """Put a file in a group yourself, and freeze it there. Returns the note's new frontmatter.

    `file_ref` is a file id, or a file name that only one sorted file has. `group_ref` is a
    group id or name; None or "Unsorted" means Unsorted, for a file that fits no group.
    With `exact` (the page uses it), the file is matched by its id only and the group by
    its id only, so a file can never be mistaken for another one of the same name.
    The note moves to the group's folder, says the user placed it, and "moved" is written.
    A file that is already in that group is not moved: it is frozen where it is, and
    "confirmed" is written unless it already has an active label. Raises ValueError for a
    reference that names nothing, or a file name that several files share.
    """
    set_version = sync(root).set_version
    notes = vault.read_card_notes(root)
    by_file = _notes_by_file(notes)
    if exact and file_ref not in by_file:
        raise ValueError(f"no sorted file with the id {file_ref!r}")
    note = by_file[file_ref] if exact else _find_note(by_file, file_ref)
    groups = vault.read_groups(root)
    refs = {group.id: group.id for group in groups} if exact else _group_refs(groups)
    if group_ref is not None and group_ref not in refs:
        names = ", ".join([*(group.name for group in groups), UNSORTED_DIR])
        raise ValueError(f"no group with the id or name {group_ref!r}; pick one of: {names}")
    target = None if group_ref is None else refs[group_ref]
    original = _originals(root).get(note.placement.file_id)
    if target != note.placement.group_id:
        return _move(root, note, notes, target, set_version, original)
    has_label = note.placement.file_id in active_labels(root)
    return _freeze_note(root, note, has_label, set_version, original) or note.placement


def freeze(root: Path, ref: str, *, kind: RefKind | None = None) -> list[str]:
    """Freeze a file, or every file in a group, so that reruns leave it where it is.

    `ref` is a group id or name (or "Unsorted"), or else a file id or unique file name.
    `kind` settles which, for a caller that knows (see `_selected`).
    Each note gets frozen: true, and "confirmed" is written for it with its current group.
    The freeze is for the original as it is now: a file that changed since its card was made
    gets its new hash and card written into the note. A file that is already frozen and has
    an active label is left as it is. Returns the ids of the files it froze. Raises
    ValueError for a reference that names nothing.
    """
    set_version = sync(root).set_version
    active = active_labels(root)
    originals = _originals(root)
    frozen = []
    for note in _selected(root, ref, kind):
        file_id = note.placement.file_id
        placement = _freeze_note(root, note, file_id in active, set_version, originals.get(file_id))
        if placement is not None:
            frozen.append(file_id)
    return frozen


def unfreeze(root: Path, ref: str, *, kind: RefKind | None = None) -> list[str]:
    """Unfreeze a file, or every file in a group, so that the next rerun sorts it again.

    `ref` is read as in `freeze`. Each frozen note gets frozen: false, and "unfrozen" is
    written for it. A file that is not frozen is left as it is. Returns the ids of the files
    it unfroze. Raises ValueError for a reference that names nothing.

    One more case: a file id that has an active label but no card note (the note was
    deleted, or can no longer be read). The sorter will not send such a file to the model.
    Unfreezing it withdraws the label ("unfrozen", from and to the label's group, with no
    probabilities), and the next sort starts the file from scratch.
    """
    set_version = sync(root).set_version
    label = _label_without_note(root, ref, kind)
    if label is not None:
        withdrawn = Label(
            file_id=ref,
            kind="unfrozen",
            from_group=label.to_group,
            to_group=label.to_group,
            jev_probs=None,
            jev_set_version=None,
            set_version=set_version,
            at=_now(),
        )
        append_jsonl(decide_path(root, LABELS_FILE), withdrawn.model_dump(mode="json"))
        return [ref]
    unfrozen = []
    for note in _selected(root, ref, kind):
        if note.placement.frozen:
            unfreeze_note(root, note, set_version)
            unfrozen.append(note.placement.file_id)
    return unfrozen


def suggest(root: Path, backend: Backend, failed: dict[str, str] | None = None) -> list[Suggestion]:
    """Ask the model about every frozen file, and list the ones it would now place elsewhere.

    The model's answers move nothing and write no label. Each frozen file is carded with the
    config's variant, judged against the current groups, and run through the placement
    rules. When the outcome differs from the file's group, one line goes to
    .decide/suggestions.jsonl right away, so a later failure cannot lose it. A file the
    backend fails on, a file with no text and a file whose original is gone are passed over.
    A failure that is about the run and not about one file (the spend cap, a call log that
    cannot be written) ends the run, and what was found until then is returned. With no
    groups there is nothing to suggest.

    A failure is not an answer, so it is not agreement either: when `failed` is a dict, each
    file the model could not be asked about is added to it with the reason (for a run that
    was stopped, the file it stopped at). Raises ValueError when the config names an unknown
    variant.
    """
    sync(root)
    config = Config.load(root)
    group_set = vault.sync_group_set(root)
    if not group_set.groups:
        return []
    paths = dict(vault.dump_files(root, config))
    found = []
    for file_id, note in _notes_by_file(vault.read_card_notes(root)).items():
        if not note.placement.frozen or file_id not in paths:
            continue
        try:
            card = build_card(paths[file_id], file_id, config)
            if not card.states:
                continue
            result = judge(card, group_set.groups, group_set.version, backend, config.variant)
        except RunStopped as error:
            if failed is not None:
                failed[file_id] = str(error)
            break
        except (BackendError, OSError) as error:
            if failed is not None:
                failed[file_id] = str(error)
            continue
        decision = decide(result.probs, result.fits_any, config)
        if decision.group_id != note.placement.group_id:
            suggestion = Suggestion(
                file_id=file_id,
                current_group=note.placement.group_id,
                suggested_group=decision.group_id,
                probs=result.probs,
                fits_any=result.fits_any,
                set_version=group_set.version,
                model=result.model,
                at=_now(),
            )
            append_jsonl(decide_path(root, SUGGESTIONS_FILE), suggestion.model_dump(mode="json"))
            found.append(suggestion)
    return found
