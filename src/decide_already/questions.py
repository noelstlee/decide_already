"""The two questions asked about every file, in TypeSafe's wire format.

All question wording lives here and nowhere else. Each request carries a Choice over the groups
and a yes/no (a Noul) on whether any group fits. A Choice always picks an option, even when
nothing fits, so there is no "none of these" option: code decides Unsorted from the two answers
(see sort.decide).
"""

import random

from .types import Group

GROUP_QUESTION = "group"
FITS_ANY_QUESTION = "fits_any"

GROUP_INSTRUCTIONS = "Which group does this document belong in?"
FITS_ANY_STATEMENT = "This document clearly belongs in one of these groups."
FITS_ANY_TRUE = "The document's content matches what one of the groups covers."
FITS_ANY_FALSE = "The document is about something that none of the groups cover."


def _description(group: Group) -> dict[str, str]:
    """What the model reads about one group: what it covers and, when set, what it excludes."""
    description = {"covers": group.covers}
    if group.excludes:
        description["excludes"] = group.excludes
    return description


def build_questions(
    groups: list[Group], rng: random.Random, with_descriptions: bool = True
) -> tuple[dict, list[str]]:
    """Build the questions for one request. Return them with the option order as group ids.

    The options are shuffled with `rng` on every call, because the model leans slightly toward
    the first option. Group names are the option names the model sees; the question ids are not
    sent to it. The model answers each question on its own, so fits_any lists the groups itself,
    in the same order. With `with_descriptions` false, only the group names are sent.
    """
    if not groups:
        raise ValueError("there are no groups to choose from")
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("two groups share a name; every option name must be unique")
    ordered = list(groups)
    rng.shuffle(ordered)
    if with_descriptions:
        criteria = {group.name: _description(group) for group in ordered}
        listed = {group.name: _description(group) for group in ordered}
    else:
        criteria = {group.name: None for group in ordered}
        listed = [group.name for group in ordered]
    questions = {
        GROUP_QUESTION: {
            "type": "choice",
            "instructions": GROUP_INSTRUCTIONS,
            "criteria": criteria,
        },
        FITS_ANY_QUESTION: {
            "type": "noul",
            "instructions": {"statement": FITS_ANY_STATEMENT, "groups": listed},
            "criteria": {"true": FITS_ANY_TRUE, "false": FITS_ANY_FALSE},
        },
    }
    return questions, [group.id for group in ordered]
