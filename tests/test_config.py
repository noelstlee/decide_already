"""config.py: the settings of one vault, and what Config and Config.load refuse."""

import unicodedata

import pytest

from decide_already import vault
from decide_already.config import Config, clean_dump_dir, folder_key

# A made-up key, shaped like a real one. No error message may ever repeat it.
PASTED_KEY = "ts_live_MADEUP_0123456789abcdefghijklmnopqrstuvwxyz"


@pytest.fixture
def root(tmp_path):
    """A new vault with the default config."""
    vault.init_vault(tmp_path)
    return tmp_path


def write_config(root, text):
    """Replace the vault's config file with this text, as an edit by hand would."""
    Config.path(root).write_text(text, encoding="utf-8")


def change_config(root, old, new):
    """Change one line of the vault's config file by hand."""
    text = Config.path(root).read_text(encoding="utf-8")
    assert old in text
    write_config(root, text.replace(old, new))


# The model pin


@pytest.mark.parametrize("model", ["jev-1.14.0", "jev-1.13.0", "jev-2.0.10", "jev-mini-3.1.4"])
def test_an_exact_model_version_is_accepted(model):
    """A lower-case name followed by a full dotted version is a pin."""
    assert Config(model=model).model == model


@pytest.mark.parametrize(
    "model",
    [
        "jev-latest",
        "jev-preview",
        "JEV-LATEST",
        "Jev-1.14.0",
        " jev-1.14.0",
        "jev-1.14.0 ",
        "jev-1.14.0\n",
        "latest",
        "jev",
        "jev-1",
        "jev-1.14",
        "jev_1.14.0",
        "1.14.0",
        "",
    ],
)
def test_a_model_that_is_not_an_exact_version_is_refused(model, root):
    """A moving alias, a bare name, a partial version and a padded value never get through."""
    with pytest.raises(ValueError, match="an exact model version is needed, such as jev-1.14.0"):
        Config(model=model)
    # The same value in a vault's config file: the config does not load.
    Config().model_copy(update={"model": model}).save(root)
    with pytest.raises(ValueError, match="model: an exact model version is needed") as error:
        Config.load(root)
    assert str(Config.path(root)) in str(error.value)


def test_the_default_model_is_an_exact_version():
    """The pin `p0 init` writes passes its own check."""
    assert Config().model == "jev-1.14.0"


# The dump folder


@pytest.mark.parametrize(
    ("typed", "kept"),
    [
        ("_dump", "_dump"),
        ("course/", "course"),
        ("course///", "course"),
        ("./course", "course"),
        ("././course/", "course"),
        ("  course  ", "course"),
        ("course/files/", "course/files"),
        ("course//files", "course/files"),
        ("./a/./b/", "a/b"),
        ("My files", "My files"),
    ],
)
def test_the_dump_folder_is_kept_in_one_plain_form(typed, kept):
    """Spaces around it, slashes after it and a leading ./ are dropped."""
    assert clean_dump_dir(typed) == kept
    assert Config(dump_dir=typed).dump_dir == kept


@pytest.mark.parametrize(
    "typed",
    [
        "",
        "   ",
        ".",
        "./",
        "/",
        "/Users/someone/files",
        "..",
        "../files",
        "course/../../files",
        "course\\files",
        "_groups",
        "_Groups",
        "Unsorted",
        "unsorted/",
        "UNSORTED/files",
        ".decide",
        ".Decide/sets",
        "./_groups/x",
    ],
)
def test_a_dump_folder_that_is_not_a_plain_folder_inside_the_vault_is_refused(typed):
    """Nothing, the vault itself, a path out of the vault, and the tool's own folders."""
    with pytest.raises(ValueError, match="the dump folder"):
        clean_dump_dir(typed)
    with pytest.raises(ValueError, match="the dump folder"):
        Config(dump_dir=typed)


def test_an_old_config_with_a_trailing_slash_loads_in_the_plain_form(root):
    """A config written before the check ("course/") reads as "course" from now on."""
    change_config(root, "dump_dir: _dump", "dump_dir: course/")
    assert Config.load(root).dump_dir == "course"


def test_a_config_whose_dump_folder_is_the_vault_itself_does_not_load(root):
    """`dump_dir: .` made every note an original. Such a config is refused by name."""
    change_config(root, "dump_dir: _dump", "dump_dir: .")
    with pytest.raises(ValueError, match="dump_dir: the dump folder must be a folder inside"):
        Config.load(root)


# Config.load: errors name the file and never repeat a value


def test_load_names_the_file_when_the_yaml_cannot_be_read(root):
    """A slip in the YAML is a ValueError with the file and the place, not a YAML error."""
    change_config(root, "max_spend_usd: 5.0", "max_spend_usd: : 3")
    with pytest.raises(ValueError) as error:
        Config.load(root)
    message = str(error.value)
    assert message.startswith(f"{Config.path(root)} is not valid YAML: ")
    assert "line 9" in message


def test_load_names_the_setting_and_never_the_value_that_was_typed(root):
    """A key pasted into the config by mistake is not shown again, whichever line it is on."""
    slips = {
        f"api_key: {PASTED_KEY}\n": "api_key: Extra inputs are not permitted",
        f"model: {PASTED_KEY}\n": "model: an exact model version is needed",
        f"backend: {PASTED_KEY}\n": "backend: Input should be 'jev'",
        f"max_spend_usd: {PASTED_KEY}\n": "max_spend_usd: Input should be a valid number",
        f"card_token_budget: {PASTED_KEY}\n": "card_token_budget: Input should be a valid integer",
        f"dump_dir: ../{PASTED_KEY}\n": "dump_dir: the dump folder must be inside the vault",
        f"dump_dir: /{PASTED_KEY}\n": "dump_dir: the dump folder must be given relative",
        f"{PASTED_KEY}\n": "Input should be a valid dictionary",
        f"api_key: {PASTED_KEY}: oops\n": "is not valid YAML",
        f"api_key: '{PASTED_KEY}\n": "is not valid YAML",
        f"api_key: !{PASTED_KEY} x\n": "is not valid YAML",
        f"api_key: *{PASTED_KEY}\n": "is not valid YAML",
    }
    for text, expected in slips.items():
        write_config(root, text)
        with pytest.raises(ValueError) as error:
            Config.load(root)
        message = str(error.value)
        assert expected in message, text
        assert str(Config.path(root)) in message
        assert PASTED_KEY not in message, text
        assert PASTED_KEY[:12] not in message, text
        # Nothing that quotes the value is printed along with the error either.
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__


def test_load_lists_every_setting_that_is_wrong(root):
    """Two slips in one file: both settings are named, in one line."""
    write_config(root, "variant: card\nunsorted_top_below: high\nspend: 3\n")
    with pytest.raises(ValueError) as error:
        Config.load(root)
    assert str(error.value) == (
        f"{Config.path(root)} is not a valid config: "
        "unsorted_top_below: Input should be a valid number, unable to parse string as a number; "
        "spend: Extra inputs are not permitted"
    )


def test_load_names_a_config_that_is_not_text(root):
    """Bytes that are not UTF-8 are reported with the file's name."""
    Config.path(root).write_bytes(b"model: caf\xe9\n")
    with pytest.raises(ValueError, match="config.yaml is not UTF-8 text"):
        Config.load(root)


def test_an_empty_config_file_gives_the_defaults(root):
    """Nothing in the file means every setting at its default."""
    write_config(root, "")
    assert Config.load(root) == Config()


def test_a_config_survives_the_round_trip(root):
    """What save writes, load reads back."""
    config = Config(model="jev-1.13.0", dump_dir="course/files", max_spend_usd=1.5)
    config.save(root)
    assert Config.load(root) == config


# folder_key


def test_folder_key_ignores_capitals_and_the_unicode_form():
    """Names the file system may treat as one folder get one key, on every file system."""
    assert folder_key("HW") == folder_key("hw") == folder_key("Hw")
    assert folder_key("Unsorted") == folder_key("UNSORTED")
    composed = "Übungen"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert folder_key(composed) == folder_key(decomposed)
    assert folder_key("HW") != folder_key("HW 2")
    assert vault.folder_key is folder_key  # the vault module still offers it
