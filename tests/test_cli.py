from typer.testing import CliRunner

from decide_already.cli import app

runner = CliRunner()


def test_init_then_group_commands(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "dump: _dump" in result.output

    vault = ["--vault", str(tmp_path)]
    result = runner.invoke(app, ["group", "add", "HW", "--covers", "homework", *vault])
    assert result.exit_code == 0, result.output
    assert "Set version 1" in result.output

    result = runner.invoke(
        app, ["group", "add", "Logistics", "--covers", "syllabus", "--excludes", "projects", *vault]
    )
    assert "Set version 2" in result.output

    result = runner.invoke(
        app, ["group", "edit", "Logistics", "--covers", "syllabus and deadlines", *vault]
    )
    assert "Set version 3" in result.output

    result = runner.invoke(app, ["group", "rename", "HW", "Homework", *vault])
    assert "Set version 4" in result.output

    result = runner.invoke(app, ["group", "list", *vault])
    assert result.exit_code == 0, result.output
    assert "Set version 4" in result.output
    assert "Homework (hw): covers homework" in result.output
    assert "excludes projects" in result.output

    result = runner.invoke(app, ["group", "delete", "Homework", *vault])
    assert "Set version 5" in result.output


def test_errors_exit_nonzero(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    vault = ["--vault", str(tmp_path)]
    result = runner.invoke(app, ["group", "add", "HW", "--covers", "  ", *vault])
    assert result.exit_code == 1
    assert "covers description" in result.output
    result = runner.invoke(app, ["group", "edit", "Nope", "--covers", "x", *vault])
    assert result.exit_code == 1
    result = runner.invoke(app, ["group", "edit", "HW", *vault])
    assert result.exit_code == 1


def test_init_with_missing_dump_fails(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path), "--dump", "missing"])
    assert result.exit_code == 1
    assert "does not exist" in result.output
