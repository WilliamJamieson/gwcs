from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import nox

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet

# Make nox default to uv if its available, if not fallback on virtualenv
nox.options.default_venv_backend = "uv|virtualenv"


@dataclass(frozen=True, slots=True, kw_only=True)
class PythonSupport:
    """Information about supported Python versions."""

    PYTHON_RELEASE_URL: ClassVar[str] = (
        "https://www.python.org/api/v2/downloads/release/"
    )
    PYTHON_RELEASE_PATTERN: ClassVar[re.Pattern] = re.compile(r"Python 3\.(\d+)\.(\d+)")
    PYTHON_CANDIDATE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"Python 3\.(\d+)\.0rc(\d+)"
    )

    versions: tuple[str, ...]
    """The currently supported Python versions. e.g. 3.11, 3.12, ..."""
    release_candidate: str | None = None
    """The latest release candidate python version. e.g. 3.15rc2"""

    @property
    def oldest(self) -> str:
        """Return the oldest supported Python version."""
        return self.versions[0]

    @property
    def latest(self) -> str:
        """Return the latest supported Python version."""
        return self.versions[-1]

    @property
    def default(self) -> str:
        """
        Return the default supported Python version.

        This is the second latest supported Python version because the
        latest one may not be supported completely when a new Python version is
        released.

        This is intended to be the default version of python used for CI testing
        """
        return self.versions[-2]

    @classmethod
    def from_pyproject(cls, pyproject_path: Path) -> PythonSupport:
        """
        Read the supported Python versions from a pyproject.toml file.

        Parameters
        ----------
        pyproject_path :
            The path to the pyproject.toml file.

        Returns
        -------
            An instance of PythonSupport populated with the supported Python versions
            specified in the pyproject.toml file.
        """
        return cls(
            versions=tuple(
                sorted(
                    nox.project.python_versions(nox.project.load_toml(pyproject_path))
                )
            )
        )

    @classmethod
    def from_python(cls, requires_python: SpecifierSet | None = None) -> PythonSupport:
        """
        Read from python.org the available and supported Python versions.

        Parameters
        ----------
        requires_python :
            The Python version specifier to filter supported versions.
            This is typically what one would specify in the `pyproject.toml` file
            under the `requires-python` field.

        Returns
        -------
            An instance of PythonSupport populated with the available and supported
            Python versions from python.org.
        """
        import requests

        # Fetch the list of Python releases from python.org
        response = requests.get(cls.PYTHON_RELEASE_URL, timeout=15)
        response.raise_for_status()

        # Look through the listed releases from the python.org API for what
        # we are interested in.
        stable_versions: set[tuple[int, int]] = set()
        release_candidates: dict[tuple[int, int], int] = {}
        for release in response.json():
            # Only consider python 3 releases that are published
            if release.get("version") != 3 or not release.get("is_published"):
                continue

            # Match against the stable and release candidate name patterns
            name = release.get("name", "")
            stable_match = cls.PYTHON_RELEASE_PATTERN.fullmatch(name)
            candidate_match = cls.PYTHON_CANDIDATE_PATTERN.fullmatch(name)

            # If no match for either stable or release candidate, skip this release
            if stable_match is None and candidate_match is None:
                continue

            # Skip releases that do not match the required Python version specifier.
            #   Only do this if a requires_python specifier is provided.
            if (
                requires_python is not None
                and name.removeprefix("Python ") not in requires_python
            ):
                continue

            # Add the match if it matches the stable pattern and is not marked
            #   by python as a pre-release.
            if stable_match is not None and not release.get("pre_release"):
                minor = int(stable_match.group(1))
                stable_versions.add((3, minor))
                continue

            # Add the match if it matches the release candidate pattern and is marked
            #   by python as a pre-release.
            if candidate_match is not None and release.get("pre_release"):
                minor, candidate = map(int, candidate_match.groups())
                version = (3, minor)
                release_candidates[version] = max(
                    candidate, release_candidates.get(version, 0)
                )

        if not stable_versions:
            msg = "No published matching stable Python 3 releases found on python.org"
            raise ValueError(msg)

        # Python's annual minor releases are supported across the latest five series.
        # See https://peps.python.org/pep-0602/#2-years-of-full-support-3-more-years-of-security-fixes
        # for the five-year lifecycle. Current statuses: https://devguide.python.org/versions/
        versions = tuple(
            f"{major}.{minor}" for major, minor in sorted(stable_versions)[-5:]
        )

        # Find the release candidate that comes after the latest stable release
        latest_stable = max(stable_versions)
        release_candidate = max(
            (
                (major, minor, candidate)
                for (major, minor), candidate in release_candidates.items()
                if (major, minor) > latest_stable
            ),
            default=None,
        )
        return cls(
            versions=versions,
            release_candidate=(
                f"{release_candidate[0]}.{release_candidate[1]}.0rc{release_candidate[2]}"
                if release_candidate
                else None
            ),
        )


PYTHON_SUPPORT = PythonSupport.from_pyproject(Path("pyproject.toml"))


@dataclass(frozen=True, slots=True, kw_only=True)
class PythonProject:
    """Represents the Python project's classifiers and support information."""

    PYTHON_CLASSIFIER_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"Programming Language :: Python :: (\d+\.\d+)"
    )

    general_classifiers: tuple[str, ...]
    """The non-python release related classifiers."""

    python_classifiers: MappingProxyType[str, str]
    """
    The Python release related classifiers.

    Maps Python version string to classifier e.g.
        Programming Language :: Python :: 3.<minor>
    """

    python_support: PythonSupport
    """The Python support information for this project"""

    @classmethod
    def from_pyproject(
        cls, pyproject: dict[str, Any], use_python_org: bool = True
    ) -> PythonProject:
        """
        Creates a ProjectClassifiers instance from a pyproject.toml dictionary.

        Parameters
        ----------
        pyproject :
            The parsed pyproject.toml dictionary.

        use_python_org :
            Whether to query the python.org release page for the python versions
            or just use the classifiers specified in the pyproject.toml.

        Returns
        -------
            An instance of ProjectClassifiers populated with data from the
            pyproject.toml.
        """
        from packaging.specifiers import SpecifierSet

        project = pyproject["project"]
        general_classifiers = []
        python_classifiers = {}
        for classifier in project.get("classifiers", []):
            if (match := cls.PYTHON_CLASSIFIER_PATTERN.fullmatch(classifier)) is None:
                general_classifiers.append(classifier)
            else:
                python_classifiers[match.group(1)] = classifier

        python_support = (
            PythonSupport.from_python(SpecifierSet(project["requires-python"]))
            if use_python_org
            else PythonSupport(versions=tuple(sorted(python_classifiers.keys())))
        )

        return cls(
            general_classifiers=tuple(general_classifiers),
            python_classifiers=MappingProxyType(python_classifiers),
            python_support=python_support,
        )

    @property
    def has_classifiers(self) -> bool:
        """
        Checks if the project has all the necessary Python classifiers given its
        python support.

        Returns
        -------
        bool
            True if all required Python classifiers are present, False otherwise.
        """
        return set(self.python_support.versions) == set(self.python_classifiers.keys())

    @property
    def classifiers(self) -> tuple[str, ...]:
        """
        Returns the complete list of classifiers given the project's Python support.

        Returns
        -------
        tuple[str, ...]
            A tuple containing all the relevant Python classifiers.
        """
        classifiers: list[str] = list(self.general_classifiers)
        classifiers.extend(
            f"Programming Language :: Python :: {version}"
            for version in self.python_support.versions
        )

        return tuple(classifiers)

    def update_classifiers(self, pyproject_file: Path) -> None:
        """
        Updates the Python classifiers in the given pyproject.toml file based on the
        project's current Python support.

        Parameters
        ----------
        pyproject_file : Path
            The path to the pyproject.toml file to update.
        """
        import tomlkit

        content = pyproject_file.read_text(encoding="utf-8")
        document = tomlkit.parse(content)
        project = document["project"]
        target = list(self.classifiers)

        if "classifiers" not in project:
            project["classifiers"] = tomlkit.array().multiline(True)
        classifiers = project["classifiers"]

        # Edit the array in place so tomlkit keeps its existing formatting.
        # Note: entries that get deleted and re-inserted lose their trailing comments,
        # and comment lines stay at their original position rather than following the
        # entry below them.
        for index, classifier in enumerate(target):
            # Already in the right position, nothing to do
            if index < len(classifiers) and classifiers[index] == classifier:
                continue

            # Present later in the array, so remove it before moving it here
            if classifier in classifiers[index:]:
                del classifiers[list(classifiers).index(classifier, index)]

            # Place the classifier at its target position
            classifiers.insert(index, classifier)

        # Leftovers past the target length are classifiers no longer wanted
        while len(classifiers) > len(target):
            del classifiers[-1]

        updated = tomlkit.dumps(document)
        if updated != content:
            pyproject_file.write_text(updated, encoding="utf-8")


@nox.session(venv_backend="none")
def check_python_classifiers(session: nox.Session) -> None:
    """
    Check the python classifiers in the pyproject.toml file are consistent.

    This checks against the officially Python release web API. It ensures that
    the classifiers listed in the pyproject.toml file are consistent with the
    listed `python-requires` specification and the versions of python that are
    actually available.
    """
    pyproject = Path("pyproject.toml")
    project = PythonProject.from_pyproject(nox.project.load_toml(pyproject))

    if not project.has_classifiers:
        project.update_classifiers(pyproject)
        session.error(
            "Python classifiers were missing and have been updated. "
            "Please review and commit the changes!"
        )

    session.log(f"Python classifiers are consistent with: {project.python_support}")


def _list_dependencies(session: nox.Session) -> None:
    """List the packages installed in a session's environment."""
    if session.venv_backend == "uv":
        session.run(
            "uv",
            "pip",
            "list",
            "--python",
            session.virtualenv.location,
            external=True,
        )
    else:
        session.run("python", "-m", "pip", "list")


def _add_standard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard command-line arguments to the parser."""
    parser.add_argument(
        "--xdist",
        nargs="?",
        const=True,
        default=False,
        type=int,
        help="Enable pytest-xdist for parallel test execution",
    )

    # Ensure that only one of --editable or --wheel can be specified
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--editable",
        action="store_true",
        help="Install gwcs in editable mode",
    )
    install_group.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Install gwcs from a built wheel file instead of the source tree",
    )


def _install_gwcs(
    session: nox.Session,
    args: argparse.Namespace,
    install_args: list[str] | None = None,
) -> None:
    """Install gwcs in the virtual environment based on the command-line arguments."""
    install_args = install_args or []

    # Setup the install arguments for gwcs itself
    if args.wheel is not None:
        install_args += [f"{args.wheel}[test]"]
    elif args.editable:
        install_args += ["-e", ".[test]"]
    else:
        install_args += [".[test]"]

    session.log("Installing the gwcs package for testing:")

    # Add pytest-xdist if requested
    if args.xdist:
        session.log("  Including pytest-xdist for parallel test execution")
        install_args += ["pytest-xdist"]

    session.install(*install_args)


def _init_pytest_arguments(session: nox.Session, args: argparse.Namespace) -> list[str]:
    """Initialize the pytest arguments based on the command-line options."""
    session.log("Running tests:")

    arguments: list[str] = []
    if args.xdist:
        session.log("  Including pytest-xdist for parallel test execution")
        arguments += ["-n", "auto" if args.xdist is True else str(args.xdist)]

    return arguments


@nox.session(python=None)
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    if session.posargs and session.posargs[0] in PYTHON_SUPPORT.versions:
        target_python = session.posargs.pop(0)
    else:
        target_python = PYTHON_SUPPORT.default

    session.log(f"Target Python interpreter: {target_python}")
    session.notify(f"run_tests-{target_python}", session.posargs)


@nox.session(python=PYTHON_SUPPORT.versions, reuse_venv=False)
def run_tests(session: nox.Session) -> None:
    """Run the tests for python versions"""
    parser = argparse.ArgumentParser(
        prog="nox -s test --",
        allow_abbrev=False,
        description="Run the gwcs test suite.",
    )
    _add_standard_arguments(parser)

    # Build the parser, adding the options as we go
    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
    )

    # Ensure that only one of --dev or --oldest can be specified
    dependency_group = parser.add_mutually_exclusive_group()
    dependency_group.add_argument(
        "--develop", action="store_true", help="Install development dependencies"
    )
    dependency_group.add_argument(
        "--oldest",
        action="store_true",
        help="Install the oldest compatible dependencies",
    )
    args, pytest_args = parser.parse_known_args(session.posargs)
    install_args: list[str] = []

    # Prepare the install arguments for dev/oldest if required
    if args.develop:
        install_args += ["-r", "requirements-dev.txt"]
    if args.oldest:
        if session.venv_backend != "uv":
            session.error("--oldest requires the uv backend")

        install_args += ["--resolution", "lowest-direct"]

    _install_gwcs(session, args, install_args)

    # Install coverage if requested
    if args.coverage:
        session.log("Installing coverage dependencies:")
        session.install("pytest-cov")

    _list_dependencies(session)

    # Configure the pytest arguments
    arguments = _init_pytest_arguments(session, args)
    if args.coverage:
        session.log("  Enabling coverage reporting")
        arguments += [
            "--cov=.",
            "--cov-config=pyproject.toml",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ]
    arguments += pytest_args

    session.run("pytest", *arguments)


@dataclass(frozen=True, slots=True, kw_only=True)
class MatrixEntry:
    """Represents an entry in a github workflow job matrix."""

    DEFAULT_RUNS_ON: ClassVar[str] = "ubuntu-latest"
    MACOS_RUNS_ON: ClassVar[str] = "macos-latest"

    execute_session: str
    """The name of the session that will execute this matrix entry."""
    python: str = PYTHON_SUPPORT.default
    """The Python version for this matrix entry."""
    options: tuple[str | tuple[str, str], ...] = field(default_factory=tuple)
    """
    The options for this matrix entry

    Each option can be either a string or a tuple of two strings.
    -> A single string will be used as `--option` in the nox command.
    -> A tuple of two strings will be used as `--option=value` in the nox command.
    """
    runs_on: str = field(default=DEFAULT_RUNS_ON)
    """The runner environment for this matrix entry."""
    tags: tuple[str, ...] = field(default_factory=tuple)
    """
    The tags for this matrix entry.

    These will be the GitHub labels that need to be applied for this entry to run.
        -> Empty tuple means no tags are required for this entry.
        -> Downstream-CI, in this means entries that should run in the downstream CI.
    """

    @property
    def args(self) -> tuple[str, ...]:
        """The positional arguments for this matrix entry."""
        return ()

    @property
    def python_id(self) -> str:
        """The Python identifier for this matrix entry."""
        return f"py{self.python}"

    @property
    def args_id(self) -> str:
        """The arguments identifier for this matrix entry."""
        return "-".join(self.args)

    @property
    def option_id(self) -> str:
        """The option identifier for this matrix entry."""
        return "-".join(
            f"{option[0]}-{option[1]}" if isinstance(option, tuple) else option
            for option in self.options
        )

    @property
    def runs_on_id(self) -> str:
        """The runner environment identifier for this matrix entry."""
        return self.runs_on

    @property
    def nox_id(self) -> str:
        """
        The identifier for this matrix entry.

        This is the string that shows as an argument to the origin_session when invoking
        nox.
        -> `nox -s origin_session(nox_id)` will invoke the represented session
        """
        nox_id = f"{self.python_id}"

        # Append the the arguments, options, and runner environment identifiers
        #    to the nox_id.
        if self.args:
            nox_id += f"--{self.args_id}"
        if self.options:
            nox_id += f"--{self.option_id}"
        if self.runs_on != self.DEFAULT_RUNS_ON:
            nox_id += f"--{self.runs_on_id}"

        return nox_id

    @property
    def nox_param(self) -> nox.param:
        """The nox parameter for this matrix entry."""
        return nox.param(self, id=self.nox_id, tags=self.tags if self.tags else None)

    @staticmethod
    def check_session(session: str, python: str) -> str:
        """
        Check if the given session and python version are valid

        Parameters
        ----------
        session :
            The name of the nox session to check.
        python :
            The python version to check, by default None.

        Raises
        ------
        ValueError
            If the session or python version is not valid.

        Returns
        -------
            The session name if the session and python version are valid.
        """
        # Check that the execute_session has been registered as a nox session.
        func = nox.registry.get().get(session)
        if func is None:
            msg = f"{session!r} is not a valid nox session!"
            raise ValueError(msg)

        # nox drops the python suffix from session names when there is no venv
        #   it will fail if we include the python suffix in that case.
        if func.venv_backend == "none":
            return session

        # Check that the specified python version is specified and supported by
        #   the execute_session.
        pythons = [func.python] if isinstance(func.python, str) else func.python
        if not isinstance(pythons, list | tuple) or python not in pythons:
            msg = (
                f"{session!r} is not a valid nox session: {session!r} "
                f"supports python {func.python!r}"
            )
            raise ValueError(msg)

        return f"{session}-{python}"

    def nox_session(self, session: str) -> str:
        """
        The session name for this matrix entry.

        GitHub Actions will run nox with this session name.
        -> `nox -s nox_session`
        """
        return f"{self.check_session(session, self.python)}({self.nox_id})"

    @property
    def nox_name(self) -> str:
        """The name for the GitHub Actions job that will run this entry"""
        if self.runs_on == self.DEFAULT_RUNS_ON:
            return f"{self.nox_id}"

        return f"{self.nox_id} ({self.runs_on})"

    @property
    def option_args(self) -> tuple[str, ...]:
        """Return the option arguments for this matrix entry."""
        return tuple(
            f"--{option[0]}={option[1]}" if isinstance(option, tuple) else f"--{option}"
            for option in self.options
        )

    @property
    def posargs(self) -> tuple[str, ...]:
        """Return the positional arguments for this matrix entry."""
        return (*self.args, *self.option_args)

    @property
    def session(self) -> str:
        """Return the name of the session to execute for this matrix entry."""
        return self.check_session(self.execute_session, self.python)

    def run_session(
        self, session: nox.session, posargs: list[str] | None = None
    ) -> None:
        """
        Run the nox session for this matrix entry.

        Parameters
        ----------
        session :
            The nox session to run.
        posargs :
            The positional arguments to pass to the session, by default None.
        """
        if posargs := [*self.posargs, *(posargs or [])]:
            posargs = ["--", *posargs]

        session.run(
            "nox",
            "-s",
            self.session,
            *posargs,
            external=True,
        )

    def github_matrix_entry(
        self, session: str, labels: list[str], force_run: bool = False
    ) -> dict[str, str] | None:
        """Return the github matrix entry for this matrix entry."""
        # Only include this matrix entry if it has no tags or if at least one of
        #     the labels matches a tag.
        if force_run or not self.tags or any(label in self.tags for label in labels):
            # Ensure that the session that this matrix entry will actually run is
            #   valid
            _ = self.session

            return {
                "name": self.nox_name,
                # Note that nox_session checks that the session represented by
                #   this matrix entry is valid.
                "session": self.nox_session(session),
                "python": self.python,
                "runs-on": self.runs_on,
            }

        return None


CI_MATRIX = (
    MatrixEntry(
        execute_session="run_tests",
        python=PYTHON_SUPPORT.oldest,
        options=("oldest",),
    ),
    MatrixEntry(
        execute_session="run_tests",
        python=PYTHON_SUPPORT.latest,
        options=("develop",),
    ),
    MatrixEntry(
        execute_session="run_tests",
        python=PYTHON_SUPPORT.default,
        options=("coverage",),
    ),
    MatrixEntry(
        execute_session="run_tests",
        python=PYTHON_SUPPORT.default,
        options=("editable",),
    ),
    MatrixEntry(
        execute_session="run_tests",
        python=PYTHON_SUPPORT.default,
        runs_on=MatrixEntry.MACOS_RUNS_ON,
    ),
    *[
        MatrixEntry(
            execute_session="run_tests",
            python=python,
        )
        for python in PYTHON_SUPPORT.versions
    ],
)


@nox.session(venv_backend="none")
@nox.parametrize("matrix_entry", [entry.nox_param for entry in CI_MATRIX])
def run_ci(session: nox.session, matrix_entry: MatrixEntry) -> None:
    """Run the CI session for the given matrix entry."""
    matrix_entry.run_session(session, posargs=session.posargs)


@dataclass(frozen=True, slots=True, kw_only=True)
class DownstreamEntry(MatrixEntry):
    """Represents an entry for downstream testing."""

    JWST_CRDS: ClassVar[dict[str, str]] = {
        "CRDS_SERVER_URL": "https://jwst-crds.stsci.edu"
    }
    ROMAN_CRDS: ClassVar[dict[str, str]] = {
        "CRDS_SERVER_URL": "https://roman-crds.stsci.edu"
    }

    package: str
    """The name of the downstream package."""
    repo: str
    """The URL of the downstream repository."""
    branch: str
    """The branch of the downstream repository to clone."""
    extra: tuple[str, ...] | None = None
    """The extra requirements to install for the downstream package."""

    pytest_extra: tuple[str, ...] = field(default_factory=tuple)
    """
    Additional arguments to pass to pytest when running the downstream package's tests.
    """
    env: dict[str, str] = field(default_factory=dict)
    """The environment variables to set when running the downstream package's tests."""

    @property
    def args(self) -> tuple[str, ...]:
        """The positional arguments for this matrix entry."""
        return (self.package,)

    def install(self, session: nox.Session, path: Path) -> Path:
        """
        Clone and then install the downstream repository to the specified path.

        Returns the path to the cloned downstream repository.
        """
        downstream = path / self.package

        session.run(
            "git",
            "clone",
            "--branch",
            self.branch,
            # A blobless clone keeps the tags that setuptools-scm needs to
            # determine a version, without paying for the full file history.
            "--filter=blob:none",
            self.repo,
            str(downstream),
            external=True,
        )

        with session.chdir(downstream):
            session.log(f"Installing the downstream package: {self.package}")
            # -e fixes issues with C extensions not being available for some reason
            session.install(
                "-e",
                f".[{', '.join(self.extra)}]" if self.extra is not None else ".",
            )

        return downstream

    def run_downstream(
        self,
        session: nox.Session,
        path: Path,
        session_args: list[str],
        pytest_args: list[str],
    ) -> None:
        """
        Run the downstream package's tests in the specified path.

        Assumes the downstream package has already been installed.

        Parameters
        ----------
        session :
            The Nox session to use for running the tests.
        path :
            The path to the downstream package.
        session_args :
            The arguments to pytest created by the session running the tests.
        pytest_args :
            The additional posargs that are not processed by the session and are
            passed directly to pytest.
        """
        # Combine all the provided arguments together for pytest
        arguments = session_args + list(self.pytest_extra) + pytest_args

        # Change into the directory that contains the downstream package before
        #   running tests.
        with session.chdir(path):
            # Note that we will run this with specific environment variables set.
            session.run("pytest", *arguments, env=self.env)


DOWNSTREAM_MATRIX = (
    DownstreamEntry(
        execute_session="run_downstream",
        package="jwst",
        repo="https://github.com/spacetelescope/jwst.git",
        branch="main",
        extra=("test",),
        env=DownstreamEntry.JWST_CRDS,
        options=("xdist",),
    ),
    DownstreamEntry(
        execute_session="run_downstream",
        package="romancal",
        repo="https://github.com/spacetelescope/romancal.git",
        branch="main",
        extra=("test",),
        env=DownstreamEntry.ROMAN_CRDS,
        options=("xdist",),
    ),
    DownstreamEntry(
        execute_session="run_downstream",
        package="stcal",
        repo="https://github.com/spacetelescope/stcal.git",
        branch="main",
        extra=("test",),
        env=DownstreamEntry.JWST_CRDS,
        options=("xdist",),
    ),
    DownstreamEntry(
        execute_session="run_downstream",
        package="romanisim",
        repo="https://github.com/spacetelescope/romanisim.git",
        branch="main",
        extra=("test",),
        env=DownstreamEntry.ROMAN_CRDS,
        options=("xdist",),
    ),
    DownstreamEntry(
        execute_session="run_downstream",
        package="specutils",
        repo="https://github.com/astropy/specutils.git",
        branch="main",
        extra=("test",),
        tags=("Downstream CI",),
    ),
    DownstreamEntry(
        execute_session="run_downstream",
        package="dkist",
        repo="https://github.com/DKISTDC/dkist.git",
        branch="main",
        extra=("tests",),
        pytest_extra=("--benchmark-skip",),
        options=("xdist",),
        tags=("Downstream CI",),
    ),
    DownstreamEntry(
        execute_session="run_downstream",
        package="ndcube",
        repo="https://github.com/sunpy/ndcube.git",
        branch="main",
        extra=("dev",),
        options=("xdist",),
        tags=("Downstream CI",),
    ),
)
DOWNSTREAM = MappingProxyType({entry.package: entry for entry in DOWNSTREAM_MATRIX})


@nox.session(python=None)
def downstream(session: nox.Session) -> None:
    """Run the downstream package tests."""
    if session.posargs and session.posargs[0] in PYTHON_SUPPORT.versions:
        target_python = session.posargs.pop(0)
    else:
        target_python = PYTHON_SUPPORT.default

    session.log(f"Target Python interpreter: {target_python}")
    session.notify(f"run_downstream-{target_python}", session.posargs)


@nox.session(python=PYTHON_SUPPORT.versions, reuse_venv=False)
def run_downstream(session: nox.Session) -> None:
    """Run the downstream package tests for python versions"""

    parser = argparse.ArgumentParser(
        prog="nox -s downstream --",
        allow_abbrev=False,
        description="Run a downstream package's tests against this version of gwcs.",
    )
    parser.add_argument(
        "package",
        choices=sorted(DOWNSTREAM),
        help="Downstream package to test against gwcs",
    )
    _add_standard_arguments(parser)

    args, pytest_args = parser.parse_known_args(session.posargs)
    downstream = DOWNSTREAM[args.package]

    # Clone into a temporary directory so stale state cannot leak between runs
    # and the repo working tree is never touched.
    # This uses the tempfile module instead of the session.create_tmp() method
    # so that the clone is performed freshly each time the session is run.
    with tempfile.TemporaryDirectory(prefix="gwcs-downstream-") as tmp_dir:
        # Install the downstream package and then gwcs
        #    Note the downstream must be first so that it does not clobber the
        #    gwcs installation.
        path = downstream.install(session, Path(tmp_dir))
        _install_gwcs(session, args)

        _list_dependencies(session)

        downstream.run_downstream(
            session, path, _init_pytest_arguments(session, args), pytest_args
        )


@nox.session(venv_backend="none")
@nox.parametrize("matrix_entry", [entry.nox_param for entry in DOWNSTREAM_MATRIX])
def run_downstream_ci(session: nox.session, matrix_entry: MatrixEntry) -> None:
    """Run the CI session for the given matrix entry."""
    matrix_entry.run_session(session, posargs=session.posargs)


def _parse_github_labels(value: str) -> list[str]:
    """
    Parse the JSON output of ``toJSON(github.event.pull_request.labels.*.name)``.

    Non-PR events produce ``null`` (or an empty string), which yields no labels.
    """
    if not value.strip():
        return []

    try:
        labels = json.loads(value)
    except json.JSONDecodeError as err:
        msg = f"labels must be a JSON array of strings, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from err

    if labels is None:
        return []

    if not isinstance(labels, list) or not all(isinstance(lbl, str) for lbl in labels):
        msg = f"labels must be a JSON array of strings, got {value!r}"
        raise argparse.ArgumentTypeError(msg)

    return labels


def _parse_github_bool(value: str) -> bool:
    """Parse a GitHub Actions boolean expression result (``true``/``false``)."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized in ("false", ""):
        return False

    msg = f"expected 'true' or 'false', got {value!r}"
    raise argparse.ArgumentTypeError(msg)


def _add_github_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels",
        type=_parse_github_labels,
        default=[],
        help=(
            "JSON array of PR label names, e.g. the output of "
            "${{ toJSON(github.event.pull_request.labels.*.name) }}"
        ),
    )
    parser.add_argument(
        "--force-run",
        type=_parse_github_bool,
        default=False,
        help="Include every matrix entry regardless of labels ('true' or 'false')",
    )


def _write_github_output(
    session: nox.Session,
    session_name: str,
    matrix: tuple[MatrixEntry, ...],
    labels: list[str],
    force_run: bool = False,
) -> None:
    """Write the GitHub Actions matrix to the environment."""

    outputs = [
        github_entry
        for entry in matrix
        if (
            github_entry := entry.github_matrix_entry(
                session_name, labels, force_run=force_run
            )
        )
        is not None
    ]

    if (github_output := os.getenv("GITHUB_OUTPUT")) is None:
        session.log("GITHUB_OUTPUT environment variable is not set, listing matrix:")
        for output in outputs:
            session.log(f"    {output}")

        session.error("GITHUB_OUTPUT environment variable is not set")
        return  # For mypy type checking the error should stop nox

    with Path(github_output).open("a", encoding="utf-8") as out:
        out.write(f"matrix={json.dumps(outputs)}\n")


@nox.session(venv_backend="none")
def ci_matrix(session: nox.Session) -> None:
    """Write the GitHub Actions matrix to the environment for the CI matrix."""
    parser = argparse.ArgumentParser(
        prog="nox -s ci_matrix --",
        allow_abbrev=False,
        description="Setup the GitHub Actions CI matrix.",
    )
    _add_github_matrix_arguments(parser)
    args = parser.parse_args(session.posargs)

    _write_github_output(
        session,
        "run_ci",
        CI_MATRIX,
        args.labels,
        force_run=args.force_run,
    )


@nox.session(venv_backend="none")
def downstream_ci_matrix(session: nox.Session) -> None:
    """
    Write the GitHub Actions matrix to the environment for the downstream CI matrix.
    """
    parser = argparse.ArgumentParser(
        prog="nox -s downstream_ci_matrix --",
        allow_abbrev=False,
        description="Setup the GitHub Actions downstream CI matrix.",
    )
    _add_github_matrix_arguments(parser)
    args = parser.parse_args(session.posargs)

    _write_github_output(
        session,
        "run_downstream_ci",
        DOWNSTREAM_MATRIX,
        args.labels,
        force_run=args.force_run,
    )
