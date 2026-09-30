from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import nox

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet

# Make nox default to uv if its available, if not fallback on virtualenv
nox.options.default_venv_backend = "uv|virtualenv"


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@nox.session(python=PYTHON_SUPPORT.versions)
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
        "--dev", action="store_true", help="Install development dependencies"
    )
    dependency_group.add_argument(
        "--oldest",
        action="store_true",
        help="Install the oldest compatible dependencies",
    )
    args, pytest_args = parser.parse_known_args(session.posargs)
    install_args: list[str] = []

    # Prepare the install arguments for dev/oldest if required
    if args.dev:
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
