# Copyright (C) 2024 by Vd <ubump@vd2.org>.
# This file is part of ubump project.
# ubump is released under the MIT License (see LICENSE).


import logging
import os
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import suppress
from dataclasses import dataclass, asdict, replace
from enum import StrEnum, auto
from string import Template
from typing import Optional, Self

import tomlkit
from packaging.version import parse
from tomlkit.exceptions import TOMLKitError

NAME = "ubump"
VERSION = "v0.1.13"

DEFAULT_TEMPLATE = "${major}.${minor}.${patch}"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(NAME)
logger.setLevel(logging.DEBUG if sys.flags.debug else logging.INFO)


class ConfigError(ValueError):
    pass


class ConfigNotFoundError(ConfigError):
    pass


class BothConfigFoundError(ConfigError):
    pass


class ConfigMode(StrEnum):
    pyproject = "pyproject.toml"
    ubump = ".ubump.toml"


class Action(StrEnum):
    init = auto()
    major = auto()
    minor = auto()
    patch = auto()
    set = auto()


@dataclass(frozen=True)
class Version:
    major: int
    minor: Optional[int] = None
    patch: Optional[int] = None

    @classmethod
    def from_str(cls, version: str):
        v = cls(*parse(version).release)
        if v.to_raw_str() != version:
            raise ConfigError(f"Invalid version: {version} != {v.to_raw_str()}!")
        return v

    def to_str(self, template: str = DEFAULT_TEMPLATE):
        return Template(template).substitute(asdict(self))

    def to_raw_str(self):
        match (self.major, self.minor, self.patch):
            case (major, None, None):
                template = f"{major}"
            case (major, minor, None):
                template = f"{major}.{minor}"
            case (major, minor, patch):
                template = f"{major}.{minor}.{patch}"
            case _:
                raise RuntimeError(f"Unknown version configuration: {self}")

        return Template(template).substitute(asdict(self))


class Config:
    def __init__(
            self,
            version: Version,
            template: str,
            message: Optional[str] = None,
            tag: Optional[bool] = None,
            files: Optional[list[str]] = None
    ):
        self._version = version
        self._template = template
        self._files = files or []
        self._tag = tag or True
        self._message = message or "Bump to ${version}"
        self._validate()

    def _validate(self, new_version: Optional[Version] = None):
        version = new_version or self.version
        template_ids = set(Template(self.template).get_identifiers())
        version_keys = set(k for k, v in asdict(version).items() if v is not None)
        if template_ids != version_keys:
            raise ConfigError(f"Version {version.to_raw_str()} isn't suitable for template {self.template}!")

    @property
    def version(self) -> Version:
        return self._version

    @property
    def str_version(self) -> str:
        return self._version.to_str(self.template)

    @version.setter
    def version(self, value: Version):
        self._validate(value)
        self._version = value

    @property
    def template(self) -> str:
        return self._template

    @property
    def message(self) -> str:
        return self._message

    @property
    def tag(self) -> bool:
        return self._tag

    @property
    def files(self) -> list[str]:
        return self._files.copy()

    @files.setter
    def files(self, value: list[str]):
        self._files = value.copy()

    def save(self, mode: ConfigMode):
        ubump = {
            "template": self.template,
            "message": self._message,
            "tag": self.tag,
            "files": self.files
        }

        with open(mode, "a+") as file:
            file.seek(0)
            content = tomlkit.load(file)

            if mode is ConfigMode.ubump:
                ubump["version"] = self.version.to_raw_str()
                content["ubump"] = ubump
            else:
                if "project" not in content:
                    content["project"] = {}
                content["project"]["version"] = self.version.to_raw_str()
                if "tool" not in content:
                    content["tool"] = {}
                content["tool"]["ubump"] = ubump

            file.truncate(0)
            file.seek(0)
            tomlkit.dump(content, file)

    @classmethod
    def load(cls, mode: ConfigMode) -> Self:
        try:
            with open(mode, "r") as file:
                content = tomlkit.load(file)

                if mode is ConfigMode.ubump:
                    version = Version.from_str(content["ubump"]["version"])
                    data = content["ubump"]
                else:
                    version = Version.from_str(content["project"]["version"])
                    data = content["tool"]["ubump"]

                return cls(
                    version=version,
                    template=data["template"],
                    message=data["message"],
                    tag=data["tag"],
                    files=data["files"]
                )
        except (OSError, TOMLKitError, KeyError) as e:
            logger.debug(f"Can't load config from file {mode}: {e}")

        raise ConfigNotFoundError(f"Can't load config from file {mode}!")

    @classmethod
    def try_load(cls) -> tuple[ConfigMode, Self]:
        config = pyproject = None

        with suppress(ConfigNotFoundError):
            config = Config.load(ConfigMode.ubump)
            mode = ConfigMode.ubump
        with suppress(ConfigNotFoundError):
            pyproject = Config.load(ConfigMode.pyproject)
            mode = ConfigMode.pyproject

        if config and pyproject:
            raise BothConfigFoundError(f"Version config found in both files: {mode.pyproject} and {mode.ubump}!")

        if not config and not pyproject:
            raise ConfigNotFoundError(f"No version config found, use '{Action.init}' command to create one.")

        return mode, config or pyproject


class Tools:
    @staticmethod
    def walk(config):
        files = []
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in filenames:
                full = os.path.join(dirpath, filename).removeprefix("./")

                if filename in iter(ConfigMode):
                    logger.debug(f"Skipping config file {full}...")
                    continue

                if any(part.startswith(".") for part in full.split(os.sep)):
                    logger.debug(f"Skipping dotted {full}...")
                    continue

                try:
                    with open(full, 'rb') as f:
                        if b'\0' in f.read(1024):
                            logger.debug(f"Skipping binary {full}...")
                            continue

                    with open(full, "r") as file:
                        content = file.read()
                        if config.str_version not in content:
                            logger.debug(f"Version not found in {full}...")
                            continue

                    files.append(full)
                    logger.info(f"Found {full}...")
                except (OSError, UnicodeDecodeError):
                    logger.warning(f"Can't open {full} due to error, skipping...")
                    continue
        return files

    @staticmethod
    def update_files(config: Config, old_str_version: str, *, dry: bool = False):
        cwd = os.getcwd()
        nok = False
        for file_name in config.files:
            with open(os.path.join(cwd, file_name), "r+") as file:
                content = file.read()
                new_content = content.replace(old_str_version, config.str_version)
                if new_content == content:
                    logger.error(f"Version not found in {file}...")
                    nok = True

                if dry:
                    continue

                file.truncate(0)
                file.seek(0)
                file.write(new_content)
            logger.info(f"Updated {file_name}.")

        return nok


class Git:
    @staticmethod
    def _call(*args, log_error: bool = True):
        logger.debug(f"Calling git with {args}...")
        try:
            subprocess.check_output(('git', *args))
            return True
        except FileNotFoundError:
            logger.warning("Git is not found, can't commit.")
            return False
        except subprocess.CalledProcessError:
            if log_error or sys.flags.debug:
                logger.error("Call to git failed.")
            return False

    @classmethod
    def is_repo_clean(cls) -> bool:
        return cls._call('diff', '--exit-code', '--quiet', log_error=False)

    @classmethod
    def commit(cls, message: str) -> bool:
        return cls._call('commit', '-am', message)

    @classmethod
    def tag(cls, tag: str, message: str) -> bool:
        return cls._call('tag', '-a', tag, '-m', message)


class Actions:
    @staticmethod
    def init(version: str, template: str, no_pyproject: bool = False, dry: bool = False):
        logger.info(f"Initializing ubump config...")

        with suppress(ConfigNotFoundError):
            Config.try_load()
            raise ConfigError("Version config is already exist!")

        logger.info(f"Initializing ubump config using version {version} and template {template}...")

        mode = None
        if not no_pyproject:
            with suppress(FileNotFoundError):
                with open(ConfigMode.pyproject, "rb"):
                    mode = ConfigMode.pyproject

        if not mode:
            mode = ConfigMode.ubump

        logger.info(f"Using {mode} as config file...")

        config = Config(version=Version.from_str(version), template=template)

        logger.info(f"Searching for files contains current version {config.str_version}...")
        config.files = Tools.walk(config)

        if not dry:
            config.save(mode)

        logger.info("Done.")

    @staticmethod
    def bump(action: Action, version: Optional[str] = None, dry: bool = False):
        mode, config = Config.try_load()

        if not Git.is_repo_clean() and not dry:
            raise ConfigError(f"Git repo is not clean, aborting...")

        logger.info(f"Using version config from {mode}...")

        old_str_version = config.str_version

        if not version:
            logger.info(f"Bumping version {action} from {old_str_version}...")

        match action:
            case Action.set:
                ver = Version.from_str(version)
                logger.info(f"Setting version to {ver.to_str(config.template)} from {old_str_version}...")
                config.version = Version.from_str(version)
            case Action.major:
                config.version = replace(config.version, major=config.version.major + 1,
                                         minor=0 if config.version.minor is not None else None,
                                         patch=0 if config.version.patch is not None else None)
            case Action.minor:
                if config.version.minor is None:
                    raise ConfigError(f"Can't bump minor version for {config.version.to_raw_str()}!")
                config.version = replace(config.version, minor=config.version.minor + 1,
                                         patch=0 if config.version.patch is not None else None)
            case Action.patch:
                if config.version.patch is None:
                    raise ConfigError(f"Can't bump patch version for {config.version.to_raw_str()}!")
                config.version = replace(config.version, patch=config.version.patch + 1)
            case _:
                raise RuntimeError(f"Unknown action: {action}")

        if old_str_version == config.str_version:
            raise ConfigError(f"The version is already {config.str_version}, nothing to do.")

        if not version:
            logger.info(f"The new version is {config.str_version}...")

        logger.info(f"Updating files...")
        nok = Tools.update_files(config, old_str_version, dry=True)
        if nok:
            raise ConfigError(f"The current version is not found in some files, aborting...")

        if not dry:
            Tools.update_files(config, old_str_version)

            message = Template(config.message).substitute(version=config.str_version)
            config.save(mode)
            Git.commit(message)
            Git.tag(config.str_version, message)

        logger.info("Done.")


def main():
    parser = ArgumentParser(prog=NAME, description="Minimalistic version bumper.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    subs = parser.add_subparsers(title="Action", metavar="action", dest="action")

    init = subs.add_parser(Action.init, help="Initialize ubump config.")
    init.add_argument("--no-pyproject", default=False, action="store_true",
                      help=f"Don't use {ConfigMode.pyproject}, use {ConfigMode.ubump} instead.")
    init.add_argument("version", help="Current version.")
    init.add_argument("-t", "--template", default=f"v{DEFAULT_TEMPLATE}",
                      help="The version template.")

    major = subs.add_parser(Action.major, help="Bump major version.")

    minor = subs.add_parser(Action.minor, help="Bump minor version.")

    patch = subs.add_parser(Action.patch, help="Bump patch version.")

    set_version = subs.add_parser(Action.set, help="Set version.")
    set_version.add_argument("version", help="The new version to set.")

    for sub in (init, major, minor, patch, set_version):
        sub.add_argument("--dry", default=False, action="store_true",
                         help="Dry run, don't change anything.")

    args = vars(parser.parse_args())

    try:
        if args.get("dry"):
            logger.warning("Dry run, nothing will be changed!")

        if args.get("action") == Action.init:
            args.pop("action")
            Actions.init(**args)
        else:
            if not args.get("action"):
                args["action"] = Action.patch
            Actions.bump(**args)
    except ConfigError as e:
        logger.error(e)
        sys.exit(-1)
    except Exception as e:
        logger.exception(f"Something weird happened: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
