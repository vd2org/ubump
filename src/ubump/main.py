# Copyright (C) 2024 by Vd <ubump@vd2.org>.
# This file is part of ubump project.
# ubump is released under the MIT License (see LICENSE).


import logging
import os
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import suppress
from enum import StrEnum
from string import Template
from typing import Optional, NamedTuple, Self

import tomlkit
from packaging.version import parse
from tomlkit.exceptions import TOMLKitError

NAME = "ubump"
VERSION = "v0.1.0"

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


class Version(NamedTuple):
    major: int
    minor: int
    patch: int

    @classmethod
    def from_str(cls, version: str):
        v = cls(*parse(version).release)
        if v.to_str() != version:
            raise ValueError(f"Invalid version: {version}")
        return v

    def to_str(self, template: str = "${major}.${minor}.${patch}"):
        return Template(template).substitute(self._asdict())


class Config:
    def __init__(
            self,
            version: Version,
            template: Optional[str],
            message: Optional[str] = None,
            tag: Optional[bool] = None,
            files: Optional[list[str]] = None
    ):
        self._version = version
        self._template = template
        self._files = files or []
        self._tag = tag or True
        self._message = message or "Bump to {version}"

    @property
    def version(self) -> Version:
        return self._version

    @property
    def str_version(self) -> str:
        return self._version.to_str(self.template)

    @version.setter
    def version(self, value: Version):
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

        with open(mode, "r+") as file:
            content = tomlkit.load(file)

            if mode is ConfigMode.ubump:
                ubump["version"] = self.version.to_str()
                content["ubump"] = ubump
            else:
                if "project" not in content:
                    content["project"] = {}
                content["project"]["version"] = self.version.to_str()
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
            raise ConfigNotFoundError(f"No version config found, use 'init' command to create one.")

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

        return nok


class Git:
    @staticmethod
    def _call(*args, log_error: bool = True):
        try:
            subprocess.check_output(('git', *args))
            return True
        except FileNotFoundError:
            logger.warning("Git is not found, can't commit.")
            return False
        except subprocess.CalledProcessError:
            if log_error:
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
    def init(version: str, template: Optional[str], no_pyproject: bool = False, dry: bool = False):
        logger.info(f"Initializing ubump config...")

        with suppress(ConfigNotFoundError):
            Config.try_load()
            raise ConfigError("Version config is already exist!")

        logger.info(f"Initializing ubump config using version {version} and template {template}...")

        if not no_pyproject:
            with suppress(FileNotFoundError):
                with open(ConfigMode.pyproject, "rb"):
                    mode = ConfigMode.pyproject
        else:
            mode = ConfigMode.ubump

        config = Config(
            version=Version.from_str(version),
            template=template or "v${major}.${minor}.${patch}"
        )

        logger.info(f"Using {mode} as config file...")

        logger.info(f"Searching for files contains current version {config.str_version}...")
        config.files = Tools.walk(config)

        if not dry:
            config.save(mode)

        logger.info("Done.")

    @staticmethod
    def bump(action: str, version: Optional[str] = None, dry: bool = False):
        mode, config = Config.try_load()

        if not Git.is_repo_clean() and not dry:
            logger.error(f"Git repo is not clean, aborting...")
            return

        logger.info(f"Using version config from {mode}...")

        old_str_version = config.str_version

        if not version:
            logger.info(f"Bumping version {action} from {old_str_version}...")

        if action == "set":
            logger.info(f"Setting version to {version} from {old_str_version}...")
            config.version = Version.from_str(version)
        elif action == "major":
            config.version = config.version._replace(major=config.version.major + 1, minor=0, patch=0)
        elif action == "minor":
            config.version = config.version._replace(minor=config.version.minor + 1, patch=0)
        elif action == "patch":
            config.version = config.version._replace(patch=config.version.patch + 1)

        if old_str_version == config.str_version:
            logger.error(f"The version is already {config.str_version}, nothing to do.")
            return

        if not version:
            logger.info(f"The new version is {config.version.to_str()}...")

        logger.info(f"Updating files...")

        nok = Tools.update_files(config, old_str_version, dry=True)

        if nok:
            logger.error(f"The current version is not found in some files, aborting...")
            return

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

    init_ubump = subs.add_parser("init", help="Initialize ubump config.")
    init_ubump.add_argument("--no-pyproject", default=False, action="store_true",
                            help="Don't use pyproject.toml, use .ubump.toml instead.")
    init_ubump.add_argument("version", help="Current version.")
    init_ubump.add_argument("-t", "--template", default="v${major}.${minor}.${patch}", help="The version template.")

    major = subs.add_parser("major", help="Bump major version.")

    minor = subs.add_parser("minor", help="Bump minor version.")

    patch = subs.add_parser("patch", help="Bump patch version.")

    set_version = subs.add_parser("set", help="Set version.")
    set_version.add_argument("version", help="The new version to set.")

    for sub in [init_ubump, major, minor, patch, set_version]:
        sub.add_argument("--dry", default=False, action="store_true", help="Dry run, don't change anything.")

    args = vars(parser.parse_args())

    try:
        if args.get("dry"):
            logger.warning("Dry run, nothing will be changed!")

        if args.get("action") == "init":
            args.pop("action")
            Actions.init(**args)
        else:
            Actions.bump(**args)
    except ConfigError as e:
        logger.error(e)
        sys.exit(-1)
    except Exception as e:
        logger.exception(f"Something weird happened: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
