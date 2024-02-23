# ubump

Yet another version bumper for your project.

## Why?

I wanted to have a simple version bumper that I could use in my CI/CD pipeline.
I didn't want to have to install a bunch of dependencies or have to write a bunch of configs to just change the version number.

This tool having less than 400 lines of code and just one dependency (`tomlkit`) is the result of that.

## What does it do?

This tool covers basic version management where only the major, minor, and patch numbers are used.

Supporting `pyproject.toml` or `.ubump.toml` configuration files.

The git commit and tag are also created with the new version number.

If you need more than that take a look at [bump-my-version](https://github.com/callowayproject/bump-my-version) project.

## Installation

```shell
pip install ubump
```

## Usage

- Initialize the config file

```shell
ubump init 
```

- Bump the version

```shell
ubump pathch
```

```shell
ubump minor
```

```shell
ubump major
```

**That's it!**
