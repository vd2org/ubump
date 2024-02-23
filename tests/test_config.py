from unittest import mock

from ubump import main


def test_config():
    with mock.patch("builtins.open", mock.mock_open(read_data=b'file content'), side_effect=FileNotFoundError):
        config = main.Config()

        print(config.file)