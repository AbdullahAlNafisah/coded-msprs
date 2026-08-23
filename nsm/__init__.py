from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nsm")
except PackageNotFoundError:
    __version__ = "unknown"

from nsm import modem
from nsm import channel, codec
