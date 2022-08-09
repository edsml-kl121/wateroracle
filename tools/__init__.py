from .config import * # noqa
from .metrics_ import * # noqa
from .model import * # noqa
from .preprocessing import * # noqa
from .sampling import * # noqa
from .losses_ import * # noqa
from .images import * # noqa

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass