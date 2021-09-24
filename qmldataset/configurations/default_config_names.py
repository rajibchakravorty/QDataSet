"""Dictionary of default configuration names
"""

from . import config_1q_X
from . import config_1q_X_N1Z
from . import config_1q_X_N2Z
from . import config_1q_X_N3Z
from . import config_1q_X_N4Z
from . import config_1q_XY
from . import config_1q_XY_N1X_N5Z
from . import config_1q_XY_N1X_N6Z
from . import config_1q_XY_N3X_N6Z
from . import config_2q_IX_XI_XX
from . import config_2q_IX_XI_XX_N1N5IZ_N1N5ZI
from . import config_2q_IX_XI_XX_N1N6IZ_N1N6ZI
from . import config_2q_IX_XI_N1N6IZ_N1N6ZI

default_configs = {
    '1q_X': config_1q_X,
    '1q_X_N1Z': config_1q_X_N1Z,
    '1q_X_N2Z': config_1q_X_N2Z,
    '1q_X_N3Z': config_1q_X_N3Z,
    '1q_X_N4Z': config_1q_X_N4Z,
    '1q_XY': config_1q_XY,
    '1q_XY_N1X_N5Z': config_1q_XY_N1X_N5Z,
    '1q_XY_N1X_N6Z': config_1q_XY_N1X_N6Z,
    '1q_XY_N3X_N6Z': config_1q_XY_N3X_N6Z,
    '2q_IX_XI_XX': config_2q_IX_XI_XX,
    '2q_IX_XI_XX_N1N5IZ_N1N5ZI': config_2q_IX_XI_XX_N1N5IZ_N1N5ZI,
    '2q_IX_XI_XX_N1N6IZ_N1N6ZI': config_2q_IX_XI_XX_N1N6IZ_N1N6ZI,
    '2q_IX_XI_N1N6IZ_N1N6ZI': config_2q_IX_XI_N1N6IZ_N1N6ZI
}
