"""Dictionary of default configuration names
"""

from . import config_1q_X
from . import config_1q_XZ_N1
from . import config_1q_XZ_N2
from . import config_1q_XZ_N3
from . import config_1q_XZ_N4
from . import config_1q_XY
from . import config_1q_XY_XZ_N1N5
from . import config_1q_XY_XZ_N1N6
from . import config_1q_XY_XZ_N3N6
from . import config_2q_IX_XI_XX
from . import config_2q_IX_XI_XX_IZ_ZI_N1N5
from . import config_2q_IX_XI_XX_IZ_ZI_N1N6
from . import config_2q_IX_XI_IZ_ZI_N1N6

default_configs = {
    '1q_X': config_1q_X,
    '1q_XZ_N1': config_1q_XZ_N1,
    '1q_XZ_N2': config_1q_XZ_N2,
    '1q_XZ_N3': config_1q_XZ_N3,
    '1q_XZ_N4': config_1q_XZ_N4,
    '1q_XY': config_1q_XY,
    '1q_XY_XZ_N1N5': config_1q_XY_XZ_N1N5,
    '1q_XY_XZ_N1N6': config_1q_XY_XZ_N1N6,
    '1q_XY_XZ_N3N6': config_1q_XY_XZ_N3N6,
    '2q_IX_XI_XX': config_2q_IX_XI_XX,
    '2q_IX_XI_XX_IZ_ZI_N1N5': config_2q_IX_XI_XX_IZ_ZI_N1N5,
    '2q_IX_XI_XX_IZ_ZI_N1N6': config_2q_IX_XI_XX_IZ_ZI_N1N6,
    '2q_IX_XI_IZ_ZI_N1N6': config_2q_IX_XI_IZ_ZI_N1N6
}
