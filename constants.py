from enum import Enum

class WindowType(Enum):
    Variable = 'Variable_Length'
    Fixed = 'Fixed_Length'

class Models(Enum):
    Transfomer = 'Transformer'

class Subsets(Enum):
    FD001 = 'FD001'
    FD002 = 'FD002'
    FD003 = 'FD003'
    FD004 = 'FD004'
    
class MomentumGM(Enum):
    gtw = 'gtw'
    gow = 'gow'
    wog = 'wog'
    cold_start = 'cold'
    hot_start = 'hot' 

class GM_Method(Enum):
    Vanilla_GM = 'Vanilla_GM'
    Momentum_GM = 'Momentum_GM'
    Adaptive_MGM = 'Adaptive_MGM'