from enum import Enum

class Worldsize(Enum):
    MINI = 7
    TINY = 11
    SMALL = 14
    MEDIUM = 16
    LARGE = 18
    HUGE = 20
    MASSIVE = 30

class Worldtype(Enum):
    PLAINS_4_CITY = 0
    ARCHEPELIGO = 1
    PANGEA = 2

def generate_world(worldsize, worldtype):
    match worldtype:
        case PLAINS_4_CITY:
