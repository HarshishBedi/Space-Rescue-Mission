import math
import random

from src.Utilities.utility import get_manhattan_distance


class Crew_Member_Sensor:
    def __init__(self,crew_member_positions:list[tuple[int,int]],alpha:float):
        self.crew_member_positions = crew_member_positions
        self.alpha = alpha

    def crew_members_beep(self,bot_position:tuple[int,int]):
        for posn  in self.crew_member_positions:
            if self.crew_member_beep(posn,bot_position):
                return True
        return False

    def crew_member_beep(self, crew_member_position: tuple[int,int], bot_position: tuple[int,int])->bool:
        d = get_manhattan_distance(crew_member_position,bot_position)
        chance_of_beep = math.exp(-self.alpha*(d-1))
        random_number = random.uniform(0,1)
        if random_number <= chance_of_beep:
            return True
        return False

