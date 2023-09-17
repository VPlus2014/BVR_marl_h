"""
    A PAC-3 Missile Unit
"""

from datetime import datetime
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from simulator.cmano_simulator import Unit, Position, Event, CmanoSimulator, units_distance_km, UnitDestroyedEvent
from utils.angles import signed_heading_diff

SS = 2

class Pac3Missile(Unit):
    max_deg_sec = 10
    speed_profile_time = np.array([0, 20, 36, 96, 156, 216, 276, 336])
    speed_profile_knots = np.array([150, 2300, 2300, 1238, 724, 510, 383, 350])  # Max speed was 2650
    speed_profile = interp1d(speed_profile_time, speed_profile_knots, kind='quadratic', assume_sorted=True,
                             bounds_error=False, fill_value=(150, 350))

    def __init__(self, position: Position, heading: float, firing_time: datetime, target: Unit, source: Unit):
        self.speed = Pac3Missile.speed_profile(0)
        super().__init__("Pac3Missile", position, heading, self.speed)
        self.new_heading = heading
        self.firing_time = firing_time
        self.target = target
        self.source = source

    def set_heading(self, new_heading: float):
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"Pac3Missile.set_heading Heading must be in [0, 360), got {new_heading}")
        self.new_heading = new_heading

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Check if the target has been hit
        if units_distance_km(self, self.target) < 1:
            sim.remove_unit(self.id)
            sim.remove_unit(self.target.id)
            return [UnitDestroyedEvent(self, self.source, self.target)]
        
        if SS == 2:
            # check if friendly aircraft has been hit
            friendly_id = 1 if self.source.id == 2 else 2
            if sim.unit_exists(friendly_id):
                friendly_unit = sim.get_unit(friendly_id)
                if units_distance_km(self, friendly_unit) < 1.5:
                    sim.remove_unit(self.id)
                    sim.remove_unit(friendly_id)
                    return [UnitDestroyedEvent(self, self.source, friendly_unit)]

        # Check if eol is arrived
        life_time = (sim.utc_time - self.firing_time).seconds
        if life_time > Pac3Missile.speed_profile_time[1]:
            sim.remove_unit(self.id)
            return []

        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Pac3Missile.max_deg_sec * tick_secs
            if abs(delta) <= max_deg:
                self.heading = self.new_heading
            else:
                self.heading += max_deg if delta >= 0 else -max_deg

        # Update speed
        self.speed = Pac3Missile.speed_profile(life_time)

        # Update position
        return super().update(tick_secs, sim)
