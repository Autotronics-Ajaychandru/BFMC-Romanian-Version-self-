import time
import math
from .parking_config import MAX_SLOTS, SLOT_LENGTH_CM, DEBOUNCE_DURATION

class ParkingSlotManager:
    def __init__(self):
        self.occupancy_map = {}
        self.last_detection_time = {}
        self.current_slot = 1
        
        self.selected_slot = None
        self.selected_side = None

    def _init_slot(self, slot_num):
        if slot_num not in self.occupancy_map:
            self.occupancy_map[slot_num] = {'left': False, 'right': False}
            self.last_detection_time[slot_num] = {'left': 0.0, 'right': 0.0}

    def update_slot_occupancy(self, distance_cm, detected_cars):
        """
        Updates the current slot based on distance, and applies debounced 
        occupancy tracking from detected cars.
        Returns: True if a slot decision boundary was crossed (we finished scanning a slot).
        """
        calculated_slot = math.floor(distance_cm / SLOT_LENGTH_CM) + 1
        self._init_slot(calculated_slot)
        
        slot_crossed = False
        if calculated_slot > self.current_slot:
            slot_crossed = True
            
        current_time = time.time()
        slot_for_detection = calculated_slot
        
        for car in detected_cars:
            side_key = car['side']
            if current_time - self.last_detection_time[slot_for_detection][side_key] > DEBOUNCE_DURATION:
                self.occupancy_map[slot_for_detection][side_key] = True
                self.last_detection_time[slot_for_detection][side_key] = current_time
                
        return slot_crossed, calculated_slot

    def decide_target_slot(self):
        """
        Evaluates the slot that was just completely scanned (self.current_slot).
        Updates self.selected_side and self.selected_slot if valid.
        """
        left_occ = self.occupancy_map.get(self.current_slot, {}).get('left', False)
        right_occ = self.occupancy_map.get(self.current_slot, {}).get('right', False)
        
        side_chosen = None
        if not right_occ and left_occ:
            side_chosen = "right"
        elif not left_occ and right_occ:
            side_chosen = "left"
        elif not right_occ and not left_occ:
            side_chosen = "right"
            
        print(f"[Parking] Slot {self.current_slot}: Left {'occupied' if left_occ else 'free'}, Right {'occupied' if right_occ else 'free'}")
        
        if side_chosen is not None:
            self.selected_side = side_chosen
            self.selected_slot = self.current_slot
            print(f"[Parking] Target side: {self.selected_side.upper()}")
            
    def update_current_slot(self, calculated_slot):
        self.current_slot = calculated_slot
