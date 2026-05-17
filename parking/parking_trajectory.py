import csv

class ParkingTrajectory:
    def __init__(self, left_csv="csv/left_parallel_parking.csv", right_csv="csv/right_parallel_parking.csv"):
        self.left_csv_path = left_csv
        self.right_csv_path = right_csv
        self.trajectory_points = None

    def _load_csv(self, file_path):
        trajectory = []
        try:
            with open(file_path, mode='r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        trajectory.append({
                            "time": float(row[0]),
                            "steering": float(row[1]),
                            "speed": float(row[2])
                        })
        except Exception as e:
            print(f"[Parking] Error loading CSV {file_path}: {e}")
        return trajectory

    def load_left_trajectory(self):
        self.trajectory_points = self._load_csv(self.left_csv_path)
        return self.trajectory_points
        
    def load_right_trajectory(self):
        self.trajectory_points = self._load_csv(self.right_csv_path)
        return self.trajectory_points
