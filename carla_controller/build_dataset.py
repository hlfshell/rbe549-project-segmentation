from simulation import Simulation

from random import randrange

TIME_FOR_RUN = 60
MIN_NUMBER_VEHICLES = 100
MAX_NUMBER_VEHICLES = 500
MIN_NUMBER_PEDESTRIANS = 100
MAX_NUMBER_PEDESTRIANS = 300

def single_run():
    number_of_vehicles = randrange(MIN_NUMBER_VEHICLES, MAX_NUMBER_VEHICLES + 1)
    number_of_pedestrians = randrange(MIN_NUMBER_PEDESTRIANS, MAX_NUMBER_PEDESTRIANS + 1)

    simulation = Simulation(vehicles=number_of_vehicles, pedestrians=number_of_pedestrians)
    simulation.launch(TIME_FOR_RUN)

def build_dataset(runs : int):
    for _ in range(runs):
        single_run()

if __name__ == '__main__':
    build_dataset(100)