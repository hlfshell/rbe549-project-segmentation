from simulation import Simulation

from random import randrange


TIME_FOR_RUN_SECONDS = 60
NUMBER_OF_RUNS = 100
MIN_NUMBER_VEHICLES = 30
MAX_NUMBER_VEHICLES = 200
MIN_NUMBER_PEDESTRIANS = 20
MAX_NUMBER_PEDESTRIANS = 100

def create_movie() -> str:
    number_of_vehicles = randrange(MIN_NUMBER_VEHICLES, MAX_NUMBER_VEHICLES + 1)
    number_of_pedestrians = randrange(MIN_NUMBER_PEDESTRIANS, MAX_NUMBER_PEDESTRIANS + 1)

    simulation = Simulation(vehicles=number_of_vehicles, pedestrians=number_of_pedestrians, run_id="movie")
    simulation.record_movie_run(TIME_FOR_RUN_SECONDS)

    return simulation.run_id


if __name__ == '__main__':
    print("Creating CARLA movie")
    create_movie()
    print("Completed movie run")