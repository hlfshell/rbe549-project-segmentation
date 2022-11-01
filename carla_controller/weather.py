import carla

from random import choices, uniform


def random_weather():
    params = choices(
        [clear, stormy, light_rain, foggy, cloudy],
        weights=[30, 15, 20, 10, 25],
        k=1
    )[0]()
    return randomize_sun(params)


def randomize_sun(params):
    # -90 = midnight, 90 = midday
    params.sun_altitude_angle = uniform(-90, 90)
    params.sun_azimuth_angle = uniform(80, 170)

    return params


def clear():
    params = carla.WeatherParameters()

    return params

def stormy():
    params = carla.WeatherParameters()

    params.cloudiness = uniform(50, 100)
    params.precipitation = uniform(50, 100)
    params.precipitation_deposits = uniform(50, 100)
    params.wind_intensity = uniform(40,100)
    params.fog_density = uniform(0, 50)

    return params

def light_rain():
    params = carla.WeatherParameters()

    params.cloudiness = uniform(10, 40)
    params.precipitation = uniform(10, 35)
    params.precipitation_deposits = uniform(0, 25)
    params.wind_intensity = uniform(0, 35)

    return params


def foggy():
    params = carla.WeatherParameters()

    params.cloudiness = uniform(25, 70)
    params.precipitation = uniform(10, 35)
    params.precipitation_deposits = uniform(0, 25)
    params.fog_density = uniform(30, 100)

    return params


def cloudy():
    params = carla.WeatherParameters()

    params.cloudiness = uniform(25, 70)
    params.precipitation_deposits(0, 25)
    params.wetness = uniform(0, 20)

    return params