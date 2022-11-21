

SEMANTIC_CATEGORIES = {
    0: "Unlabeled",
    1: "Buildings",
    2: "Traffic Sign/Lights",
    3: "Roads",
    4: "Road Lines",
    5: "Sidewalk",
    6: "Ground",
    7: "Vehicles",
    8: "Pedestrians",
}


# By default, CARLA has a set number of categories, but depending on what
# version, a different number of labels. The default documentation
# https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-semantic-segmentation
# lists 13 labels; however the latest build actually has an additional
# 10 for 23 labels.
# https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
# Luckily, the overlapping label ID's are equivalent.
# We don't wish to cover all the categories of these labels, and want to
# consolidate it into our own. Thus SEMANTIC_CATEGORIES are our actual
# labels, and below is the CARLA=> our labels mapping.
# Thus irregardless if it's a dataset with 0-22 or 0-13, we can map it
# back to this
CARLA_SEMANTIC_CATEGORY_MAPPINGS = {
    0: 0, # Unlabeled
    1: 1, # Building
    2: 1, # Fence
    3: 0, # Other
    4: 8, # Pedestrian
    5: 1, # Pole
    6: 4, # RoadLine
    7: 3, # Road
    8: 5, # SideWalk
    9: 0, # Vegetation
    10: 7, # Vehicles
    11: 1, # Wall
    12: 2, # TrafficSign
    13: 0, # Sky
    14: 6, # Ground
    15: 1, # Bridge
    16: 0, # RailTrack
    17: 1, # GuardRail
    18: 2, # TrafficLight
    19: 0, # Static
    20: 0, # Dynamic
    21: 6, # Water
    22: 6, # Terrain
}


SEMANTIC_COLORS = {
    0: (0, 0, 0), # Unlabeled
    1: (255, 0, 0), # Buildings
    2: (220, 220, 0), # Traffic Sign/Lights
    3: (0, 255, 0),#(128, 64, 128), # Roads
    4: (157, 234, 50), # Road Lines
    5: (244, 35, 232), # Sidewalk
    6: (107, 142, 35), # Ground
    7: (0, 0, 255), # Vehicles
    8: (220, 20, 60), # Pedestrians
}