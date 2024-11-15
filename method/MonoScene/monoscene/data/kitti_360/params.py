import numpy as np

kitti_360_class_frequencies = np.array(
    [
        2264087502,
        20098728,
        104972,
        96297,
        1149426,
        4051087,
        125103,
        105540713,
        16292249,
        45297267,
        14454132,
        110397082,
        6766219,
        295883213,
        50037503,
        1561069,
        406330,
        30516166,
        1950115,
    ]
)

kitti_360_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "terrain",
    "pole",
    "traffic-sign",
    "other-structure",
    "other-object",
]

kitti_360_unified_class_frequencies = np.array(
    [
        2305812911,
        123212463,
        96297,
        4051087,   
        45297267,
        110397082,  
        295883213,   
        50037503,    
        1561069,   
        30516166,
        1950115
    ]
)

kitti_360_unified_class_names = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "person",
    "road",
    "sidewalk",
    "other-ground",
    "building",
    "vegetation",
    "other-object",
]
