# coco_colors.py
coco_classes = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", 
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", 
    "bed", "N/A", "dining table", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
# COCO class labels (80 object categories) and their corresponding colors
coco_colors = [
    (0, 0, 0),        # 0: Background
    (128, 0, 0),      # 1: Person
    (0, 128, 0),      # 2: Bicycle
    (0, 0, 128),      # 3: Car
    (128, 128, 0),    # 4: Motorcycle
    (128, 0, 128),    # 5: Airplane
    (0, 128, 128),    # 6: Bus
    (128, 128, 128),  # 7: Train
    (64, 0, 0),       # 8: Truck
    (192, 0, 0),      # 9: Boat
    (64, 128, 0),     # 10: Traffic light
    (192, 128, 0),    # 11: Fire hydrant
    (64, 128, 128),   # 12: Stop sign
    (192, 128, 128),  # 13: Parking meter
    (0, 0, 64),       # 14: Bench
    (0, 0, 192),      # 15: Bird
    (128, 128, 64),   # 16: Cat
    (128, 0, 64),     # 17: Dog
    (128, 128, 192),  # 18: Horse
    (0, 128, 192),    # 19: Sheep
    (0, 0, 64),       # 20: Cow
    (64, 64, 0),      # 21: Elephant
    (0, 0, 128),      # 22: Bear
    (64, 0, 128),     # 23: Zebra
    (192, 128, 128),  # 24: Giraffe
    (64, 64, 128),    # 25: Backpack
    (0, 128, 64),     # 26: Umbrella
    (128, 128, 192),  # 27: Handbag
    (0, 0, 128),      # 28: Tie
    (128, 0, 0),      # 29: Suitcase
    (0, 128, 128),    # 30: Frisbee
    (128, 128, 0),    # 31: Skis
    (192, 0, 128),    # 32: Snowboard
    (0, 128, 192),    # 33: Sports ball
    (128, 0, 192),    # 34: Kite
    (64, 64, 192),    # 35: Baseball bat
    (64, 0, 64),      # 36: Baseball glove
    (0, 128, 64),     # 37: Skateboard
    (128, 128, 128),  # 38: Surfboard
    (192, 0, 192),    # 39: Tennis racket
    (0, 0, 0),        # 40: Bottle
    (192, 0, 0),      # 41: Wine glass
    (64, 128, 128),   # 42: Cup
    (128, 64, 0),     # 43: Fork
    (128, 64, 128),   # 44: Knife
    (192, 128, 64),   # 45: Spoon
    (0, 128, 0),      # 46: Bowl
    (0, 0, 128),      # 47: Banana
    (64, 128, 192),   # 48: Apple
    (192, 128, 0),    # 49: Sandwich
    (64, 0, 128),     # 50: Orange
    (128, 0, 64),     # 51: Broccoli
    (128, 128, 0),    # 52: Carrot
    (64, 64, 0),      # 53: Hot dog
    (192, 0, 128),    # 54: Pizza
    (0, 128, 192),    # 55: Donut
    (128, 128, 128),  # 56: Cake
    (0, 0, 64),       # 57: Chair
    (128, 128, 192),  # 58: Couch
    (192, 0, 0),      # 59: Potted plant
    (0, 128, 0),      # 60: Bed
    (64, 0, 0),       # 61: Dining table
    (192, 128, 128),  # 62: Toilet
    (64, 128, 0),     # 63: TV
    (128, 0, 0),      # 64: Laptop
    (0, 128, 64),     # 65: Mouse
    (128, 128, 0),    # 66: Remote
    (64, 0, 128),     # 67: Keyboard
    (255, 255, 0),    # 68: Cell phone
    (64, 128, 64),    # 69: Microwave
    (128, 0, 192),    # 70: Oven
    (0, 128, 128),    # 71: Toaster
    (128, 128, 128),  # 72: Sink
    (64, 64, 64),     # 73: Refrigerator
    (0, 0, 0),        # 74: Book
    (192, 0, 0),      # 75: Clock
    (64, 128, 128),   # 76: Vase
    (128, 64, 0),     # 77: Scissors
    (192, 128, 192),  # 78: Teddy bear
    (128, 0, 0),      # 79: Hair drier
    (0, 0, 192)       # 80: Toothbrush
]

# Function to get color for a class ID
def get_class_color(class_id):
    if 0 <= class_id < len(coco_colors):
        return coco_colors[class_id]
    else:
        return (0, 0, 0)  # Default to black if the class ID is invalid
