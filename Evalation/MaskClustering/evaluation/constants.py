MATTERPORT_LABELS = ('door', 'picture', 'window', 'chair', 'pillow', 'lamp', 
                            'cabinet', 'curtain', 'table', 'plant', 'mirror', 'towel', 'sink', 'shelves', 'sofa', 
                            'bed', 'night stand', 'toilet', 'column', 'banister', 'stairs', 'stool', 'vase', 
                            'television', 'pot', 'desk', 'box', 'coffee table', 'counter', 'bench', 'garbage bin', 
                            'fireplace', 'clothes', 'bathtub', 'book', 'air vent', 'faucet', 'photo', 'toilet paper', 
                            'fan', 'railing', 'sculpture', 'dresser', 'rug', 'ottoman', 'bottle', 'refridgerator', 
                            'bookshelf', 'wardrobe', 'pipe', 'monitor', 'stand', 'drawer', 'container', 'light switch', 
                            'purse', 'door way', 'basket', 'chandelier', 'oven', 'clock', 'stove', 'washing machine', 
                            'shower curtain', 'fire alarm', 'bin', 'chest', 'microwave', 'blinds', 'bowl', 'tissue box', 
                            'plate', 'tv stand', 'shoe', 'heater', 'headboard', 'bucket', 'candle', 'flower pot', 
                            'speaker', 'furniture', 'sign', 'air conditioner', 'fire extinguisher', 'curtain rod', 
                            'floor mat', 'printer', 'telephone', 'blanket', 'handle', 'shower head', 'soap', 'keyboard', 
                            'thermostat', 'radiator', 'kitchen island', 'paper towel', 'sheet', 'glass', 'dishwasher', 
                            'cup', 'ladder', 'garage door', 'hat', 'exit sign', 'piano', 'board', 'rope', 'ball', 
                            'excercise equipment', 'hanger', 'candlestick', 'light', 'scale', 'bag', 'laptop', 'treadmill', 
                            'guitar', 'display case', 'toilet paper holder', 'bar', 'tray', 'urn', 'decorative plate', 'pool table', 
                            'jacket', 'bottle of soap', 'water cooler', 'utensil', 'tea pot', 'stuffed animal', 'paper towel dispenser', 
                            'lamp shade', 'car', 'toilet brush', 'doll', 'drum', 'whiteboard', 'range hood', 'candelabra', 'toy', 
                            'foot rest', 'soap dish', 'placemat', 'cleaner', 'computer', 'knob', 'paper', 'projector', 'coat hanger', 
                            'case', 'pan', 'luggage', 'trinket', 'chimney', 'person', 'alarm')

MATTERPORT_IDS = [28, 64, 59, 5, 119, 144, 3, 89, 19, 82, 122, 135, 24, 42, 83, 157, 158, 124, 94, 453, 
 215, 150, 78, 172, 16, 36, 26, 356, 7, 204, 12, 372, 141, 136, 1, 25, 9, 508, 139, 74, 497, 294, 
 169, 130, 359, 2, 17, 88, 772, 41, 49, 50, 174, 140, 301, 181, 609, 39, 342, 238, 56, 242, 278, 
 123, 338, 307, 344, 13, 80, 22, 138, 233, 291, 149, 111, 161, 427, 137, 146, 54, 524, 208, 79, 
 10, 582, 143, 66, 32, 312, 758, 650, 133, 47, 110, 236, 456, 113, 559, 612, 8, 35, 48, 850, 193, 
 86, 298, 408, 560, 60, 457, 211, 148, 62, 639, 55, 37, 458, 300, 540, 647, 51, 179, 151, 383, 515, 
 324, 502, 509, 267, 678, 177, 14, 859, 530, 630, 99, 145, 45, 380, 605, 389, 163, 638, 154, 548, 
 46, 652, 15, 90, 400, 851, 589, 783, 844, 702, 331, 525]

# SCANNET_LABELS = ['chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
# 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
# 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
# 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
# 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
# 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
# 'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
# 'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
# 'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
# 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
# 'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']

# SCANNET_IDS = [2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
# 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
# 155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
# 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191]

SCANNET_LABELS = ['sit down', 'be unpluged', 'have a bath', 'using a VCR', 'wake up in the morning', 'playing a guitar', 'balance to ride', 'grooming', 'get warm', 'water and sun', 'print', 'drink', 'buying food', 'make a phone call', 'bring suit', 'go on the internet', 'write', 'type', 'cook a curry', 'play the piano', 'playing soccer', 'eating breakfast in bed', 'paint a house', 'going on a vacation', 'a goldfish', 'washing your clothes', 'cleaning clothing', 'cleaning your room']

SCANNET_LABELS = ['sit down requires this term', 'this term do not desire be unpluged', 'have a bath requires this term', 'using a VCR requires this term', 'wake up in the morning requires this term', 'playing a guitar requires this term', 'This term requires balance to ride', 'grooming requires this term', 'get warm requires this term', 'this term requires water and sun', 'print requires this term', 'drink requires this term', 'buying food requires this term', 'make a phone call requires this term', 'bring suit requires this term', 'go on the internet requires this term', 'write requires this term', 'type requires this term', 'cook a curry requires this term', 'play the piano requires this term', 'playing soccer requires this term', 'eating breakfast in bed requires this term', 'paint a house requires this term', 'going on a vacation requires this term', 'a goldfish requires this term', 'washing your clothes requires this term', 'cleaning clothing requires this term', 'cleaning your room requires this term']
SCANNET_IDS = list(range(1,29))

SCANNETPP_LABELS = ['door', 'table', 'cabinet', 'ceiling lamp', 'curtain', 'chair', 'blinds', 'storage cabinet', 'bookshelf', 'office chair', 'window', 'whiteboard', 'ceiling light', 'monitor', 'shelf', 'object', 'window frame', 'pipe', 'structure', 'box', 'heater', 'kitchen cabinet', 'storage rack', 'sofa', 'bed', 'shower wall', 'doorframe', 'door frame', 'roof', 'wardrobe', 'pillar', 'plant', 'blanket', 'machine', 'windowsill', 'linked retractable seats', 'window sill', 'cardboard box', 'tv', 'books', 'desk', 'computer tower', 'kitchen counter', 'trash can', 'trash bin', 'jacket', 'electrical duct', 'blackboard', 'cable tray', 'air duct', 'sink', 'carpet', 'bag', 'counter', 'refrigerator', 'picture', 'pillow', 'cupboard', 'window blind', 'towel', 'beam', 'office table', 'stool', 'suitcase', 'backpack', 'bathtub', 'rug', 'keyboard', 'rack', 'gym mat', 'toilet', 'suspended ceiling', 'shower floor', 'clothes', 'pipe storage rack', 'air conditioner', 'fume hood', 'printer', 'blind', 'poster', 'experiment bench', 'electrical control panel', 'shower curtain', 'windowframe', 'book', 'ceiling beam', 'painting', 'paper', 'ladder', 'laboratory bench', 'bench', 'milling machine', 'microwave', 'partition', 'board', 'office cabinet', 'rolling cart', 'laboratory cabinet', 'crate', 'raised floor', 'electrical panel', 'mattress', 'bottle', 'pedestal fan', 'sofa chair', 'headboard', 'fridge', 'bucket', 'kitchen unit', 'beanbag', 'oven', 'cushion', 'power socket', 'office desk', 'whiteboards', 'lab equipment', 'shoes', 'work bench', 'file cabinet', 'mirror', 'basket', 'beverage crate', 'washing machine', 'shoe rack', 'hydraulic press', 'photocopy machine', 'telephone', 'lab machine', 'sliding door', 'tv stand', 'objects', 'couch', 'coat', 'open cabinet', 'scientific equipment', 'coffee table', 'garage door', 'bin', 'radiator', 'standing lamp', 'stove', 'roller blinds', 'fume cupboard', 'pc', 'stairs', 'medical appliance', 'closet', 'trolley', 'file folder', 'projector', 'cloth', 'conference table', 'cardboard', 'blind rail', 'dishwasher', 'room divider', 'copier', 'ventilation pipe', 'bathroom cabinet', 'laptop', 'electrical box', 'arm chair', 'bar counter', 'stage', 'ceiling ventilator', 'lounge chair', 'plant pot', 'bathroom stall', 'pinboard', 'comforter', '3d printer', 'steel beam', 'projector screen', 'electric duct', 'cart', 'air pipe', 'training equipment', 'floor mounted air conditioner', 'tile wall', 'glass wall', 'exhaust fan', 'vacuum cleaner', 'laundry basket', 'nightstand', 'armchair', 'drying rack', 'indoor crane', 'storage trolley', 'dresser', 'l-shaped sofa', 'coat hanger', 'dining chair', 'office visitor chair', 'interactive board', 'hose', 'light switch', 'shower ceiling', 'coffee machine', 'cables', 'floor lamp', 'fan', 'wire tray', 'compressor', 'laboratory equipment', 'dining table', 'speaker', 'climbing wall', 'light', 'paper towel dispenser', 'coat rack', 'table lamp', 'frame', 'duvet', 'fire extinguisher', 'range hood', 'high table', 'backdrop', 'ventilation duct', 'seat', 'tablecloth', 'electrical cabinet', 'ping pong table', 'bathroom floor', 'ceiling pipe', 'bedside table', 'coffee maker', 'computer desk', 'urinal', 'loft bed', 'air vent', 'chairs', 'bedsheet', 'television', 'lamp', 'rolling chair', 'wall cabinet', 'book shelf', 'brick wall', 'treadmill', 'vent', 'shirt', 'canopy bed', 'clothes hanger', 'kettle', 'shoe', 'high stool', 'tripod', 'bar stool', 'exhaust duct', 'wooden plank', 'squat rack', 'cubicle door', 'folding screen', 'kitchen sink', 'container', 'bottles', 'ottoman', 'bicycle', 'staircase railing', 'overhead projector', 'surfboard', 'folder', 'power strip', 'high bench', 'wall beam', 'pallet cage', 'interactive whiteboard', 'floor sofa', 'duct', 'flat panel display', 'wooden frame', 'folding room divider', 'cable', 'mug', 'rolling table', 'locker', 'standing banner', 'decoration', 'clothes drying rack', 'foosball table', 'standing poster', 'bath tub', 'yoga mat', 'microscope', 'paper bag', 'mouse', 'umbrella', 'medical machine', 'smoke detector', 'cup', 'cutting board', 'console', 'drum', 'bathroom counter', 'toilet paper', 'robot car', 'exhaust pipe', 'bath cabinet', 'whiteboard stand', 'notice board', 'paper towel', 'crates', 'bed frame', 'bathroom mat', 'shower partition', 'cloth hangers', 'clothes cabinet', 'tv screen', 'babyfoot table', 'rolling curtain', 'coat stand', 'kitchen towel', 'plank', 'side table', 'storage shelf', 'mat', 'shower', 'white board', 'information board', 'backsplash', 'guitar', 'cloth rack', 'ceiling vent', 'partition wall', 'kitchen shelf', 'banner', 'file binder', 'cleaning trolley', 'racing simulator', 'workbench', 'pot', 'ac system', 'power panel', 'desk lamp', 'broom', 'cpu', 'fitted wardrobe', 'tote bag', 'plumbing pipe', 'slippers', 'blackboard frame', 'magazine', 'hose pipe', 'rolled paper', 'sweater', 'clock', 'tray', 'desk fan', 'vaccum cleaner', 'projection curtain', 'freezer display counter', 'pan', 'vase', 'glass', 'folders', 'tap', 'wall lamp', 'plate', 'laptop stand', 'small cabinet', 'file organizer', 'clothes dryer', 'wall painting', 'curtain rail', 'wheelchair', 'bottle crate', 'sheet', 'folding sofa', 'shower pan', 'plastic case', 'christmas tree', 'piano', 'ottoman chair', 'jar', 'foldable closet', 'notebook', 'calendar', 'janitor cart', 'storage box', 'rolling blinds', 'bathroom shelf', 'soap dispenser', 'binder', 'copy machine', 'rice cooker', 'gym mattress', 'car door', 'table football', 'bowl', 'light panel', 'tissue box', 'bedframe', 'wall hanging', 'jug', 'skylight', 'ceiling fan', 'dish rack', 'shelving cart', 'instant pot', 'whiteboard eraser', 'floor mat', 'socket', 'mini fridge', 'wall clock', 'boots', 'barbecue grill', 'paper shredder', 'file rack', 'floor scrubber', 'metal board', 'water heater', 'tool rack', 'recliner', 'barber chair', 'ventilator', 'trolley table', 'standing fan', 'water filter', 'shoes holder', 'vr setup', 'trashcan', 'bike', 'lab materials', 'wooden pallet', 'dustbin', 'curtain rod', 'reflection', 'toilet brush', 'exercise ball', 'air purifier', 'kitchen back splash', 'paper rack', 'toolbox', 'monitor cover', 'file', 'surfsuit', 'night stand', 'paper organizer', 'serving trolley', 'phone', 'canvas', 'camping bed', 'tower pc', 'cylinder', 'magazine stand', 'toy', 'slipper', 'air conditioning', 'hanger', 'vertical blinds', 'desk organizer', 'guitar bag', 'spray bottle', 'suit cover', 'toaster', 'spotlight', 'machine container', 'foot rest', 'shopping trolley', 'decoration piece', 'control panel', 'multifunction printer', 'jerry can', 'window head', 'cooker hood', 'basin', 'panel', 'papasan chair', 'tv mount', 'toilet seat', 'shopping bag', 'photocopier', 'tube', 'studio light', 'stuffed toy', 'cord cover', 'power cabinet', 'filer organizer', 'garage shelf', 'luggage', 'gym bag', 'exhaust hood', 'microwave oven', 'floor cushion', 'easy chair', 'bar table', 'shoe cabinet', 'paper tray', 'lab coat', 'toilet paper dispenser', 'kitchen storage rack', 'equipment', 'computer table', 'mouse pad', 'drawer', 'headphones', 'bathroom sink', 'outlet', 'toaster oven', 'tv table', 'bedside cabinet', 'rolling trolley', 'step stool', 'trousers', 'bathroom rack', 'shelf trolley', 'glass shelf', 'fabric', 'lab fridge', 'work station', 'barrel', 'mop', 'deck chair', 'bath counter', 'helmet', 'standing clothes hanger', 'garbage bin', 'study table', 'air fryer', 'plastic bag', 'oven range', 'headphone', 'kitchen counter top', 'clothes rack', 'wall unit', 'grab bar', 'flipchart', 'scarf', 'labcoat', 'hat', 'bedside lamp', 'sewing machine table', 'shower head', 'switchboard cabinet', 'flip paper', 'storage container', 'canister', 'wall board', 'shower rug', 'plastic box', 'stovetop', 'information stand', 'footstool', 'pack', 'push cart', 'table cloth', 'celing lamp', 'cupoard', 'jeans', 'smoke alarm', 'bath mat', 'softbox', 'whiteboard mount', 'paper cutter', 'cable raceway', 'water kettle', 'pelican case', 'towel rack', 'rolling shelf cart', 'built-in shelf', 'equipment cover', 'television stand', 'sheets', 'small dresser', 'light stand', 'beach umbrella', 'faucet', 'bagpack', 'dumbbell', 'water dispenser', 'medicine cabinet', 'tv console', 'mirror frame', 'chandelier', 'pen holder', 'messenger bag', 'ball', 'glass bottle', 'softbox light', 'gym ball', 'briefcase', 'plastic bottle', 'monitor stand', 'human skeleton', 'podium', 'wall strip', 'tablet', 'bedside shelf', 'headrail', 'sink counter', 'doormat', 'baseboard', 'bulletin board', 'electric hob', 'bean bag', 'high pressure cylinder', 'portable fan', 'flush button', 'wooden post', 'lectern', 'curtain frame', 'computer monitor', 'folding chair', 'tabletop', 'led ceiling fan', 'high chair', 'grill', 'metal rack', 'air conditioner tower', 'sliding door frame', 'cable rack', 'bench press', 'ironing board', 'wooden palette', 'kitchenware', 'blind rails', 'plastic container', 'weighing scale', 'headset', 'tree trunk', 'shower screen', 'wall shelf', 'watering can', 'tool box', 'bed  sheet', 'glass pane', 'tower fan', 'switch', 'notice', 'sack', 'table mat', 'flower pot', 'dog bed', 'laundry hanger', 'mobile tv stand', 'file holder', 'floor couch', 'tv trolley', 'chopping board', 'centrifuge', 'tubelight', 'bedpost', 'step', 'center table', 'upholstered bench', 'sink pipe', 'door mat', 'storage bin', 'towel radiator', 'shower tray', 'electronic appliance', 'boiler', 'food container', 'cable pathway', 'carboard box', 'metal sheet', 'hand bag', 'sign', 'laundry rack', 'screen', 'cardbox', 'fireplace surround', 'boot', 'envelope', 'carton', 'tool organizer', 'paper roll', 'water bottle', 'shoe changing stool', 'balcony door', 'espresso machine', 'water pipe', 'recesssed shelf', 'drum set', 'skiboard', 'speaker stand', 'kitchen wall', 'suit', 'photo', 'globe', 'spice rack', 'delivery bag', 'router', 'rolling blind', 'easel', 'shower cubicle', 'dish drainer', 'doorway', 'folded table', 'pants', 'computer', 'stuffed animal', 'office  chair', 'cable conduit', 'picture frame', 'shoe stool', 'recessed shelve', 'toilet paper holder', 'panelboard', 'stapler', 'skateboard', 'workshop tool', 'projector holder', 'flag', 'chemical canister', 'web cam', 'hoodie', 'towel heater', 'towel warmer', 'shower curtain rod', 'shower faucet', 'shower door', 'laboratory power supply', 'tool', 'ventilation', 'soap bottle', 'bathrobe', 'pictures board', 'cap', 'woofer', 'tshirt', 'rolling bag', 'shoe box', 'luggage bag', 'file storage', 'cat bed', 'stack of paper', 'surfing board', 'electric kettle', 'rolling stand', 'cover', 'main switchboard', 'pressure cooker', 'stepladder', 'countertop', 'flip flops', 'short table', 'sit-up pillow', 'duffel bag', 'shower seating', 'washbasin', 'teddy bear', 'stair', 'plate rack', 'ornament', 'jerrycan', 'filter jug', 't shirt', 'cooking pot', 'platform trolley', 'blinds rod', 'hand shower', 'power socket unit', 'sheep doll', 'laptop bag', 'game console', 'bottles case', 'lid', 'dumbbell case', 'rolled blanket', 'paper stapler', 'kitchen pot', 'charcoal bag', 'laundry hamper', 'rolled poster', 'bath towel', 'apron', 'dustpan', 'trash bag', 'document tray', 'camera', 'mirror cabinet', 'dish drying rack', 'gas tank', 'cable roller', 'case', 'ring light', 'hair dryer', 'gym plate', 'hand towel', 'sill', 'sidetable', 'vice', 'bench stool', 'billboard', 'rolling cabinet', 'shower sink', 'cloth piece', 'oscilloscope', 'magazine rack', 'wash basin', 'cable panel', 'photo frame', 'tv receiver', 'stand', 'milk jug', 'wooden board', 'bladeless fan', 'door  frame', 'wall paper', 'scale', 'purse', 'electronic device', 'sofa cushion', 'sponge', 'dish washer', 'crate trolley', 'kitchen hood', 'laundry vent', 'medical stool', 'exhaustive fan', 'portable ladder', 'chemical container', 'toilet paper rolls', 'rag', 'blender', 'window pane', 'dog bowl', 'shopping basket', 'piano stool', 'electric box', 'wall calendar', 'paper holder', 'chemical bottle', 'sandals', 'foreman grill', 'guitar case', 'heater tube', 'running shoes', 'shower tap', 'cloth hanger', 'microphone', 'cabinet frame', 'decorative object', 'light fixture', 'ceiling lamp bar', 'paperbag', 'chemical barrel', 'wicker basket', 'exit sign', 'bottles rack', 'water jug', 'bottle carrier', 'laboratory pellet press', 'mini oven', 'shower arm', 'paper tube', 'suitcase stand', 'table fan', 'shelve', 'full-length mirror', 'wood piece', 'can', 'suit bag', 'water bubbler', 'first aid kit', 'kitchen robot', 'toilet flush button', 'pillow toy', 'plush doll', 'styrofoam box', 'document organizer', 'pet carrier', 'folding table', 'gloves', 'pitcher', 'cable spool', 'rolled cable', 'folding umbrella', 'robot vacuum cleaner', 'tower ventilator', 'brush', 'planter', 'baseball cap', 'gas cylinder', 'stereo', 'baby stroller', 'water bucket', 'rucksack', 'shower door frame', 'drone', 'kitchen cloth', 'hand soap dispenser', 'pegboard', 'alarm', 'emergency light', 'sign board', 'weight plate', 'rolled projection screen', 'laptop table', 'hole puncher', 'mixer', 'piano chair', 'paper bin', 'wooden stick', 'fireplace', 'rolled backdrop', 'mousepad', 'long pillow', 'bananas', 'column', 'cd player', 'eraser', 'laptop sleeve', 'hairdryer', 'seat cushion', 'plant pot mat', 'wastebin', 'wood panel', 'detergent bottle', 'safe box', 'pouch', 'blind rod', 'mop basin', 'plug', 'document holder', 'railing', 'plastic drum', 'cat tree', 'mirror light', 'kettlebell', 'chart', 'dust pan', 'sandal', 'first aid cabinet', 'bracket', 'wire', 'scooter', 'racing wheel', 'wine rack', 'belt', 'tissue dispenser stand', 'towel paper dispenser', 'air heater', 'plushie', 'knife set', 'iron', 'intercom', 'kitchen ceiling', 'package', 'toilet paper dispensor', 'sneakers', 'umbrella stand', 'egg carton', 'organizer', 'stick', 'shampoo bottle', 'cone', 'file tray', 'wooden plan', 'cooking pan', 'brief', 'paper towel package', 'glove dispenser', 'dispenser bottle', 'kitchen drawer', 'remote control', 'powerstrip', 'emergency shower', 'scanner', 'towel holder', 'vr headset', 'watering pot', 'soda machine', 'pole stand', 'roomba', 'rolling mat', 'fluorescent lamp', 'flower', 'pc tower', 'metal mount', 'oven gloves', 'soap', 'flush tank', 'notepad', 'pull up bar', 'loafers', 'water meter cover', 'plastic tray', 'webcam', 'barbell', 'tea pot', 'wall hanger', 'cabinet base panel', 'laptop case', 'paper towel holder', 'skeleton', 'socket extender', 'extension chord reel', 'wooden crate', 'guillotine paper cutter', 'sewing machine', 'model car', 'bed cover', 'storage', 'freezer', 'piano book', 'wifi router', 'overhead shower', 'chrismas tree', 'drill', 'clothes drying stand', 'dumbell', 'cabel', 'badminton racket', 'cool box', 'thermostat', 'dartboard', 'switchboard cover', 'candle', 'electric stove', 'paper ram', 'insulated can', 'prosthetic leg', 'desk power strip', 'closet rail', 'water pitcher', 'plate weights', 'extension cord', 'tea box', 'tissue', 'sauce pan', 'toothbrush', 'frying pan', 'package of paper', 'microphone stand', 'socket box', 'leather mattress', 'plastic can', 'garbage bin cover', 'plush toy', 'electric guitar', 'weight scale', 'fruit', 'tape dispenser', 'pallet', 'emergency kit', 'bathroom mirror', 'door lamp', 'power extension', 'electric circuit board', 'oven panel', 'electrical pipe', 'shade', 'photoframe', 'file stack', 'pizza box', 'tennis racket', 'wall hook', 'recycle bag', 'recessed shower shelve', 'wall mounted telephone', 'facsimile', 'kitchen roll', 'totebag', 'floor cleaner', 'body weight scale', 'stuffed animal door insulator', 'neck pillow', 'basketball', 'cable wheel', 'door vent', 'foot massager', 'pumper', 'hanging light fixture', 'interphone', 'paper towel roll', 'control unit', 'folder oragnizer', 'hanging frame', 'paper stack', 'door handle', 'mini shelf', 'beverage carton', 'wok pan', 'bedside counter', 'bar', 'pot lid', 'radio', 'paper box', 'stabilizer', 'headphone case', 'folded cardboard box', 'coaster', 'pen tray', 'marker', 'shower mat', 'rod', 'device', 'electric pot', 'mixer machine', 'barstool', 'bread toaster', 'electric mixer', 'wall cord cover', 'camera bag', 'charger', 'knife', 'metal frame', 'folded bag', 'conduit pipe', 'french press', 'cabinet side panel', 'cosmetic bag', 'surveillance camera', 'bathroom holder', 'footrest', 'glasses case', 'handbag', 'cooling pad', 'flip flop', 'game controller', 'packet of toilet paper', 'cable duct', 'toilet paper roll', 'monitor support', 'fire alarm', 'kitchen stove', 'key hanger', 'tub', 'network socket', 'ceilng light', 'christmas ornament', 'pepper mill', 'wall outlet', 'light cover', 'caution board', 'heels', 'package bag', 'blinds rail', 'candle holder', 'electric toothbrush', 'tool case', 'toilet flush', 'wooden brush', 'hand washing soap', 'hygiene product', 'water tap', 'cosmetic pouch', 'mount', 'knife holder', 'dustpan and brush', 'circular tray', 'shoe case', 'ar tag', 'flipflop', 'sculpture', 'recycle bin', 'kitchen utensil', 'multiplug', 'beaker stand', 'chessboard', 'pen', 'toothpaste', 'tv remote', 'floor wiper', 'wire hider', 'water meter', 'magazine holder', 'mailbox', 'paper file', 'wall coat hanger', 'utensil holder', 'detergent', 'safe', 'packet', 'dvd', 'dvd player', 'tupperware', 'electrical board', 'radiator pipe', 'plat', 'decorative mirror', 'telephone stand', 'water boiler', 'cutboard', 'hanging deer skull', 'file orginizer', 'satchel', 'head model', 'parcel', 'dish soap bottle', 'glass plate', 'pencil case', 'cleaning mop', 'mixer glass', 'joystick', 'vacuum flask', 'bag of oranges', 'pencil holder', 'pamphlet', 'rope', 'cd', 'knife stand', 'pencil cup', 'binding machine', 'grill pan', 'tape', 'cabinet top panel', 'cutti̇ng board', 'salad spinner', 'water filter jug', 'mop cloth', 'shower valve', 'laptop cover', 'running shoe', 'alligator clips', 'insulated coffee mug', 'pencil stand', 'action figure', 'desk light', 'midi controller', 'bathroom slippers', 'elephant decoration piece', 'switchboard', 'remote', 'arch folder', 'power cord', 'hot bag', 'pencils cup', 'knife block', 'thermos', 'power switch', 'toothbrush holder', 'cleaning liquid', 'power board', 'pull-up bar', 'sandwich maker', 'statue', 'brief case', 'airdyer', 'bike helmet', 'mirror lamp', 'tennis rackets', 'paint jar', 'citrus juicer', 'scissors', 'hanging hook', 'napkin', 'laundry detergent', 'disinfectant dispenser', 'cleaning brush', 'food', 'pan set', 'kitchen appliance', 'mop pad', 'dish soap', 'plunger', 'paper package', 'plastic mat', 'calculator', 'voltage stabilizer', 'whiteboard marker', 'wall coat rack', 'milk carton', 'cosmetic bottle', 'shower handle', 'tissue paper', 'shorts', 'game console controller', 'remote controller', 'teapot', 'drain', 'juice box', 'phone stand', 'soap container', 'flush', 'rubber water bag', 'shower gel', 'snack bag', 'watering bucket', 'first aid box', 'banana', 'paper notebook', 'hammer', 'kitchen glove', 'handle', 'personal hygiene product', 'duster', 'robot vaccuum cleaner', 'chain', 'soap dish', 'cereal box', 'door window', 'elephant toy', 'frisbee', 'cream tube', 'folder holder', 'internet socket', 'moka pot', 'stereo box', 'face mask', 'bunny chocolate', 'magazine collector', 'shower holder', 'toilet  brush', 'oven glove', 'shower drain', 'penholder', 'switch board', 'coffee pot', 'marker eraser', 'squeegee', 'spray', 'game cd box', 'table clock', 'apple', 'spool', 'portable speaker', 'guitar pedal', 'bread', 'glass jar', 'paper puncher', 'dishwashing sponge', 'shower hose', 'surface cleaning liquid', 'wallet', 'circuit box', 'toiletry', 'hair brush', 'headphone bag', 'metal saw', 'power adapter', 'wlan router', 'plant pot coaster', 'oven mitt', 'hanging light', 'kitchen object', 'machine button', 'vanity light', 'flush plate', 'leaf fan', 'punching machine', 'window handle', 'wine bottle', 'tennis cap', 'chips can', 'food pot', 'hand washing soap dispenser', 'wall light', 'detergent bag', 'glasses cover', 'hangbag', 'bottle spray', 'kitchen tap', 'knob', 'figurine', 'hand vacuum', 'potted plant', 'marker storage', 'mixing bowl', 'power supply', 'intercom screen', 'coffee mug', 'drilling machine', 'electric socket', 'hand shower handle', 'hole punch', 'silicon gun', 'document  holder', 'wooden chest', 'cheese', 'headphones case', 'measuring spoon', 'monitor light', 'ear muffs', 'night lamp', 'wall handle', 'intercom device', 'lanyard', 'rugby ball', 'shower loofah', 'cable socket', 'cleaning cloth', 'dish washer soap', 'power brick', 'coffee', 'monitor base', 'mushroom lamp', 'trivet', 'surface cleaner', 'toiletry bottle', 'paper note', 'strainer', 'colander', 'kitchen brush', 'multi socket', 'saucer', 'scissor', 'stamp', 'emergency button', 'toilet brush holder', 'coffee jar', 'deodorant', 'shower rod', 'wet tissue', 'tissue paper roll', 'funnel', 'oragnizer', 'toilet plunger', 'tumbler', 'door knob', 'coarser', 'postit note', 'punching tool', 'beverage can', 'dish', 'star', 'flask', 'in-table power socket', 'laptop charger', 'wall intercom', 'flour bag', 'blowtorch', 'cleaner', 'takeout box', 'hand soap', 'hard drive', 'monitor holder', 'soldering iron', 'control switch', 'drainage', 'box lid', 'playstation controller', 'shaving foam', 'guitar stand', 'toilet cleaner', 'co detector', 'covered power socket', 'spray can', 'tooth brush', 'phone tripod', 'bluetooth speaker', 'dish washing liquid', 'powerbank', 'dried plant', 'shampoo', 'sticky note', 'toilet seat brush', 'dishwashing soap', 'lan port', 'scrubber', 'cord', 'ashtray', 'handle bar', 'dongle', 'liquid soap', 'mobile phone', 'bunny decoration', 'palm rest', 'stream deck', 'blinds chain', 'dishwashing liquid', 'hammer holder', 'hand washing liquid', 'shower wiper', 'duct tape', 'valve', 'glove', 'stationery', 'ceiling speaker', 'holder', 'power plug', 'wall speaker', 'card', 'electrical tape', 'pot cover', 'razor', 'barcode scanner', 'lotion', 'grab rail', 'hairbrush', 'lint roller', 'post-it', 'dusting cloth', 'glue bottle', 'handwash', 'network outlet', 'card reader', 'kitchen light', 'pc charger', 'product dispenser bottle', 'hair dryer holder', 'screwdriver', 'shower squeegee', 'juice tetrapack', 'pliers', 'sunblock', 'bed sheet', 'hook', 'headphone holder', 'light bulb', 'ladle', 'bar soap', 'button', 'dish brush', 'toy car', 'note', 'post it', 'sponge cloth', 'wood stick', 'paper weight', 'product tube', 'cell phone', 'smartphone', 'comb', 'jump rope', 'lan', 'wireless charger', 'door stopper', 'cigarette packet', 'pencil', 'soap holder', 'electrical plug', 'toilet holder', 'mask', 'pocket calculator', 'cabbage', 'spoon', 'doorknob', 'fan switch', 'rubber duck', 'backdrop hook', 'table tennis racket', 'receipt', 'soap bar', 'tape roll', 'tooth paste', 'electrical adapter', 'probe', 'highlighter', 'correction fluid', 'dispenser', 'pi̇cture', 'door hinge', 'whiteboard marker holder', 'botttle', 'power outlet', 'towel hanger', 'whiteboard duster', 'magnet', 'sticker', 'nose spray', 'vertical blind control', 'razor blade', 'post it note', 'wireless headphones', 'cabinet top', 'cream bottle', 'datashow socket', 'earbuds', 'cabinet door', 'chair cushion', 'cosmetic tube']

SCANNETPP_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 495, 496, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 575, 576, 577, 578, 580, 581, 583, 584, 585, 586, 588, 589, 591, 592, 593, 594, 595, 596, 597, 598, 602, 603, 604, 605, 606, 607, 608, 609, 611, 612, 613, 614, 615, 616, 617, 618, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 653, 654, 655, 656, 657, 658, 659, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 705, 706, 709, 710, 711, 712, 713, 714, 716, 717, 718, 719, 720, 721, 722, 723, 724, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 919, 920, 921, 922, 923, 924, 925, 926, 928, 929, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 943, 944, 946, 947, 948, 949, 950, 951, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 977, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1080, 1081, 1082, 1083, 1085, 1086, 1087, 1089, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1107, 1108, 1109, 1110, 1111, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1125, 1126, 1127, 1128, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1279, 1280, 1281, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1413, 1414, 1415, 1417, 1418, 1419, 1420, 1421, 1423, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1535, 1536, 1537, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1565, 1566, 1567, 1568, 1570, 1571, 1572, 1573, 1574, 1576, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1589, 1590, 1591, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1641, 1642, 1643, 1644, 1645, 1646, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658]