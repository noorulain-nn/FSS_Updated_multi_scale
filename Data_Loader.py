import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, Omniglot, Caltech101
# from data.fss_dataset.pascal5i_reader import Pascal5iReader
import numpy as np
import random
from collections import defaultdict

# Module-level class for pickling support (required for Windows multiprocessing)
class RemappedSubset(torch.utils.data.Dataset):
	"""Custom dataset class that maps old labels to new sequential labels"""
	def __init__(self, dataset, indices):
		self.dataset = dataset
		self.indices = indices
	
	def __len__(self):
		return len(self.indices)
	
	def __getitem__(self, idx):
		original_idx, new_label = self.indices[idx]
		image, _ = self.dataset[original_idx]
		return image, new_label

NAMES = [
    "tench", "Tinca tinca",
    "goldfish", "Carassius auratus",
    "great white shark", "white shark", "man-eater", "man-eating shark", "Carcharodon caharias",
    "tiger shark", "Galeocerdo cuvieri",
    "hammerhead", "hammerhead shark",
    "electric ray", "crampfish", "numbfish", "torpedo",
    "stingray",
    "cock",
    "hen",
    "ostrich", "Struthio camelus",
    "brambling", "Fringilla montifringilla",
    "goldfinch", "Carduelis carduelis",
    "house finch", "linnet", "Carpodacus mexicanus",
    "junco", "snowbird",
    "indigo bunting", "indigo finch", "indigo bird", "Passerina cyanea",
    "robin", "American robin", "Turdus migratorius",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel", "dipper",
    "kite",
    "bald eagle", "American eagle", "Haliaeetus leucocephalus",
    "vulture",
    "great grey owl", "great gray owl", "Strix nebulosa",
    "European fire salamander", "Salamandra salamandra",
    "common newt", "Triturus vulgaris",
    "eft",
    "spotted salamander", "Ambystoma maculatum",
    "axolotl", "mud puppy", "Ambystoma mexicanum",
    "bullfrog", "Rana catesbeiana",
    "tree frog", "tree-frog",
    "tailed frog", "bell toad", "ribbed toad", "tailed toad", "Ascaphus trui",
    "loggerhead", "loggerhead turtle", "Caretta caretta",
    "leatherback turtle", "leatherback", "leathery turtle", "Dermochelys coriacea",
    "mud turtle",
    "terrapin",
    "box turtle", "box tortoise",
    "banded gecko",
    "common iguana", "iguana", "Iguana iguana",
    "American chameleon", "anole", "Anolis carolinensis",
    "whiptail", "whiptail lizard",
    "agama",
    "frilled lizard", "Chlamydosaurus kingi",
    "alligator lizard",
    "Gila monster", "Heloderma suspectum",
    "green lizard", "Lacerta viridis",
    "African chameleon", "Chamaeleo chamaeleon",
    "Komodo dragon", "Komodo lizard", "dragon lizard", "giant lizard", "Varanus komodoeis'", 
    "African crocodile", "Nile crocodile", "Crocodylus niloticus",
    "American alligator", "Alligator mississipiensis",
    "triceratops",
    "thunder snake", "worm snake", "Carphophis amoenus",
    "ringneck snake", "ring-necked snake", "ring snake",
    "hognose snake", "puff adder", "sand viper",
    "green snake", "grass snake",
    "king snake", "kingsnake",
    "garter snake", "grass snake",
    "water snake",
    "vine snake",
    "night snake", "Hypsiglena torquata",
    "boa constrictor", "Constrictor constrictor",
    "rock python", "rock snake", "Python sebae",
    "Indian cobra", "Naja naja",
    "green mamba",
    "sea snake",
    "horned viper", "cerastes", "sand viper", "horned asp", "Cerastes cornutus",
    "diamondback", "diamondback rattlesnake", "Crotalus adamanteus",
    "sidewinder", "horned rattlesnake", "Crotalus cerastes",
    "trilobite",
    "harvestman", "daddy longlegs", "Phalangium opilio",
    "scorpion",
    "black and gold garden spider", "Argiope aurantia",
    "barn spider", "Araneus cavaticus",
    "garden spider", "Aranea diademata",
    "black widow", "Latrodectus mactans",
    "tarantula",
    "wolf spider", "hunting spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse", "partridge", "Bonasa umbellus",
    "prairie chicken", "prairie grouse", "prairie fowl",
    "peacock",
    "quail",
    "partridge",
    "African grey", "African gray", "Psittacus erithacus",
    "macaw",
    "sulphur-crested cockatoo", "Kakatoe galerita", "Cacatua galerita",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser", "Mergus serrator",
    "goose",
    "black swan", "Cygnus atratus",
    "tusker",
    "echidna", "spiny anteater", "anteater",
    "platypus", "duckbill", "duckbilled platypus", "duck-billed platypus", "Ornithorhyhus anatinus'",
    "wallaby", "brush kangaroo",
    "koala", "koala bear", "kangaroo bear", "native bear", "Phascolarctos cinereus",
    "wombat",
    "jellyfish",
    "sea anemone", "anemone",
    "brain coral",
    "flatworm", "platyhelminth",
    "nematode", "nematode worm", "roundworm",
    "conch",
    "snail",
    "slug",
    "sea slug", "nudibranch",
    "chiton", "coat-of-mail shell", "sea cradle", "polyplacophore",
    "chambered nautilus", "pearly nautilus", "nautilus",
    "Dungeness crab", "Cancer magister",
    "rock crab", "Cancer irroratus",
    "fiddler crab",
    "king crab", "Alaska crab", "Alaskan king crab", "Alaska king crab", "Paralithodesamtschatica'",
    "American lobster", "Northern lobster", "Maine lobster", "Homarus americanus",
    "spiny lobster", "langouste", "rock lobster", "crawfish", "crayfish", "sea crawfish",
    "crayfish", "crawfish", "crawdad", "crawdaddy",
    "hermit crab",
    "isopod",
    "white stork", "Ciconia ciconia",
    "black stork", "Ciconia nigra",
    "spoonbill",
    "flamingo",
    "little blue heron", "Egretta caerulea",
    "American egret", "great white heron", "Egretta albus",
    "bittern",
    "crane", "bird",
    "limpkin", "Aramus pictus",
    "European gallinule", "Porphyrio porphyrio",
    "American coot", "marsh hen", "mud hen", "water hen", "Fulica americana",
    "bustard",
    "ruddy turnstone", "Arenaria interpres",
    "red-backed sandpiper", "dunlin", "Erolia alpina",
    "redshank", "Tringa totanus",
    "dowitcher",
    "oystercatcher", "oyster catcher",
    "pelican",
    "king penguin", "Aptenodytes patagonica",
    "albatross", "mollymawk",
    "grey whale", "gray whale", "devilfish", "Eschrichtius gibbosus", "Eschrichtius rostus'",
    "killer whale", "killer", "orca", "grampus", "sea wolf", "Orcinus orca",
    "dugong", "Dugong dugon",
    "sea lion",
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog", "Maltese terrier", "Maltese",
    "Pekinese", "Pekingese", "Peke",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound", "Afghan",
    "basset", "basset hound",
    "beagle",
    "bloodhound", "sleuthhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound", "Walker foxhound",
    "English foxhound",
    "redbone",
    "borzoi", "Russian wolfhound",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound", "Ibizan Podenco",
    "Norwegian elkhound", "elkhound",
    "otterhound", "otter hound",
    "Saluki", "gazelle hound",
    "Scottish deerhound", "deerhound",
    "Weimaraner",
    "Staffordshire bullterrier", "Staffordshire bull terrier",
    "American Staffordshire terrier", "Staffordshire terrier", "American pit bull rrier", "pit bull terrier'",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier", "Sealyham",
    "Airedale", "Airedale terrier",
    "cairn", "cairn terrier",
    "Australian terrier",
    "Dandie Dinmont", "Dandie Dinmont terrier",
    "Boston bull", "Boston terrier",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier", "Scottish terrier", "Scottie",
    "Tibetan terrier", "chrysanthemum dog",
    "silky terrier", "Sydney silky",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa", "Lhasa apso",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla", "Hungarian pointer",
    "English setter",
    "Irish setter", "red setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber", "clumber spaniel",
    "English springer", "English springer spaniel",
    "Welsh springer spaniel",
    "cocker spaniel", "English cocker spaniel", "cocker",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog", "bobtail",
    "Shetland sheepdog", "Shetland sheep dog", "Shetland",
    "collie",
    "Border collie",
    "Bouvier des Flandres", "Bouviers des Flandres",
    "Rottweiler",
    "German shepherd", "German shepherd dog", "German police dog", "alsatian",
    "Doberman", "Doberman pinscher",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard", "St Bernard",
    "Eskimo dog", "husky",
    "malamute", "malemute", "Alaskan malamute",
    "Siberian husky",
    "dalmatian", "coach dog", "carriage dog",
    "affenpinscher", "monkey pinscher", "monkey dog",
    "basenji",
    "pug", "pug-dog",
    "Leonberg",
    "Newfoundland", "Newfoundland dog",
    "Great Pyrenees",
    "Samoyed", "Samoyede",
    "Pomeranian",
    "chow", "chow chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke", "Pembroke Welsh corgi",
    "Cardigan", "Cardigan Welsh corgi",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "timber wolf", "grey wolf", "gray wolf", "Canis lupus",
    "white wolf", "Arctic wolf", "Canis lupus tundrarum",
    "red wolf", "maned wolf", "Canis rufus", "Canis niger",
    "coyote", "prairie wolf", "brush wolf", "Canis latrans",
    "dingo", "warrigal", "warragal", "Canis dingo",
    "dhole", "Cuon alpinus",
    "African hunting dog", "hyena dog", "Cape hunting dog", "Lycaon pictus",
    "hyena", "hyaena",
    "red fox", "Vulpes vulpes",
    "kit fox", "Vulpes macrotis",
    "Arctic fox", "white fox", "Alopex lagopus",
    "grey fox", "gray fox", "Urocyon cinereoargenteus",
    "tabby", "tabby cat",
    "tiger cat",
    "Persian cat",
    "Siamese cat", "Siamese",
    "Egyptian cat",
    "cougar", "puma", "catamount", "mountain lion", "painter", "panther", "Felis concolor",
    "lynx", "catamount",
    "leopard", "Panthera pardus",
    "snow leopard", "ounce", "Panthera uncia",
    "jaguar", "panther", "Panthera onca", "Felis onca",
    "lion", "king of beasts", "Panthera leo",
    "tiger", "Panthera tigris",
    "cheetah", "chetah", "Acinonyx jubatus",
    "brown bear", "bruin", "Ursus arctos",
    "American black bear", "black bear", "Ursus americanus", "Euarctos americanus",
    "ice bear", "polar bear", "Ursus Maritimus", "Thalarctos maritimus",
    "sloth bear", "Melursus ursinus", "Ursus ursinus",
    "mongoose",
    "meerkat", "mierkat",
    "tiger beetle",
    "ladybug", "ladybeetle", "lady beetle", "ladybird", "ladybird beetle",
    "ground beetle", "carabid beetle",
    "long-horned beetle", "longicorn", "longicorn beetle",
    "leaf beetle", "chrysomelid",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant", "emmet", "pismire",
    "grasshopper", "hopper",
    "cricket",
    "walking stick", "walkingstick", "stick insect",
    "cockroach", "roach",
    "mantis", "mantid",
    "cicada", "cicala",
    "leafhopper",
    "lacewing", "lacewing fly",
    "dragonfly", "darning needle", "devil's darning needle", "sewing needle", "snake fder", "snake doctor", "mosquito hawk", "skeeter hawk",
    "damselfly",
    "admiral",
    "ringlet", "ringlet butterfly",
    "monarch", "monarch butterfly", "milkweed butterfly", "Danaus plexippus",
    "cabbage butterfly",
    "sulphur butterfly", "sulfur butterfly",
    "lycaenid", "lycaenid butterfly",
    "starfish", "sea star",
    "sea urchin",
    "sea cucumber", "holothurian",
    "wood rabbit", "cottontail", "cottontail rabbit",
    "hare",
    "Angora", "Angora rabbit",
    "hamster",
    "porcupine", "hedgehog",
    "fox squirrel", "eastern fox squirrel", "Sciurus niger",
    "marmot",
    "beaver",
    "guinea pig", "Cavia cobaya",
    "sorrel",
    "zebra",
    "hog", "pig", "grunter", "squealer", "Sus scrofa",
    "wild boar", "boar", "Sus scrofa",
    "warthog",
    "hippopotamus", "hippo", "river horse", "Hippopotamus amphibius",
    "ox",
    "water buffalo", "water ox", "Asiatic buffalo", "Bubalus bubalis",
    "bison",
    "ram", "tup",
    "bighorn", "bighorn sheep", "cimarron", "Rocky Mountain bighorn", "Rocky Mountain eep", "Ovis canadensis'",
    "ibex", "Capra ibex",
    "hartebeest",
    "impala", "Aepyceros melampus",
    "gazelle",
    "Arabian camel", "dromedary", "Camelus dromedarius",
    "llama",
    "weasel",
    "mink",
    "polecat", "fitch", "foulmart", "foumart", "Mustela putorius",
    "black-footed ferret", "ferret", "Mustela nigripes",
    "otter",
    "skunk", "polecat", "wood pussy",
    "badger",
    "armadillo",
    "three-toed sloth", "ai", "Bradypus tridactylus",
    "orangutan", "orang", "orangutang", "Pongo pygmaeus",
    "gorilla", "Gorilla gorilla",
    "chimpanzee", "chimp", "Pan troglodytes",
    "gibbon", "Hylobates lar",
    "siamang", "Hylobates syndactylus", "Symphalangus syndactylus",
    "guenon", "guenon monkey",
    "patas", "hussar monkey", "Erythrocebus patas",
    "baboon",
    "macaque",
    "langur",
    "colobus", "colobus monkey",
    "proboscis monkey", "Nasalis larvatus",
    "marmoset",
    "capuchin", "ringtail", "Cebus capucinus",
    "howler monkey", "howler",
    "titi", "titi monkey",
    "spider monkey", "Ateles geoffroyi",
    "squirrel monkey", "Saimiri sciureus",
    "Madagascar cat", "ring-tailed lemur", "Lemur catta",
    "indri", "indris", "Indri indri", "Indri brevicaudatus",
    "Indian elephant", "Elephas maximus",
    "African elephant", "Loxodonta africana",
    "lesser panda", "red panda", "panda", "bear cat", "cat bear", "Ailurus fulgens",
    "giant panda", "panda", "panda bear", "coon bear", "Ailuropoda melanoleuca",
    "barracouta", "snoek",
    "eel",
    "coho", "cohoe", "coho salmon", "blue jack", "silver salmon", "Oncorhynchus kisutch",
    "rock beauty", "Holocanthus tricolor",
    "anemone fish",
    "sturgeon",
    "gar", "garfish", "garpike", "billfish", "Lepisosteus osseus",
    "lionfish",
    "puffer", "pufferfish", "blowfish", "globefish",
    "abacus",
    "abaya",
    "academic gown", "academic robe", "judge's robe",
    "accordion", "piano accordion", "squeeze box",
    "acoustic guitar",
    "aircraft carrier", "carrier", "flattop", "attack aircraft carrier",
    "airliner",
    "airship", "dirigible",
    "altar",
    "ambulance",
    "amphibian", "amphibious vehicle",
    "analog clock",
    "apiary", "bee house",
    "apron",
    "ashcan", "trash can", "garbage can", "wastebin", "ash bin", "ash-bin", "ashbin", "dustb", "trash barrel", "trash bin'",
    "assault rifle", "assault gun",
    "backpack", "back pack", "knapsack", "packsack", "rucksack", "haversack",
    "bakery", "bakeshop", "bakehouse",
    "balance beam", "beam",
    "balloon",
    "ballpoint", "ballpoint pen", "ballpen", "Biro",
    "Band Aid",
    "banjo",
    "bannister", "banister", "balustrade", "balusters", "handrail",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel", "cask",
    "barrow", "garden cart", "lawn cart", "wheelbarrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "bathing cap", "swimming cap",
    "bath towel",
    "bathtub", "bathing tub", "bath", "tub",
    "beach wagon", "station wagon", "wagon", "estate car", "beach waggon", "station wagg", "waggon'",
    "beacon", "lighthouse", "beacon light", "pharos",
    "beaker",
    "bearskin", "busby", "shako",
    "beer bottle",
    "beer glass",
    "bell cote", "bell cot",
    "bib",
    "bicycle-built-for-two", "tandem bicycle", "tandem",
    "bikini", "two-piece",
    "binder", "ring-binder",
    "binoculars", "field glasses", "opera glasses",
    "birdhouse",
    "boathouse",
    "bobsled", "bobsleigh", "bob",
    "bolo tie", "bolo", "bola tie", "bola",
    "bonnet", "poke bonnet",
    "bookcase",
    "bookshop", "bookstore", "bookstall",
    "bottlecap",
    "bow",
    "bow tie", "bow-tie", "bowtie",
    "brass", "memorial tablet", "plaque",
    "brassiere", "bra", "bandeau",
    "breakwater", "groin", "groyne", "mole", "bulwark", "seawall", "jetty",
    "breastplate", "aegis", "egis",
    "broom",
    "bucket", "pail",
    "buckle",
    "bulletproof vest",
    "bullet train", "bullet",
    "butcher shop", "meat market",
    "cab", "hack", "taxi", "taxicab",
    "caldron", "cauldron",
    "candle", "taper", "wax light",
    "cannon",
    "canoe",
    "can opener", "tin opener",
    "cardigan",
    "car mirror",
    "carousel", "carrousel", "merry-go-round", "roundabout", "whirligig",
    "carpenter's kit", "tool kit",
    "carton",
    "car wheel",
    "cash machine", "cash dispenser", "automated teller machine", "automatic teller chine", "automated teller", "automatic teller", "ATM'",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello", "violoncello",
    "cellular telephone", "cellular phone", "cellphone", "cell", "mobile phone",
    "chain",
    "chainlink fence",
    "chain mail", "ring mail", "mail", "chain armor", "chain armour", "ring armor", "ring mour'",
    "chain saw", "chainsaw",
    "chest",
    "chiffonier", "commode",
    "chime", "bell", "gong",
    "china cabinet", "china closet",
    "Christmas stocking",
    "church", "church building",
    "cinema", "movie theater", "movie theatre", "movie house", "picture palace",
    "cleaver", "meat cleaver", "chopper",
    "cliff dwelling",
    "cloak",
    "clog", "geta", "patten", "sabot",
    "cocktail shaker",
    "coffee mug",
    "coffeepot",
    "coil", "spiral", "volute", "whorl", "helix",
    "combination lock",
    "computer keyboard", "keypad",
    "confectionery", "confectionary", "candy store",
    "container ship", "containership", "container vessel",
    "convertible",
    "corkscrew", "bottle screw",
    "cornet", "horn", "trumpet", "trump",
    "cowboy boot",
    "cowboy hat", "ten-gallon hat",
    "cradle",
    "crane",
    "crash helmet",
    "crate",
    "crib", "cot",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam", "dike", "dyke",
    "desk",
    "desktop computer",
    "dial telephone", "dial phone",
    "diaper", "nappy", "napkin",
    "digital clock",
    "digital watch",
    "dining table", "board",
    "dishrag", "dishcloth",
    "dishwasher", "dish washer", "dishwashing machine",
    "disk brake", "disc brake",
    "dock", "dockage", "docking facility",
    "dogsled", "dog sled", "dog sleigh",
    "dome",
    "doormat", "welcome mat",
    "drilling platform", "offshore rig",
    "drum", "membranophone", "tympan",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan", "blower",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso maker",
    "face powder",
    "feather boa", "boa",
    "file", "file cabinet", "filing cabinet",
    "fireboat",
    "fire engine", "fire truck",
    "fire screen", "fireguard",
    "flagpole", "flagstaff",
    "flute", "transverse flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster",
    "freight car",
    "French horn", "horn",
    "frying pan", "frypan", "skillet",
    "fur coat",
    "garbage truck", "dustcart",
    "gasmask", "respirator", "gas helmet",
    "gas pump", "gasoline pump", "petrol pump", "island dispenser",
    "goblet",
    "go-kart",
    "golf ball",
    "golfcart", "golf cart",
    "gondola",
    "gong", "tam-tam",
    "gown",
    "grand piano", "grand",
    "greenhouse", "nursery", "glasshouse",
    "grille", "radiator grille",
    "grocery store", "grocery", "food market", "market",
    "guillotine",
    "hair slide",
    "hair spray",
    "half track",
    "hammer",
    "hamper",
    "hand blower", "blow dryer", "blow drier", "hair dryer", "hair drier",
    "hand-held computer", "hand-held microcomputer",
    "handkerchief", "hankie", "hanky", "hankey",
    "hard disc", "hard disk", "fixed disk",
    "harmonica", "mouth organ", "harp", "mouth harp",
    "harp",
    "harvester", "reaper",
    "hatchet",
    "holster",
    "home theater", "home theatre",
    "honeycomb",
    "hook", "claw",
    "hoopskirt", "crinoline",
    "horizontal bar", "high bar",
    "horse cart", "horse-cart",
    "hourglass",
    "iPod",
    "iron", "smoothing iron",
    "jack-o'-lantern",
    "jean", "blue jean", "denim",
    "jeep", "landrover",
    "jersey", "T-shirt", "tee shirt",
    "jigsaw puzzle",
    "jinrikisha", "ricksha", "rickshaw",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat", "laboratory coat",
    "ladle",
    "lampshade", "lamp shade",
    "laptop", "laptop computer",
    "lawn mower", "mower",
    "lens cap", "lens cover",
    "letter opener", "paper knife", "paperknife",
    "library",
    "lifeboat",
    "lighter", "light", "igniter", "ignitor",
    "limousine", "limo",
    "liner", "ocean liner",
    "lipstick", "lip rouge",
    "Loafer",
    "lotion",
    "loudspeaker", "speaker", "speaker unit", "loudspeaker system", "speaker system",
    "loupe", "jeweler's loupe",
    "lumbermill", "sawmill",
    "magnetic compass",
    "mailbag", "postbag",
    "mailbox", "letter box",
    "maillot",
    "maillot", "tank suit",
    "manhole cover",
    "maraca",
    "marimba", "xylophone",
    "mask",
    "matchstick",
    "maypole",
    "maze", "labyrinth",
    "measuring cup",
    "medicine chest", "medicine cabinet",
    "megalith", "megalithic structure",
    "microphone", "mike",
    "microwave", "microwave oven",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt", "mini",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home", "manufactured home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "mortarboard",
    "mosque",
    "mosquito net",
    "motor scooter", "scooter",
    "mountain bike", "all-terrain bike", "off-roader",
    "mountain tent",
    "mouse", "computer mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook", "notebook computer",
    "obelisk",
    "oboe", "hautboy", "hautbois",
    "ocarina", "sweet potato",
    "odometer", "hodometer", "mileometer", "milometer",
    "oil filter",
    "organ", "pipe organ",
    "oscilloscope", "scope", "cathode-ray oscilloscope", "CRO",
    "overskirt",
    "oxcart",
    "oxygen mask",
    "packet",
    "paddle", "boat paddle",
    "paddlewheel", "paddle wheel",
    "padlock",
    "paintbrush",
    "pajama", "pyjama", "pj's", "jammies",
    "palace",
    "panpipe", "pandean pipe", "syrinx",
    "paper towel",
    "parachute", "chute",
    "parallel bars", "bars",
    "park bench",
    "parking meter",
    "passenger car", "coach", "carriage",
    "patio", "terrace",
    "pay-phone", "pay-station",
    "pedestal", "plinth", "footstall",
    "pencil box", "pencil case",
    "pencil sharpener",
    "perfume", "essence",
    "Petri dish",
    "photocopier",
    "pick", "plectrum", "plectron",
    "pickelhaube",
    "picket fence", "paling",
    "pickup", "pickup truck",
    "pier",
    "piggy bank", "penny bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate", "pirate ship",
    "pitcher", "ewer",
    "plane", "carpenter's plane", "woodworking plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow", "plough",
    "plunger", "plumber's helper",
    "Polaroid camera", "Polaroid Land camera",
    "pole",
    "police van", "police wagon", "paddy wagon", "patrol wagon", "wagon", "black Maria",
    "poncho",
    "pool table", "billiard table", "snooker table",
    "pop bottle", "soda bottle",
    "pot", "flowerpot",
    "potter's wheel",
    "power drill",
    "prayer rug", "prayer mat",
    "printer",
    "prison", "prison house",
    "projectile", "missile",
    "projector",
    "puck", "hockey puck",
    "punching bag", "punch bag", "punching ball", "punchball",
    "purse",
    "quill", "quill pen",
    "quilt", "comforter", "comfort", "puff",
    "racer", "race car", "racing car",
    "racket", "racquet",
    "radiator",
    "radio", "wireless",
    "radio telescope", "radio reflector",
    "rain barrel",
    "recreational vehicle", "RV", "R.V.",
    "reel",
    "reflex camera",
    "refrigerator", "icebox",
    "remote control", "remote",
    "restaurant", "eating house", "eating place", "eatery",
    "revolver", "six-gun", "six-shooter",
    "rifle",
    "rocking chair", "rocker",
    "rotisserie",
    "rubber eraser", "rubber", "pencil eraser",
    "rugby ball",
    "rule", "ruler",
    "running shoe",
    "safe",
    "safety pin",
    "saltshaker", "salt shaker",
    "sandal",
    "sarong",
    "sax", "saxophone",
    "scabbard",
    "scale", "weighing machine",
    "school bus",
    "schooner",
    "scoreboard",
    "screen", "CRT screen",
    "screw",
    "screwdriver",
    "seat belt", "seatbelt",
    "sewing machine",
    "shield", "buckler",
    "shoe shop", "shoe-shop", "shoe store",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule", "slipstick",
    "sliding door",
    "slot", "one-armed bandit",
    "snorkel",
    "snowmobile",
    "snowplow", "snowplough",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar dish", "solar collector", "solar furnace",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "speedboat",
    "spider web", "spider's web",
    "spindle",
    "sports car", "sport car",
    "spotlight", "spot",
    "stage",
    "steam locomotive",
    "steel arch bridge",
    "steel drum",
    "stethoscope",
    "stole",
    "stone wall",
    "stopwatch", "stop watch",
    "stove",
    "strainer",
    "streetcar", "tram", "tramcar", "trolley", "trolley car",
    "stretcher",
    "studio couch", "day bed",
    "stupa", "tope",
    "submarine", "pigboat", "sub", "U-boat",
    "suit", "suit of clothes",
    "sundial",
    "sunglass",
    "sunglasses", "dark glasses", "shades",
    "sunscreen", "sunblock", "sun blocker",
    "suspension bridge",
    "swab", "swob", "mop",
    "sweatshirt",
    "swimming trunks", "bathing trunks",
    "swing",
    "switch", "electric switch", "electrical switch",
    "syringe",
    "table lamp",
    "tank", "army tank", "armored combat vehicle", "armoured combat vehicle",
    "tape player",
    "teapot",
    "teddy", "teddy bear",
    "television", "television system",
    "tennis ball",
    "thatch", "thatched roof",
    "theater curtain", "theatre curtain",
    "thimble",
    "thresher", "thrasher", "threshing machine",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop", "tobacconist shop", "tobacconist",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck", "tow car", "wrecker",
    "toyshop",
    "tractor",
    "trailer truck", "tractor trailer", "trucking rig", "rig", "articulated lorry", "sem",
    "tray",
    "trench coat",
    "tricycle", "trike", "velocipede",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus", "trolley coach", "trackless trolley",
    "trombone",
    "tub", "vat",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle", "monocycle",
    "upright", "upright piano",
    "vacuum", "vacuum cleaner",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin", "fiddle",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet", "billfold", "notecase", "pocketbook",
    "wardrobe", "closet", "press",
    "warplane", "military plane",
    "washbasin", "handbasin", "washbowl", "lavabo", "wash-hand basin",
    "washer", "automatic washer", "washing machine",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool", "woolen", "woollen",
    "worm fence", "snake fence", "snake-rail fence", "Virginia fence",
    "wreck",
    "yawl",
    "yurt",
    "web site", "website", "internet site", "site",
    "comic book",
    "crossword puzzle", "crossword",
    "street sign",
    "traffic light", "traffic signal", "stoplight",
    "book jacket", "dust cover", "dust jacket", "dust wrapper",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot", "hotpot",
    "trifle",
    "ice cream", "icecream",
    "ice lolly", "lolly", "lollipop", "popsicle",
    "French loaf",
    "bagel", "beigel",
    "pretzel",
    "cheeseburger",
    "hotdog", "hot dog", "red hot",
    "mashed potato",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini", "courgette",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber", "cuke",
    "artichoke", "globe artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple", "ananas",
    "banana",
    "jackfruit", "jak", "jack",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate sauce", "chocolate syrup",
    "dough",
    "meat loaf", "meatloaf",
    "pizza", "pizza pie",
    "potpie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff", "drop", "drop-off",
    "coral reef",
    "geyser",
    "lakeside", "lakeshore",
    "promontory", "headland", "head", "foreland",
    "sandbar", "sand bar",
    "seashore", "coast", "seacoast", "sea-coast",
    "valley", "vale",
    "volcano",
    "ballplayer", "baseball player",
    "groom", "bridegroom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper", "yellow lady-slipper", "Cypripedium calceolus", "Cypripium parviflorum", 
    "corn",
    "acorn",
    "hip", "rose hip", "rosehip",
    "buckeye", "horse chestnut", "conker",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn", "carrion fungus",
    "earthstar",
    "hen-of-the-woods", "hen of the woods", "Polyporus frondosus", "Grifola frondosa",
    "bolete",
    "ear", "spike", "capitulum",
    "toilet tissue", "toilet paper", "bathroom tissue",
]



def prepare_cifar(num_classes=5, samples_per_class=1, batch_size=128, seed=42):

	# CIFAR-100 Dataset
	# Parameters allow switching between 5-way N-shot configurations

	# Constants
	NUM_CLASSES = num_classes
	SAMPLES_PER_CLASS = samples_per_class
	BATCH_SIZE = batch_size
	IMAGE_SIZE = 224


	# Define transforms for training and testing datasets
	transform_train = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_test = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	# Download CIFAR-100 datasets
	train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
	val_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
	test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)

	# Get all class names from the original CIFAR-100 dataset
	all_class_names = train_dataset.classes  # List of all 100 class names

	# Convert NAMES elements to lowercase and split them into words (use new variable to avoid scope issue)
	names_processed = [name.lower().split() for name in NAMES]

	# Function to check if any part of the class name matches any part of NAMES
	def is_excluded(class_name, names):
		if class_name in ['seal', 'trout']:
			return True
		class_parts = class_name.lower().replace('_', ' ').split()  # Split class name into words
		for class_part in class_parts:
			for name_parts in names:
				if class_part in name_parts:  # Check if any part matches
					return True
		return False


	# Filter classes by excluding those whose name is present in any element of NAMES
	filtered_class_names = [
		class_name for class_name in all_class_names if not is_excluded(class_name, names_processed)
	]

	# Check if there are enough classes after filtering
	if len(filtered_class_names) < NUM_CLASSES:
		print(f"Filtered classes ({len(filtered_class_names)}) are fewer than NUM_CLASSES ({NUM_CLASSES}). Using all filtered classes.")
		NUM_CLASSES = len(filtered_class_names)

	random.seed(seed) # Use episode-specific seed for class selection
	# Randomly select NUM_CLASSES classes
	selected_class_names = random.sample(filtered_class_names, NUM_CLASSES)

	# Select the first NUM_CLASSES classes from the filtered list
	#selected_class_names = filtered_class_names[1:NUM_CLASSES+1]
	selected_classes = [all_class_names.index(name) for name in selected_class_names]

	# Create a mapping for reassigning labels to be sequential from 0 to NUM_CLASSES - 1
	label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}

	# Function to extract 'num_images_per_class' images for each class in the dataset
	def extract_subset(dataset, selected_classes, label_mapping, num_images_per_class=5):
		class_counts = {label_mapping[old_label]: 0 for old_label in selected_classes}
		selected_indices = []
		
		for idx, (data, label) in enumerate(dataset):
			if label in selected_classes:
				new_label = label_mapping[label]
				if class_counts[new_label] < num_images_per_class:
					selected_indices.append((idx, new_label))
					class_counts[new_label] += 1
				if sum(class_counts.values()) == num_images_per_class * len(selected_classes):
					break
		
		return selected_indices

	# Filter datasets by the selected classes
	train_indices = extract_subset(train_dataset, selected_classes, label_mapping, num_images_per_class=SAMPLES_PER_CLASS)
	val_indices = extract_subset(val_dataset, selected_classes, label_mapping, num_images_per_class=100)
	test_indices = extract_subset(test_dataset, selected_classes, label_mapping, num_images_per_class=100)

	# Create Subset Datasets using module-level RemappedSubset class (for pickling)
	train_subset = RemappedSubset(train_dataset, train_indices)
	val_subset = RemappedSubset(val_dataset, val_indices)
	test_subset = RemappedSubset(test_dataset, test_indices)

	# Dataloaders (num_workers=0 for Windows compatibility)
	train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	eval_loader = DataLoader(dataset=val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
	test_loader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

	print(f"Classes After Filter: {len(filtered_class_names)}")
	print(f"Number of Classes Used: {NUM_CLASSES}")
	print(f"Selected Classes: {selected_class_names}")
	print(f"Training Sample: {len(train_subset)}")
	print(f"Evaluation samples: {len(val_subset)}")
	print(f"Test Samples: {len(test_subset)}")

	# Get all class names from the original CIFAR-100 dataset
	all_class_names = train_dataset.classes  # This is a list of all 100 class names
	# Get class names for the selected classes
	class_names = [all_class_names[i] for i in selected_classes]
	
	return train_loader, eval_loader, test_loader, NUM_CLASSES
	
	
def prepare_omniglot(seed=42):

	# Omniglot

	# Constants
	IMAGE_SIZE = 224  # Original size of Omniglot images for ResNet-50
	#IMAGE_SIZE = 105 # For ResNet-18
	NUM_CLASSES = 5  # Variable number of classes you want to use
	TRAIN_SAMPLES_PER_CLASS = 5  # Number of training samples per class
	TEST_SAMPLES_PER_CLASS = 5  # Number of testing samples per class


	# Transfroms for ResNet-18
	transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.Grayscale(num_output_channels=3),  # Convert grayscale image to 3 channels
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean values
								 std=[0.229, 0.224, 0.225])   # ImageNet std values
	])

	# Load the Omniglot datasets
	root_dir = './data'  # Directory to store the dataset

	# Download and load the background and evaluation datasets
	background_dataset = Omniglot(root=root_dir, background=True, download=True, transform=transform)
	evaluation_dataset = Omniglot(root=root_dir, background=False, download=True, transform=transform)

	# Adjust labels when combining datasets to avoid label conflicts
	class AdjustedOmniglotDataset(torch.utils.data.Dataset):
		def __init__(self, dataset, label_offset=0):
			self.dataset = dataset
			self.label_offset = label_offset
			
		def __getitem__(self, index):
			image, label = self.dataset[index]
			return image, label + self.label_offset
		
		def __len__(self):
			return len(self.dataset)

	# Adjust labels for background and evaluation datasets
	background_dataset_adjusted = AdjustedOmniglotDataset(background_dataset, label_offset=0)
	evaluation_dataset_adjusted = AdjustedOmniglotDataset(
		evaluation_dataset, label_offset=len(background_dataset._characters)
	)

	# Combine the datasets
	combined_dataset = ConcatDataset([background_dataset_adjusted, evaluation_dataset_adjusted])

	# Create a mapping from class labels to dataset indices
	class_to_indices = defaultdict(list)
	for idx in range(len(combined_dataset)):
		_, label = combined_dataset[idx]
		class_to_indices[label].append(idx)

	# Get all class labels
	all_labels = list(class_to_indices.keys())

	# Ensure NUM_CLASSES does not exceed the total number of classes
	total_classes = len(all_labels)
	if NUM_CLASSES > total_classes:
		print(f"NUM_CLASSES exceeds total number of classes ({total_classes}), setting NUM_CLASSES to {total_classes}")
		NUM_CLASSES = total_classes

	random.seed(seed) # Use episode-specific seed for class selection
	# Randomly select NUM_CLASSES classes
	selected_labels = random.sample(all_labels, NUM_CLASSES)

	# Create a mapping from original labels to new labels (0 to NUM_CLASSES-1)
	label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(selected_labels)}

	# Define a dataset that maps original labels to new labels
	class MappedLabelDataset(torch.utils.data.Dataset):
		def __init__(self, dataset, label_mapping):
			self.dataset = dataset
			self.label_mapping = label_mapping
			
		def __getitem__(self, index):
			image, label = self.dataset[index]
			if label in self.label_mapping:
				mapped_label = self.label_mapping[label]
				return image, mapped_label
			else:
				# Should not happen, as we only use indices from selected labels
				raise ValueError(f"Label {label} not in label mapping")
		
		def __len__(self):
			return len(self.dataset)

	# Collect indices for training, evaluation, and testing
	train_indices = []
	eval_indices = []
	test_indices = []

	for orig_label in selected_labels:
		indices = class_to_indices[orig_label]
		num_images = len(indices)
		if num_images == 0:
			continue  # Exclude classes with no images (should not happen)
		# Shuffle the indices for randomness
		random.shuffle(indices)
		if num_images >= TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS + 5:
			# Assign samples for training, evaluation, and testing
			train_samples = indices[:TRAIN_SAMPLES_PER_CLASS]
			eval_samples = indices[TRAIN_SAMPLES_PER_CLASS:TRAIN_SAMPLES_PER_CLASS + 10]
			test_samples = indices[TRAIN_SAMPLES_PER_CLASS + 10:TRAIN_SAMPLES_PER_CLASS + 10 + TEST_SAMPLES_PER_CLASS]
		else:
			# For classes with fewer images, attempt to distribute as evenly as possible
			split_train = min(TRAIN_SAMPLES_PER_CLASS, num_images // 3)
			split_eval = min(5, (num_images - split_train) // 2)
			split_test = num_images - split_train - split_eval

			train_samples = indices[:split_train]
			eval_samples = indices[split_train:split_train + split_eval]
			test_samples = indices[split_train + split_eval:]

		train_indices.extend(train_samples)
		eval_indices.extend(eval_samples)
		test_indices.extend(test_samples)

	# Create Subsets for training, evaluation, and testing
	mapped_dataset = MappedLabelDataset(combined_dataset, label_mapping)
	train_subset = Subset(mapped_dataset, train_indices)
	eval_subset = Subset(mapped_dataset, eval_indices)
	test_subset = Subset(mapped_dataset, test_indices)

	# DataLoaders
	train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	eval_loader = DataLoader(dataset=eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	test_loader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	# Print statistics
	print(f'Number of Classes: {NUM_CLASSES}')
	print(f"Number of training samples: {len(train_indices)}")
	print(f"Number of evaluation samples: {len(eval_indices)}")
	print(f"Number of testing samples: {len(test_indices)}")

	# Get class names for the selected classes
	class_names = [all_labels[i] for i in selected_labels]
	print(f"Selected Class Names: {class_names}")
	
	return train_loader, eval_loader, test_loader, NUM_CLASSES


def prepare_cub(data_path, seed=42):

	# CUB

	# Constants
	NUM_CLASSES = 5  # Number of classes to use
	IMAGE_SIZE = 224  # ResNet-18 expects 224x224 images
	NO_OF_SAMPLES = 5  # Number of samples per class for training

	# Directories
	images_dir = data_path

	# Verify if images directory exists
	if not os.path.exists(images_dir):
		raise FileNotFoundError(f'Images directory not found at {images_dir}')
	else:
		print('Images directory found.')

	# Define transformations suitable for ResNet-18
	transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
							 std=[0.229, 0.224, 0.225])   # ImageNet std
	])

	# Create the dataset
	dataset = ImageFolder(root=images_dir, transform=transform)

	# Get the class-to-index mapping
	class_to_idx = dataset.class_to_idx
	idx_to_class = {v: k for k, v in class_to_idx.items()}  # Map index to class name

	# Convert NAMES elements to lowercase and split them into words
	NAMES = [name.lower().split() for name in NAMES]

	# Function to check if any part of the class name matches any part of NAMES
	def is_excluded(class_name, names):
		class_parts = class_name.lower().replace('_', ' ').split()  # Split class name into words
		for class_part in class_parts:
			for name_parts in names:
				if class_part in name_parts:  # Check if any part matches
					return True
		return False

	# Filter classes by excluding those that are even remotely similar to NAMES
	filtered_class_indices = {
		idx: class_name
		for idx, class_name in idx_to_class.items()
		if not is_excluded(class_name, NAMES)
	}

	# Check if there are enough classes after filtering
	if len(filtered_class_indices) < NUM_CLASSES:
		print(f"Filtered classes ({len(filtered_class_indices)}) are fewer than NUM_CLASSES ({NUM_CLASSES}). Using all filtered classes.")
		NUM_CLASSES = len(filtered_class_indices)

	# Select NUM_CLASSES from filtered classes
	random.seed(seed) # Use episode-specific seed for class selection
	selected_class_indices = random.sample(list(filtered_class_indices.keys()), NUM_CLASSES)

	# Create a mapping from original class indices to new labels (0 to NUM_CLASSES - 1)
	class_idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_class_indices)}

	# Define a custom dataset to adjust labels according to the new mapping
	class SelectedClassesDataset(torch.utils.data.Dataset):
		def __init__(self, dataset, selected_class_indices, class_idx_mapping):
			self.dataset = dataset
			self.selected_class_indices = set(selected_class_indices)
			self.class_idx_mapping = class_idx_mapping
			self.indices = []
			for idx in range(len(dataset)):
				_, label = dataset.samples[idx]  # Use dataset.samples to avoid loading data
				if label in self.selected_class_indices:
					self.indices.append(idx)
		def __len__(self):
			return len(self.indices)
		def __getitem__(self, idx):
			dataset_idx = self.indices[idx]
			image, label = self.dataset[dataset_idx]
			new_label = self.class_idx_mapping[label]
			return image, new_label

	# Create the selected dataset
	selected_dataset = SelectedClassesDataset(dataset, selected_class_indices, class_idx_mapping)

	# Create a mapping from new labels to indices in the selected dataset
	class_to_indices_in_selected_dataset = {new_label: [] for new_label in range(NUM_CLASSES)}

	for idx in range(len(selected_dataset)):
		_, label = selected_dataset[idx]
		class_to_indices_in_selected_dataset[label].append(idx)

	# Process classes according to the instructions
	train_indices = []
	eval_indices = []
	test_indices = []

	for class_label, indices in class_to_indices_in_selected_dataset.items():
		num_images = len(indices)
		if num_images == 0:
			continue  # Exclude classes with no images (should not happen)
		
		# Shuffle indices for randomness
		random.shuffle(indices)
		
		train_samples = indices[:NO_OF_SAMPLES]  # First NO_OF_SAMPLES samples for training
		eval_samples = indices[NO_OF_SAMPLES:NO_OF_SAMPLES + 20]  # Next 20 for evaluation
		test_samples = indices[NO_OF_SAMPLES + 20:]  # Remaining samples for testing, up to 20
		
		train_indices.extend(train_samples)
		eval_indices.extend(eval_samples)
		test_indices.extend(test_samples)

	print(f"Number of valid classes used: {NUM_CLASSES}")
	print(f"Number of training samples: {len(train_indices)}")
	print(f"Number of evaluation samples: {len(eval_indices)}")
	print(f"Number of testing samples: {len(test_indices)}")

	# Create Subsets
	train_subset = Subset(selected_dataset, train_indices)
	eval_subset = Subset(selected_dataset, eval_indices)
	test_subset = Subset(selected_dataset, test_indices)

	# DataLoaders
	train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	eval_loader = DataLoader(dataset=eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	test_loader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	# NUM_CLASSES is already set
	print(f'Number of Classes: {NUM_CLASSES}')
	print(f'Classes After Filter: {len(filtered_class_indices)}')

	# Get class names for the selected classes
	class_names = [filtered_class_indices[i] for i in selected_class_indices]
	print(f"Selected Classes: {class_names}")

	return train_loader, eval_lodaer, test_loader, NUM_CLASSES


def prepare_caltech(data_path, seed=42):

	# Caltech-101

	# Constants
	NUM_CLASSES = 5  # Variable number of classes
	IMAGE_SIZE = 224  # ResNet-18 expects 224x224 images
	NO_OF_SAMPLES = 5

	# Directories
	images_dir = data_path

	# Verify if images directory exists
	if not os.path.exists(images_dir):
		raise FileNotFoundError(f'Images directory not found at {images_dir}')
	else:
		print('Images directory found.')

	# Define transformations suitable for ResNet-18
	transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
							 std=[0.229, 0.224, 0.225])   # ImageNet std
	])

	# Create the dataset
	dataset = ImageFolder(root=images_dir, transform=transform)

	# Get the class-to-index mapping (directories are class names)
	class_to_idx = dataset.class_to_idx
	idx_to_class = {v: k for k, v in class_to_idx.items()}  # Map index to class name

	# Filter classes based on substring matching with NAMES
	# Convert NAMES elements to lowercase and split them into words
	NAMES = [name.lower().split() for name in NAMES]

	# Function to check if any part of the class name matches any part of NAMES
	def is_excluded(class_name, names):
		class_parts = class_name.lower().replace('_', ' ').split()  # Split class name into words
		if class_name in ['Leopards', 'ferry', 'hawksbill', 'rooster', 'bass', 'helicopter', 'mandolin']:
			return True
		for class_part in class_parts:
			for name_parts in names:
				if class_part in name_parts:  # Check if any part matches
					return True
		return False

	# Filter classes by excluding those that are even remotely similar to NAMES
	filtered_class_indices = {
		idx: class_name
		for idx, class_name in idx_to_class.items()
		if not is_excluded(class_name, NAMES)
	}

	# Check if there are enough classes after filtering
	if len(filtered_class_indices) < NUM_CLASSES:
		print(f"Filtered classes ({len(filtered_class_indices)}) are fewer than NUM_CLASSES ({NUM_CLASSES}). Using all filtered classes.")
		NUM_CLASSES = len(filtered_class_indices)

	# Select NUM_CLASSES from filtered classes
	random.seed(seed) # Use episode-specific seed for class selection
	selected_class_indices = random.sample(list(filtered_class_indices.keys()), NUM_CLASSES)
	#selected_class_indices = list(filtered_class_indices.keys())[9:NUM_CLASSES+9]

	# Create a mapping from original class indices to new labels (0 to NUM_CLASSES - 1)
	class_idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_class_indices)}

	# Define a custom dataset to adjust labels according to the new mapping
	class SelectedClassesDataset(torch.utils.data.Dataset):
		def __init__(self, dataset, selected_class_indices, class_idx_mapping):
			self.dataset = dataset
			self.selected_class_indices = set(selected_class_indices)
			self.class_idx_mapping = class_idx_mapping
			self.indices = []
			for idx in range(len(dataset)):
				_, label = dataset.samples[idx]  # Use dataset.samples to avoid loading data
				if label in self.selected_class_indices:
					self.indices.append(idx)
		def __len__(self):
			return len(self.indices)
		def __getitem__(self, idx):
			dataset_idx = self.indices[idx]
			image, label = self.dataset[dataset_idx]
			new_label = self.class_idx_mapping[label]
			return image, new_label

	# Create the selected dataset
	selected_dataset = SelectedClassesDataset(dataset, selected_class_indices, class_idx_mapping)

	# Create a mapping from new labels to indices in the selected dataset
	class_to_indices_in_selected_dataset = {new_label: [] for new_label in range(NUM_CLASSES)}

	for idx in range(len(selected_dataset)):
		_, label = selected_dataset[idx]
		class_to_indices_in_selected_dataset[label].append(idx)

	# Process classes according to the instructions
	train_indices = []
	eval_indices = []
	test_indices = []

	for class_label, indices in class_to_indices_in_selected_dataset.items():
		num_images = len(indices)
		if num_images == 0:
			continue  # Exclude classes with no images (should not happen)
		
		# Shuffle indices for randomness
		random.shuffle(indices)
		
		train_samples = indices[:NO_OF_SAMPLES]  # First NO_OF_SAMPLES samples for training
		eval_samples = indices[NO_OF_SAMPLES:NO_OF_SAMPLES + 20]  # Next 20 for evaluation
		test_samples = indices[NO_OF_SAMPLES + 20:]  # Remaining samples for testing
		
		train_indices.extend(train_samples)
		eval_indices.extend(eval_samples)
		test_indices.extend(test_samples)

	print(f"Number of valid classes used: {NUM_CLASSES}")
	print(f"Number of training samples: {len(train_indices)}")
	print(f"Number of evaluation samples: {len(eval_indices)}")
	print(f"Number of testing samples: {len(test_indices)}")

	# Create Subsets
	train_subset = Subset(selected_dataset, train_indices)
	eval_subset = Subset(selected_dataset, eval_indices)
	test_subset = Subset(selected_dataset, test_indices)

	# DataLoaders
	train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	eval_loader = DataLoader(dataset=eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	test_loader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	# NUM_CLASSES is already set
	print(f'Number of Classes: {NUM_CLASSES}')
	print(f'Classes After Filter: {len(filtered_class_indices)}')

	# Get class names for the selected classes
	class_names = [filtered_class_indices[i] for i in selected_class_indices]
	print(f"Selected Classes: {class_names}")

	return train_loader, eval_loader, test_loader, NUM_CLASSES


def prepare_eurosat(data_path, seed=42):

    # EuroSat

	# Constants
	NUM_CLASSES = 5  # Variable number of classes 
	IMAGE_SIZE = 224  # ResNet-18 expects 224x224 images
	NO_OF_SAMPLES = 5

	# Directories
	images_dir = data_path

	# Verify if images directory exists
	if not os.path.exists(images_dir):
		raise FileNotFoundError(f'Images directory not found at {images_dir}')
	else:
		print('Images directory found.')

	# Define transformations suitable for ResNet-18
	transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
							 std=[0.229, 0.224, 0.225])   # ImageNet std
	])

	# Create the dataset
	dataset = ImageFolder(root=images_dir, transform=transform)

	# Get the class-to-index mapping
	class_to_idx = dataset.class_to_idx
	idx_to_class = {v: k for k, v in class_to_idx.items()}

	# Initialize structures
	class_to_indices = {}  # Mapping from class index to list of dataset indices

	# Organize dataset indices per class
	for idx, (data, label) in enumerate(dataset):
		if label not in class_to_indices:
			class_to_indices[label] = []
		class_to_indices[label].append(idx)

	# Get all class indices
	all_class_indices = list(class_to_indices.keys())

	# Ensure NUM_CLASSES does not exceed the total number of classes
	total_classes = len(all_class_indices)
	if NUM_CLASSES > total_classes:
		print(f"NUM_CLASSES ({NUM_CLASSES}) exceeds total number of classes ({total_classes}). Using {total_classes} classes.")
		NUM_CLASSES = total_classes

	random.seed(seed) # Use episode-specific seed for class selection
	# Randomly select NUM_CLASSES class indices
	selected_class_indices = random.sample(all_class_indices, NUM_CLASSES)

	# Create a mapping from original class indices to new labels (0 to NUM_CLASSES - 1)
	class_idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_class_indices)}

	# Define a custom dataset to adjust labels according to the new mapping
	class SelectedClassesDataset(torch.utils.data.Dataset):
		def __init__(self, dataset, selected_class_indices, class_idx_mapping):
			self.dataset = dataset
			self.selected_class_indices = set(selected_class_indices)
			self.class_idx_mapping = class_idx_mapping
			self.indices = []
			for idx in range(len(dataset)):
				_, label = dataset.samples[idx]  # Use dataset.samples to avoid loading data
				if label in self.selected_class_indices:
					self.indices.append(idx)
		def __len__(self):
			return len(self.indices)
		def __getitem__(self, idx):
			dataset_idx = self.indices[idx]
			image, label = self.dataset[dataset_idx]
			new_label = self.class_idx_mapping[label]
			return image, new_label

	# Create the selected dataset
	selected_dataset = SelectedClassesDataset(dataset, selected_class_indices, class_idx_mapping)

	# Create a mapping from new labels to indices in the selected dataset
	class_to_indices_in_selected_dataset = {new_label: [] for new_label in range(NUM_CLASSES)}

	for idx in range(len(selected_dataset)):
		_, label = selected_dataset[idx]
		class_to_indices_in_selected_dataset[label].append(idx)

	# Process classes according to the instructions
	train_indices = []
	eval_indices = []
	test_indices = []

	for class_label, indices in class_to_indices_in_selected_dataset.items():
		num_images = len(indices)
		if num_images == 0:
			continue  # Exclude classes with no images (should not happen)
		
		# Shuffle indices for randomness
		random.shuffle(indices)
		
		train_samples = indices[:NO_OF_SAMPLES]  # First 5 samples for training
		eval_samples = indices[NO_OF_SAMPLES:NO_OF_SAMPLES + 20]  # Next 20 for evaluation
		test_samples = indices[NO_OF_SAMPLES + 50:]  # Remaining samples for testing, up to 50
		
		train_indices.extend(train_samples)
		eval_indices.extend(eval_samples)
		test_indices.extend(test_samples)

	print(f"Number of valid classes used: {NUM_CLASSES}")
	print(f"Number of training samples: {len(train_indices)}")
	print(f"Number of evaluation samples: {len(eval_indices)}")
	print(f"Number of testing samples: {len(test_indices)}")

	# Create Subsets
	train_subset = Subset(selected_dataset, train_indices)
	eval_subset = Subset(selected_dataset, eval_indices)
	test_subset = Subset(selected_dataset, test_indices)

	# DataLoaders
	train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	eval_loader = DataLoader(dataset=eval_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	test_loader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	# NUM_CLASSES is already set
	print(f'Number of Classes: {NUM_CLASSES}')

	# Get all class names from the original CUB dataset
	all_class_names = list(all_class_indices)  # This is a list of all class names
	# Get class names for the selected classes
	class_names = [all_class_names[i] for i in selected_class_indices]

	return train_loader, eval_loader, test_loader, NUM_CLASSES

# ─────────────────────────────────────────────────────────────────
# PASCAL-5i SUPPORT  (added for FSS)
# ─────────────────────────────────────────────────────────────────
from PIL import Image
import torchvision.transforms as T

# ── Corrected Pascal-5i section — replace previous version ──────
import torch.nn.functional as F_dl   # avoid name collision with APM's F
from scipy.io import loadmat

class Pascal5iEpisodic(torch.utils.data.Dataset):
    """
    Episodic wrapper around Pascal5iReader.

    The raw reader returns (img, mask) pairs.
    This wrapper builds proper few-shot episodes:
      - pick a target class
      - sample k_shot support images + 1 query image for that class
      - return binary masks (1=target class, 0=everything else)

    n_episodes controls how many episodes are pre-generated.
    All episodes are deterministic given the seed.
    """
    def __init__(self, pascal5i_reader, k_shot=1,
                 img_size=473, n_episodes=1000, seed=42):
        self.reader     = pascal5i_reader
        self.k_shot     = k_shot
        self.img_size   = img_size
        self.label_set  = pascal5i_reader.label_set  # list of class IDs

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Pre-generate episodes for full reproducibility
        rng = np.random.RandomState(seed)
        self.episodes = []

        for _ in range(n_episodes * 5):   # oversample, filter below
            if len(self.episodes) >= n_episodes:
                break

            # Pick a class from the label set (1-indexed inside reader)
            # label_set index 0 → class_img_map key 1, etc.
            ls_idx  = rng.randint(0, len(self.label_set))
            cls_key = ls_idx + 1   # class_img_map uses 1-based keys

            available = pascal5i_reader.get_img_containing_class(cls_key)
            if len(available) < k_shot + 1:
                continue

            chosen          = rng.choice(available, k_shot + 1, replace=False)
            support_indices = list(chosen[:k_shot])
            query_index     = int(chosen[k_shot])
            self.episodes.append((cls_key, support_indices, query_index))

    def __len__(self):
        return len(self.episodes)

    def _process(self, img_tensor, mask_tensor, cls_key):
        """
        Resize image and mask to img_size, normalize image,
        binarize mask for cls_key.
        """
        # img_tensor: [3, H, W]  (already tensor from reader)
        img_r = F_dl.interpolate(
            img_tensor.unsqueeze(0).float(),
            size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=True
        ).squeeze(0)                                   # [3, img_size, img_size]

        # Normalize
        img_n = (img_r - self.mean) / self.std        # [3, img_size, img_size]

        # mask_tensor: [H, W] long
        mask_r = F_dl.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze().long()                             # [img_size, img_size]

        # Binary mask: 1 where target class, 0 everywhere else
        binary = (mask_r == cls_key).long()            # [img_size, img_size]

        return img_n, binary

    def __getitem__(self, idx):
        cls_key, support_indices, query_index = self.episodes[idx]

        s_imgs, s_masks = [], []
        for si in support_indices:
            img, mask = self.reader[si]               # [3,H,W], [H,W]
            img_n, bin_mask = self._process(img, mask, cls_key)
            s_imgs.append(img_n)
            s_masks.append(bin_mask)

        q_img, q_mask = self.reader[query_index]
        q_img_n, q_bin_mask = self._process(q_img, q_mask, cls_key)

        return (
            torch.stack(s_imgs),    # [k_shot, 3, H, W]
            torch.stack(s_masks),   # [k_shot, H, W]
            q_img_n,                # [3, H, W]
            q_bin_mask              # [H, W]
        )


def prepare_pascal5i(data_root, fold=0, k_shot=1, img_size=473,
                     n_train_episodes=2000, n_test_episodes=1000,
                     val_fraction=0.1, batch_size=4, seed=42):
    """
    Returns train_loader, val_loader, test_loader for Pascal-5i fold.

    Args:
        data_root         : parent folder containing BOTH sbd/ and VOCdevkit/
        fold              : 0-3
        k_shot            : 1 or 5
        img_size          : resize all images/masks to this square size
        n_train_episodes  : how many train episodes to pre-generate
        n_test_episodes   : how many test episodes to pre-generate
        val_fraction      : fraction of train episodes held out for val
        batch_size        : episodes per batch
        seed              : random seed

    IMPORTANT — data_root must look like:
        data_root/
        ├── sbd/
        │   ├── train.txt
        │   ├── val.txt
        │   ├── img/
        │   └── cls/
        └── VOCdevkit/
            └── VOC2012/
                ├── JPEGImages/
                ├── SegmentationClass/
                └── ImageSets/Segmentation/
    """
    from data.fss_dataset.pascal5i_reader import Pascal5iReader # copied from RogerQi repo

    torch.manual_seed(seed)

    # train=True  → base classes (15 classes, SBD+VOC combined)
    # train=False → novel classes (5 classes, VOC val only)
    raw_train = Pascal5iReader(data_root, fold=fold, train=True)
    raw_test  = Pascal5iReader(data_root, fold=fold, train=False)

    n_val   = max(1, int(n_train_episodes * val_fraction))
    n_train = n_train_episodes - n_val

    train_ds = Pascal5iEpisodic(raw_train, k_shot=k_shot, img_size=img_size,
                                 n_episodes=n_train, seed=seed)
    val_ds   = Pascal5iEpisodic(raw_train, k_shot=k_shot, img_size=img_size,
                                 n_episodes=n_val,   seed=seed + 1)
    test_ds  = Pascal5iEpisodic(raw_test,  k_shot=k_shot, img_size=img_size,
                                 n_episodes=n_test_episodes, seed=seed + 2)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=4, pin_memory=True
    )
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, 1   # NUM_CLASSES=1 (binary)