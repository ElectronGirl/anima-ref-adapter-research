"""Scene vocabulary sampled independently per image.

Everything here describes the scene (framing, camera, pose, expression, place, light),
never the character's identity. Training captions keep these tags and drop the
character and artist tags, so identity and style must come from the reference image.
All tags are general-rating Danbooru tags in Anima spelling.
"""
from __future__ import annotations

# framing -> (weight, candidate resolutions). Resolutions are ~1MP and multiples of 64.
FRAMINGS = {
    "portrait": (0.18, [(1024, 1024), (896, 1152)]),
    "upper body": (0.30, [(896, 1152), (1024, 1024)]),
    "cowboy shot": (0.24, [(832, 1216), (896, 1152)]),
    "full body": (0.20, [(832, 1216)]),
    "full body, wide shot": (0.08, [(1216, 832)]),
}

CAMERA = [
    ("", 0.40), ("looking at viewer", 0.25), ("from side", 0.07), ("from above", 0.07),
    ("from below", 0.06), ("dutch angle", 0.05), ("looking to the side", 0.05),
    ("looking back", 0.03), ("profile", 0.02),
]

# Poses compatible with every framing; full-body-only poses are listed separately.
POSES = [
    "standing", "arms up", "arms behind back", "hand on own hip", "hands on own hips",
    "waving", "v", "pointing at viewer", "hand on own chest", "crossed arms",
    "hand up", "outstretched arm", "head tilt", "leaning forward", "hands in pockets",
    "clenched hands", "hand on own cheek", "hand on own chin", "own hands together",
    "salute", "stretching", "adjusting hair", "tucking hair", "covering own mouth",
    "holding umbrella", "holding book", "reading", "holding cup", "drinking",
    "eating", "holding phone", "selfie", "holding flower", "holding bag",
    "holding sword", "holding staff", "holding microphone", "playing guitar",
    "writing", "hugging object", "holding stuffed toy", "heart hands", "thumbs up",
    "fighting stance", "dancing", "singing", "looking at phone", "holding box",
]
FULL_BODY_POSES = [
    "sitting", "sitting, on chair", "sitting, on bench", "walking", "running",
    "jumping", "kneeling", "squatting", "lying, on back", "lying, on stomach",
    "hugging own legs", "crossed legs", "against wall", "on one knee",
    "standing on one leg", "sitting on stairs", "seiza", "floating",
]

EXPRESSIONS = [
    ("smile", 0.20), ("open mouth, smile", 0.12), ("grin", 0.06), ("light smile", 0.08),
    ("expressionless", 0.06), ("serious", 0.06), ("surprised", 0.05), ("blush", 0.05),
    ("embarrassed", 0.04), ("pout", 0.04), ("angry", 0.04), ("sad", 0.03),
    ("crying", 0.02), ("closed eyes", 0.04), ("one eye closed", 0.03), ("smug", 0.03),
    ("laughing", 0.03), ("sleepy", 0.02),
]

BACKGROUNDS = [
    ("simple background, white background", 0.06), ("simple background, grey background", 0.04),
    ("gradient background", 0.03), ("classroom", 0.04), ("school, hallway", 0.02),
    ("city, street", 0.05), ("city, night, neon lights", 0.03), ("park", 0.04),
    ("beach, ocean", 0.04), ("forest", 0.04), ("cafe", 0.03), ("bedroom", 0.03),
    ("kitchen", 0.02), ("library", 0.03), ("train station", 0.02), ("train interior", 0.02),
    ("rooftop", 0.03), ("shrine, torii", 0.02), ("castle", 0.02), ("throne room", 0.01),
    ("mountain, snow", 0.02), ("flower field", 0.03), ("street, rain", 0.02),
    ("sunset, sky, cloud", 0.03), ("starry sky, night sky", 0.02), ("space", 0.01),
    ("underwater", 0.01), ("desert", 0.01), ("ruins", 0.02), ("bookshelf, indoors", 0.02),
    ("living room", 0.02), ("market stall", 0.02), ("bridge, river", 0.02),
    ("garden", 0.02), ("cherry blossoms", 0.03), ("autumn leaves", 0.02),
    ("snow, winter", 0.02), ("amusement park", 0.01), ("stage, spotlight", 0.02),
    ("office", 0.01), ("convenience store", 0.01), ("mountain, landscape", 0.02),
]

LIGHTING = [
    ("", 0.45), ("day", 0.10), ("sunlight", 0.08), ("sunset", 0.06), ("night", 0.06),
    ("backlighting", 0.05), ("light rays", 0.05), ("sunbeam", 0.04),
    ("shadow", 0.03), ("light particles", 0.03), ("lens flare", 0.02), ("dappled sunlight", 0.03),
]
