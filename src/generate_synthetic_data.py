"""
generate_synthetic_data.py (v2 — harder)
Generates a synthetic support-ticket dataset designed so that classifying
correctly requires understanding CONTEXT/IMPLICATION rather than spotting
a single trigger keyword. This gives a contextual model (DistilBERT) real
room to outperform a bag-of-words baseline (TF-IDF + LinearSVC).

Key differences from v1:
  - Urgency is signaled through CONSEQUENCES/SCENARIOS, not words like
    "urgent"/"critical"/"low priority" (e.g. "I have a client demo in
    10 minutes" implies Critical without ever saying "urgent").
  - Category bodies include occasional HARD-NEGATIVE distractor phrases
    that mention vocabulary from a different category, to punish naive
    keyword matching.
  - Heavier phrasing variation reduces exact n-gram overlap between
    examples of the same class.

Usage:
    python src/generate_synthetic_data.py
Produces:
    data/raw/tickets_raw.csv
"""

import random
import pandas as pd

random.seed(42)

N_ROWS = 1400

PRODUCTS = [
    "GoPro Hero", "Fitbit Charge", "Sony PlayStation", "Amazon Kindle",
    "Samsung Soundbar", "LG Smart TV", "Apple AirPods", "Garmin Forerunner",
    "Lenovo ThinkPad", "Roomba Robot Vacuum", "Adobe Photoshop", "Dropbox Plus",
]

# ---------------------------------------------------------------------------
# Category content: bodies avoid a single fixed trigger word where possible,
# and each category has "distractor" phrases borrowed from OTHER categories'
# vocabulary, injected occasionally as hard negatives.
# ---------------------------------------------------------------------------
CATEGORY_CONTENT = {
    "Technical issue": {
        "subjects": ["App keeps crashing", "Device won't turn on", "Sync not working", "Login error", "Screen freezing"],
        "bodies": [
            "Every time I open the {product} it just shuts itself back down.",
            "I can't get past the sign-in screen on my {product}, it just spins forever.",
            "The {product} locks up randomly and I have to restart it to use it again.",
            "My {product} stopped talking to my phone, nothing shows up when I try to connect them.",
            "Something goes wrong every time I try to update the {product}, it never finishes.",
            "The display on the {product} went completely black and nothing brings it back.",
            "The {product} worked fine yesterday but now none of the buttons respond at all.",
        ],
        "distractors": [
            "I did check my recent statement first, but that's unrelated to this.",
            "I'm not asking for money back, I just want it to actually work.",
            "I already know how to use the {product}, this isn't a how-to question.",
        ],
    },
    "Billing inquiry": {
        "subjects": ["Unexpected charge on my card", "Question about my invoice", "Subscription price changed", "Duplicate charge"],
        "bodies": [
            "There are two identical charges on my statement for the same {product} order.",
            "The amount on my latest invoice for {product} doesn't match what I agreed to pay.",
            "My {product} plan renewed at a higher rate than what I originally signed up for.",
            "I don't recognize one of the line items tied to my {product} account this month.",
            "I'd like to switch which card is used for my {product} payments going forward.",
            "Something on my card statement references {product} but I can't tell what it's for.",
        ],
        "distractors": [
            "The {product} itself works fine, this is purely about the charge.",
            "I'm not trying to cancel anything, I just want the amount corrected.",
            "This isn't a technical problem, the device is working as expected.",
        ],
    },
    "Refund request": {
        "subjects": ["Requesting a refund", "Want my money back", "Product not as described", "Refund for defective item"],
        "bodies": [
            "The {product} showed up with visible damage and I'd rather send it back than keep it.",
            "This isn't what I expected from the {product} at all, I'd like to return it.",
            "The {product} doesn't do what the listing said it would, I want to send it back.",
            "I mailed my {product} back last week and haven't seen anything credited yet.",
            "I'd like to return the {product} I bought and get my payment reversed.",
            "The {product} arrived in the wrong condition, I'd rather get my money returned.",
        ],
        "distractors": [
            "I'm not trying to cancel any ongoing plan, this was a one-time purchase.",
            "It's not a technical glitch, the {product} just isn't a good fit for me.",
            "There's no billing error here, I simply want to send the item back.",
        ],
    },
    "Cancellation request": {
        "subjects": ["Cancel my subscription", "Please cancel my order", "Stop my membership", "Discontinue service"],
        "bodies": [
            "I'd like my {product} plan to stop renewing starting next cycle.",
            "Please take my {product} order off the books, I no longer want it shipped.",
            "I want to end my {product} membership and not be billed again.",
            "Can you close out my recurring {product} plan for me?",
            "I no longer need the {product} service going forward, please shut it down.",
            "I'd like to be taken off the {product} plan entirely.",
        ],
        "distractors": [
            "I'm not looking for a refund on past charges, just to stop future ones.",
            "The {product} itself works fine, I just don't need it anymore.",
            "This isn't a complaint about quality, I simply want to stop the service.",
        ],
    },
    "Product inquiry": {
        "subjects": ["Question about compatibility", "Does this support...", "Product specifications", "Availability question"],
        "bodies": [
            "Before I buy, I'm wondering if the {product} works with older app versions.",
            "Would the {product} pair properly with an Android phone?",
            "How long does the battery last on the {product} under normal use?",
            "Is a larger size of the {product} available anywhere right now?",
            "What does the warranty actually cover on the {product}?",
            "Does the {product} still function without an internet connection?",
        ],
        "distractors": [
            "I haven't purchased it yet, so this isn't about a charge or refund.",
            "Nothing is broken, I'm just trying to decide if it fits my needs.",
            "I'm not cancelling anything, I don't even own one yet.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Urgency: signaled through scenario/consequence, NOT explicit priority words.
# This is the key change — a bag-of-words model has to rely on loose n-gram
# correlations with time-pressure phrases, while a contextual model can
# actually reason about what the scenario implies.
# ---------------------------------------------------------------------------
URGENCY_FRAMING = {
    "Critical": [
        "I have a live client demo in the next ten minutes and this is what I'm using.",
        "Our whole team is sitting idle right now because of this.",
        "We're set to lose the contract if this isn't sorted before end of day.",
        "Every minute this stays broken is costing us active customers.",
        "This is the only thing standing between us and going live in front of investors this afternoon.",
    ],
    "High": [
        "I have a deadline first thing tomorrow morning and was counting on this.",
        "This has been going on for two days now and it's starting to pile up.",
        "I've got a big presentation later this week that depends on this working.",
        "It's cutting into my work every single day until it's fixed.",
        "A few of my colleagues are waiting on me because of this.",
    ],
    "Medium": [
        "It's not stopping me completely, but I'd like it looked at sometime this week.",
        "I can work around it for now, but it would be nice to have it sorted soon.",
        "It's a bit of a hassle but nothing is on fire because of it.",
        "I'll manage until someone gets a chance to take a look.",
        "It's annoying enough that I wanted to flag it, but it's not blocking anything major.",
    ],
    "Low": [
        "There's no particular timeline on my end, just wanted to mention it.",
        "I noticed this a while ago and it hasn't really affected anything for me.",
        "Whenever someone has a spare moment is totally fine.",
        "I'm mostly just curious, it's not something I need resolved quickly.",
        "This has been sitting fine as-is, just thought I'd bring it up.",
    ],
}

NEUTRAL_FILLERS = [
    "I've been a customer for a couple of years now.",
    "Thanks in advance for your help.",
    "Let me know if you need any more details from my end.",
    "I contacted support once before about a different issue.",
    "I saw this mentioned in an online forum as well.",
    "Hope to hear back soon.",
    "I'm reaching out from my registered email address.",
    "This happened a few days ago.",
]

# Soften the boundary between adjacent urgency levels so it isn't perfectly
# separable even by scenario type alone.
URGENCY_OVERLAP = {
    "Critical": URGENCY_FRAMING["Critical"],
    "High": URGENCY_FRAMING["High"] + URGENCY_FRAMING["Critical"][-1:],
    "Medium": URGENCY_FRAMING["Medium"] + URGENCY_FRAMING["High"][-1:],
    "Low": URGENCY_FRAMING["Low"] + URGENCY_FRAMING["Medium"][-1:],
}

CATEGORIES = list(CATEGORY_CONTENT.keys())
URGENCIES = list(URGENCY_FRAMING.keys())


def generate_row(ticket_id: int) -> dict:
    category = random.choice(CATEGORIES)
    urgency = random.choice(URGENCIES)
    product = random.choice(PRODUCTS)

    content = CATEGORY_CONTENT[category]
    subject = random.choice(content["subjects"])

    if random.random() < 0.4:
        body = " ".join(random.sample(content["bodies"], 2)).format(product=product)
    else:
        body = random.choice(content["bodies"]).format(product=product)

    framing = random.choice(URGENCY_OVERLAP[urgency])

    parts = [body, framing]

    # Hard negative: ~35% chance of injecting a distractor phrase from a
    # different-category vocabulary domain, worded as a denial (so it's
    # actively misleading for keyword matching).
    if random.random() < 0.35:
        distractor = random.choice(content["distractors"]).format(product=product)
        parts.insert(random.randint(0, len(parts)), distractor)

    for _ in range(random.choice([0, 0, 1, 1, 2])):
        parts.insert(random.randint(0, len(parts)), random.choice(NEUTRAL_FILLERS))

    description = " ".join(parts)

    return {
        "Ticket ID": ticket_id,
        "Product Purchased": product,
        "Ticket Subject": subject,
        "Ticket Description": description,
        "Ticket Type": category,
        "Ticket Priority": urgency,
    }


def main():
    rows = [generate_row(i) for i in range(1, N_ROWS + 1)]
    df = pd.DataFrame(rows)
    df.to_csv("data/raw/tickets_raw.csv", index=False)
    print(f"Generated {len(df)} synthetic tickets -> data/raw/tickets_raw.csv")
    print("\nCategory distribution:\n", df["Ticket Type"].value_counts())
    print("\nUrgency distribution:\n", df["Ticket Priority"].value_counts())


if __name__ == "__main__":
    main()