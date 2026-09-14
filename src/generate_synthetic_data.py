"""
generate_synthetic_data.py
Generates a synthetic support-ticket dataset where the ticket text has
REAL, learnable correlation with `category` and `urgency` labels
(unlike the raw Kaggle dataset, whose text is templated filler unrelated
to its labels).

Usage:
    python src/generate_synthetic_data.py
Produces:
    data/raw/tickets_raw.csv
"""

import random
import pandas as pd

random.seed(42)

N_ROWS = 1200

PRODUCTS = [
    "GoPro Hero", "Fitbit Charge", "Sony PlayStation", "Amazon Kindle",
    "Samsung Soundbar", "LG Smart TV", "Apple AirPods", "Garmin Forerunner",
    "Lenovo ThinkPad", "Roomba Robot Vacuum", "Adobe Photoshop", "Dropbox Plus",
]

# --- Category-specific content: subject templates + body fragments ---
CATEGORY_CONTENT = {
    "Technical issue": {
        "subjects": ["App keeps crashing", "Device won't turn on", "Sync not working", "Login error", "Screen freezing"],
        "bodies": [
            "The {product} keeps crashing every time I try to open it.",
            "I can't log in to my {product} account, it keeps showing an error message.",
            "My {product} freezes randomly and I have to restart it.",
            "The {product} isn't syncing with my phone anymore.",
            "I get an error code whenever I try to update my {product}.",
            "The screen on my {product} went black and won't respond.",
        ],
    },
    "Billing inquiry": {
        "subjects": ["Unexpected charge on my card", "Question about my invoice", "Subscription price changed", "Duplicate charge"],
        "bodies": [
            "I was charged twice for my {product} subscription this month.",
            "My invoice for {product} shows an amount I don't recognize.",
            "The price for my {product} subscription increased without notice.",
            "Can you explain why I was billed for {product} when I haven't used it?",
            "I'd like to update the payment method on file for my {product} account.",
            "There's a charge on my card for {product} that I don't remember authorizing.",
        ],
    },
    "Refund request": {
        "subjects": ["Requesting a refund", "Want my money back", "Product not as described", "Refund for defective item"],
        "bodies": [
            "The {product} arrived damaged and I'd like a full refund.",
            "I'm not satisfied with my {product} and want my money back.",
            "The {product} doesn't work as advertised, please process a refund.",
            "I returned my {product} last week and I'm still waiting on my refund.",
            "Please refund my purchase of the {product}, it's not what I expected.",
            "I'd like to request a refund for the {product} I bought.",
        ],
    },
    "Cancellation request": {
        "subjects": ["Cancel my subscription", "Please cancel my order", "Stop my membership", "Discontinue service"],
        "bodies": [
            "I want to cancel my {product} subscription effective immediately.",
            "Please cancel my order for the {product}, I no longer need it.",
            "I'd like to stop my {product} membership and avoid future charges.",
            "Can you help me cancel my recurring {product} plan?",
            "I no longer want the {product} service, please discontinue it.",
            "Please cancel my account associated with {product}.",
        ],
    },
    "Product inquiry": {
        "subjects": ["Question about compatibility", "Does this support...", "Product specifications", "Availability question"],
        "bodies": [
            "Does the {product} work with older versions of the app?",
            "I'm curious if the {product} is compatible with Android devices.",
            "What are the battery specifications for the {product}?",
            "Is the {product} currently available in a larger size?",
            "Can you tell me more about the warranty on the {product}?",
            "I'm considering buying the {product} — does it support offline mode?",
        ],
    },
}

# --- Urgency-specific framing sentences, appended to the body ---
URGENCY_FRAMING = {
    "Critical": [
        "This is completely blocking my work and I need it resolved immediately.",
        "This is urgent — I'm losing money every hour this isn't fixed.",
        "This is a critical issue affecting my entire team right now.",
        "I need an emergency fix as soon as possible, this cannot wait.",
    ],
    "High": [
        "This is a significant problem and I'd appreciate a quick resolution.",
        "I need this fixed soon, it's affecting my daily work.",
        "This has been frustrating and I'd like it prioritized.",
        "Please treat this as a high priority, it's causing real disruption.",
    ],
    "Medium": [
        "It's not blocking me completely, but I'd like it resolved soon.",
        "This is a moderate inconvenience, please look into it when you can.",
        "I'd appreciate a fix in the next few days if possible.",
        "It's manageable for now, but I'd like this addressed.",
    ],
    "Low": [
        "No rush on this, just wanted to flag it.",
        "This is a minor issue, whenever you get a chance is fine.",
        "Just curious about this, not urgent at all.",
        "Low priority, but wanted to check in about it.",
    ],
}

CATEGORIES = list(CATEGORY_CONTENT.keys())
URGENCIES = list(URGENCY_FRAMING.keys())

# Neutral filler sentences with no category/urgency signal, added at random
# to make the classification task less trivially keyword-based.
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

# A couple of the milder urgency phrasings sometimes get reused across
# adjacent urgency levels, softening the boundary between High/Medium
# and Medium/Low so the task isn't perfectly separable.
URGENCY_OVERLAP = {
    "High": URGENCY_FRAMING["High"] + URGENCY_FRAMING["Medium"][:1],
    "Medium": URGENCY_FRAMING["Medium"] + URGENCY_FRAMING["High"][-1:] + URGENCY_FRAMING["Low"][:1],
    "Low": URGENCY_FRAMING["Low"] + URGENCY_FRAMING["Medium"][-1:],
    "Critical": URGENCY_FRAMING["Critical"],
}


def generate_row(ticket_id: int) -> dict:
    category = random.choice(CATEGORIES)
    urgency = random.choice(URGENCIES)
    product = random.choice(PRODUCTS)

    content = CATEGORY_CONTENT[category]
    subject = random.choice(content["subjects"])

    # Sometimes combine two body fragments from the category for more
    # varied phrasing instead of always a single fixed sentence.
    if random.random() < 0.4:
        body = " ".join(random.sample(content["bodies"], 2)).format(product=product)
    else:
        body = random.choice(content["bodies"]).format(product=product)

    framing = random.choice(URGENCY_OVERLAP[urgency])

    parts = [body, framing]
    # Randomly inject 0-2 neutral filler sentences at random positions
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