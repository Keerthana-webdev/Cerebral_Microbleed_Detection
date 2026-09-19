"""
Lesson 11a: A small reusable function that turns a raw probability into a
human-readable confidence label. Used by the dashboard later too.
"""

def get_confidence_label(probability, low_threshold=0.35, high_threshold=0.65):
    """
    probability: model's raw output, 0-1
    Anything between low_threshold and high_threshold is "too close to call"
    """
    if low_threshold <= probability <= high_threshold:
        return "REVIEW RECOMMENDED"
    elif probability > high_threshold:
        return "High confidence: MICROBLEED"
    else:
        return "High confidence: NORMAL"