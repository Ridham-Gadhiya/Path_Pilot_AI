def compute_final_score(similarity, career_score, skill_score):
    """
    Combine different signals into final score
    """
    return (
        0.5 * similarity +
        0.3 * career_score +
        0.2 * skill_score
    )