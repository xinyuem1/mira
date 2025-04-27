import re

def extract_ranking(response: str) -> list:
    """Extract ranked image IDs from model response"""
    pattern = re.compile(r'Ranking:\s*((\d+,?\s*)+)', re.IGNORECASE)
    match = pattern.search(response)
    
    if match:
        return [int(x) for x in match.group(1).split(',')]
    return []

def calculate_dcg(ranked_list, relevance_scores, k=5):
    """Calculate Discounted Cumulative Gain"""
    return sum(
        (2 ** relevance_scores[i] - 1) / np.log2(i + 2)
        for i in range(min(k, len(ranked_list)))
    )