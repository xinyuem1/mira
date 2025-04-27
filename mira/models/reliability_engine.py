class ReliableRanker:
    def __init__(self, k=3):
        self.k = k  # stable scoring
        
    def stable_rank(self, model_responses):
        "Implement the stability enhancement mechanism of Weng et al. (2023)
        aggregated = np.zeros((len(model_responses[0]), 3))  # [literal, idiomatic, na]
        
        for i, response in enumerate(model_responses):
            scores = self._response_to_scores(response)
            aggregated[i] = scores
            
        # Weighted voting
        weights = [1/(i+1) for i in range(self.k)] 
        final_scores = aggregated.T @ weights  # Matrix transpose followed by dot multiplication
        
        return final_scores.argsort()[::-1]  # descending order
    
    def _response_to_scores(self, response):
        """Parse model response as a score vector"""
        pattern = r'P=\{literal:\s*([\d.]+),\s*idiomatic:\s*([\d.]+),\s*na:\s*([\d.]+)\}'
        match = re.search(pattern, response)
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]