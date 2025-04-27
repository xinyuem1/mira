import numpy as np
from typing import List, Dict
import re

class RankingSystem:
    """reliability ranking"""
    
    def __init__(self, aggregation_mode: str = 'mean'):
        self.aggregation_mode = aggregation_mode  # mean/aggregation
    
    def stable_ranking(self, model_responses: List[str]) -> List[int]:

        parsed_scores = [self._parse_response(resp) for resp in model_responses]
        
        if self.aggregation_mode == 'mean':

            return self._mean_aggregation(parsed_scores)
        elif self.aggregation_mode == 'median':
            return self._median_aggregation(parsed_scores)
    
    def _parse_response(self, response: str) -> Dict:
        """ranking from output"""
        pattern = r'Ranking:\s*((?:\d+,?\s*)+)'
        match = re.search(pattern, response, re.IGNORECASE)
        if not match:
            return {'scores': [0.5]*5}
        
        order_str = match.group(1).replace(' ', '')
        return {
            'scores': [float(x) for x in re.findall(r'\d+', order_str)],
            'confidence': float(re.search(r'Confidence:\s*([\d.]+)', response).group(1))
        }
    
    def _mean_aggregation(self, scores: List[Dict]) -> List[int]:

        aggregated = np.mean([s['scores'] for s in scores], axis=0)
        return np.argsort(-aggregated).tolist()
    
    def evaluate_aggregation_methods(self, responses: List[str]) -> Dict:

        mean_scores = []
        median_scores = []
        
        for resp in responses:
            parsed = self._parse_response(resp)
            mean_scores.append(parsed['confidence'])
            median_scores.append(np.median(parsed['scores']))
            
        return {
            'mean_accuracy': np.mean(mean_scores),
            'median_accuracy': np.mean(median_scores),
            'agreement_score': self._calculate_agreement(mean_scores, median_scores)
        }
    
    def _calculate_agreement(self, mean_scores, median_scores):

        matches = sum(1 for m, md in zip(mean_scores, median_scores) if m == md)
        return matches / len(mean_scores)