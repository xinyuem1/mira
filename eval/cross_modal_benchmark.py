import pandas as pd
from sklearn.metrics import accuracy_score

class CrossModalBenchmark:

    def __init__(self, language: str = 'english'):
        self.language = language
        self.metrics = {
            'DCG Score': [],
            'Acc@1': [],
            'Sense Accuracy': []
        }
    
    def evaluate_alignment(self, visual_features, textual_features):

        enhanced_features = self._cross_modal_align(visual_features, textual_features)
        return {
            'dcg_score': self._calculate_dcg(enhanced_features),
            'acc@1': self._calculate_acc1(enhanced_features)
        }
    
    def _cross_modal_align(self, visual, textual):

        return (visual + textual) / 2  # sample
    
    def _calculate_dcg(self, features):

        gains = np.log2(np.arange(1, len(features)+1) + 1)
        return np.sum(features / gains) / np.sum(1 / gains)
    
    def benchmark_language_performance(self, test_data):

        results = []
        for lang in ['english', 'portuguese']:
            lang_data = test_data[test_data['language'] == lang]
            acc1 = accuracy_score(lang_data['true_order'], lang_data['predicted_order'])
            dcg = self._calculate_dcg(lang_data['confidence_scores'])
            
            results.append({
                'language': lang.capitalize(),
                'Acc@1': acc1,
                'DCG Score': dcg
                # ndcg
            })
        
        return pd.DataFrame(results)