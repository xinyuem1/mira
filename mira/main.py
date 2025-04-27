from data_processing.dataset_loader import DatasetHandler
from model_interaction.zero_shot_inference import ZeroShotInferenceEngine
from multi_step_pipeline import MultiStepImageUnderstanding
from evaluation.reliability import RankingSystem

def main():
    # init
    dataset = DatasetHandler('taskA/train/subtask_a_train.tsv', 
                           'taskA/dev/subtask_a_dev.tsv')
    inference_engine = ZeroShotInferenceEngine(
        api_key='# ur key',
        base_url="#url"
    )
    multi_step = MultiStepImageUnderstanding()
    ranker = RankingSystem(k=3)

   
    results = []
    for compound_id in dataset.dev_df['compound'].unique():
        context = dataset.get_compound_context(compound_id)
        
        # step1 multi-step refine
        refined_caption = multi_step.cross_modal_refinement(
            context['image_features'], 
            context['captions'][0]
        )
        
        # step2 underlying meaning
        latent_meaning = multi_step.latent_meaning_inference(refined_caption)
        
        # step3 scoring
        relation_probs = multi_step.relation_scoring(latent_meaning)
        
        # relevance scoring
        final_ranking = ranker.stable_ranking([relation_probs]*ranker.k)
        
        results.append({
            'compound_id': compound_id,
            'literal_prob': relation_probs[0],
            'idiomatic_prob': relation_probs[1],
            'ranking': final_ranking
        })
    
    # save result
    pd.DataFrame(results).to_csv('final_results.csv', index=False)

if __name__ == "__main__":
    main()