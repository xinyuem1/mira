import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class PerformanceAnalyzer:

    
    def plot_cross_modal_impact(self, english_data, portuguese_data):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # DCG Score
        ax1.plot(english_data['dcg'], label='English', marker='o')
        ax1.plot(portuguese_data['dcg'], label='Portuguese', marker='x')
        ax1.set_title('DCG Score Comparison')
        ax1.set_ylabel('Normalized DCG')
        

        ax2.bar(['English', 'Portuguese'], 
                [english_data['acc1'].mean(), portuguese_data['acc1'].mean()])
        ax2.set_title('Acc@1 Performance')
        
        plt.tight_layout()
        plt.savefig('cross_modal_comparison.png')
    
    def visualize_pipeline(self):

        steps = ['Input', 'Cross-Modal Refinement', 
                'Latent Inference', 'Relation Scoring']
        
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.axis('off')

        for i, step in enumerate(steps):
            color = '#4CAF50' if i % 2 == 0 else '#FF5722'
            ax.text(0.5, 0.8 - i*0.2, step, 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor=color, edgecolor='black'))
        

        for i in range(len(steps)-1):
            ax.annotate("", xy=(0.5, 0.6 - i*0.2), 
                       xytext=(0.5, 0.4 - i*0.2),
                       arrowprops=dict(arrowstyle="->"))
        
        plt.savefig('pipeline_visualization.png')