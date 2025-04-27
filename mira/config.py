from dataclasses import dataclass

@dataclass
class MiraConfig:
    # Data configuration
    train_path: str = "taskA/train/subtask_a_train.tsv"
    dev_path: str = "taskA/dev/subtask_a_dev.tsv"
    test_path: str = "taskA/test/subtask_a_test.tsv"
    
    # Model configuration
    model_name: str = "gpt-4o"
    api_key: str = "sk-your-key-here"
    base_url: str = "https://api.openai.com/v1"
    
    # Processing parameters
    max_workers: int = 40
    num_retry: int = 3
    save_freq: int = 10
    top_p: float = 0.01
    
