import json
import os
from typing import Dict, Any, List


class Dataset:
    def __init__(self, config: Dict[str, Any]):
        """Initialize dataset with config.
        
        Args:
            config: Configuration dictionary containing dataset settings
        """
        self.config = config
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load dataset from json file based on config.
        
        Returns:
            Dictionary containing the loaded dataset
        """
        task = self.config.get('task', 'gsm8k')
        split = self.config.get('split', 'test')
        
        # Construct file path
        file_path = os.path.join('./data', task, f'{split}.json')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific example by index."""
        if isinstance(idx, int):
            return self.data[str(idx)]
        return self.data[idx]
    
    def __iter__(self):
        """Iterate over dataset examples."""
        for key in self.data.keys():
            yield self.data[key]
    
    def get_examples(self) -> List[Dict[str, Any]]:
        """Get all examples as a list."""
        return list(self.data.values()) 

    def get_prompt_messages(self) -> List[List[Dict[str, str]]]:
        """Get all prompt messages as a list."""
        return [example.get('prompt', []) for example in self.data.values()]