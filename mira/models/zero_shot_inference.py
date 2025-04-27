from openai import OpenAI
import re

class ZeroShotInferenceEngine:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def generate_interpretation(self, compound, context_type):
        """执行零样本推理"""
        prompt = "Can you to analyze compound \"{compound}\" and provide its \"{type}\" meanings. \
When offering definitions, please use reliable dictionary sources such as the Cambridge Dictionary or Collins Dictionary for reference. \
Please analyze what is the subject that this meaning is most likely to describe. Like human/man/animals.\
Ensure your explanation is professional, short, clear and do not include any \"{another_type}\" meaning.\
Do not inculde \"{compound}\" in output. Output in format: {{literal: bool, explanation: str}} "

        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return self._parse_response(response.choices[0].message.content)
    
    def _parse_response(self, text):
        match = re.search(r'{{literal:\s*(true|false),\s*explanation:\s*(.*?)}}', text, re.DOTALL)
        return {
            'literal': match.group(1).lower() == 'true',
            'explanation': match.group(2).strip()
        }