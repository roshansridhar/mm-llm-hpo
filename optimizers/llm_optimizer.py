import json
from openai import OpenAI
from optimizers.base_optimizer import BaseOptimizer


class LLMOptimizer(BaseOptimizer):
    def __init__(self, benchmarker, openai_api_key):
        super().__init__(benchmarker)
        self.client = OpenAI(api_key=openai_api_key)

    def optimize(self, iterations, **kwargs):
        for _ in range(iterations):
            prompt = self.generate_prompt(self.history)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a machine learning expert."}, {"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=150, top_p=1
            )
            config = json.loads(response.choices[0].message.content.strip())
            score = self.benchmarker.evaluate(config)
            self.history.append((config, score))
        return self.history

