import json
from openai import OpenAI
from optimizers.base_optimizer import BaseOptimizer


class LLMOptimizer(BaseOptimizer):
    def __init__(self, benchmarker, openai_api_key):
        super().__init__(benchmarker)
        self.client = OpenAI(api_key=openai_api_key)

    def optimize(self, iterations, **kwargs):
        for iteration in range(iterations):
            prompt = self.generate_prompt(self.history)
            config = None
            while config is None:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a machine learning expert."},
                              {"role": "user", "content": prompt}],
                    temperature=0.8, max_tokens=100, top_p=1)
                try:
                    config = json.loads(response.choices[0].message.content.strip())
                    for parameter_name, parameter_value in config.items():
                        if parameter_value < self.benchmarker.search_space[parameter_name][0] or \
                                parameter_value > self.benchmarker.search_space[parameter_name][1]:
                            config = None
                except json.decoder.JSONDecodeError:
                    # If decoding fails, the loop continues and requests another completion
                    continue

            score = self.benchmarker.evaluate(config)
            self.history.append((iteration, config, score))
        return self.history
