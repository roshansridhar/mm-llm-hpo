import json
from openai import OpenAI
from optimizers.base_optimizer import BaseOptimizer
from loguru import logger
import re

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
                    model="gpt-4-turbo",
                    messages=[{"role": "system", "content": "You are a machine learning expert."},
                              {"role": "user", "content": prompt}],
                    temperature=0, max_tokens=2000, top_p=1)
                response_content = response.choices[0].message.content.strip()
                try:
                    config = json.loads(response_content)
                except json.decoder.JSONDecodeError:
                    try:
                        config = json.loads(
                            re.search(r'\{(.*?)\}', response_content,
                                      re.DOTALL).group())
                        for parameter_name, parameter_value in config.items():
                            if parameter_value < self.benchmarker.search_space[parameter_name][0] or \
                                    parameter_value > self.benchmarker.search_space[parameter_name][1]:
                                logger.debug(f"config is outside of the search space. Request again. Config: {config}")
                                config = None
                                break
                    except:
                        logger.debug(f"response is not json format. Request again. Response: {response_content}")
                        continue  # If decoding fails, retry the request
            print(prompt)
            print(response_content)
            score = self.benchmarker.evaluate(config)
            self.history.append((iteration, config, score))
        return self.history
