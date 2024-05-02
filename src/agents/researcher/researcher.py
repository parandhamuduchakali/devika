import json
from typing import List

from jinja2 import Environment, BaseLoader

from src.llm import LLM
from src.services.utils import retry_wrapper, validate_responses
from src.browser.search import BingSearch
from typing import Union, Dict,List


PROMPT = open("src/agents/researcher/prompt.jinja2").read().strip()


class Researcher:
    def __init__(self, base_model: str):
        self.bing_search = BingSearch()
        self.llm = LLM(model_id=base_model)

    def render(self, step_by_step_plan: str, contextual_keywords: str) -> str:
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT)
        return template.render(
            step_by_step_plan=step_by_step_plan,
            contextual_keywords=contextual_keywords
        )

    @validate_responses
    def validate_response(self, response: str) -> Union[Dict, bool]:
        if "queries" not in response and "ask_user" not in response:
            return False
        else:
            return {
                "queries": response.get("queries", None),  # Added safety with .get()
                "ask_user": response.get("ask_user", None)  # Added safety with .get()
            }
        
    @retry_wrapper
    def execute(self, step_by_step_plan: str, contextual_keywords: List[str], project_name: str) -> Union[dict, bool]:
        contextual_keywords_str = ", ".join(map(lambda k: k.capitalize(), contextual_keywords))
        prompt = self.render(step_by_step_plan, contextual_keywords_str)
        
        response = self.llm.inference(prompt, project_name)
        
        valid_response = self.validate_response(response)

        return valid_response
