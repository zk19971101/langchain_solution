from typing import Optional, List, Any, Mapping
from langchain_core.callbacks import CallbackManagerForLLMRun
from openai import OpenAI
from langchain.llms.base import LLM

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate

prompt_list = []


class CustomLLM(LLM):
    endpoint = "http://127.0.0.1:8080/v1"
    model = "llama3"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = OpenAI(
            base_url=self.endpoint,
            api_key="123456"
        )

        prompt_list.append({
                'role': 'user',
                'content': f"{prompt}"
            })

        chat_completion = client.chat.completions.create(
            messages=prompt_list,
            model="llama3",
            stream=False
        )
        replay = chat_completion.choices[0].message.content
        return replay



    @property
    def _llm_type(self) -> str:
        return self.model

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"endpoint": self.endpoint, "model": self.model}


def build_model_huggingface(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        top_p=1,
        repetition_penalty=1.15
    )
    llama_model = HuggingFacePipeline(pipeline=pipe)
    return llama_model


if __name__ == '__main__':
    llm = CustomLLM()
    print(llm("hello"))