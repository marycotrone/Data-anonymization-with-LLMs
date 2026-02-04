"""
LLM-based anonymizer using Ollama.

Supports various models: gemma2, llama3, mistral, phi3, etc.
"""

import re
import requests
from typing import List, Optional, Dict, Any
from tqdm.auto import tqdm
import time


# Template di prompt predefiniti
PROMPT_TEMPLATES = {
    "paraphrase": {
        "system": """Act as a data editor and privacy expert. Your PRIMARY task is to rephrase and anonymize the input sentence.
1. REPHRASING: Rephrase sentences using different words keeping the same sentiment.
2. STYLE PRESERVATION: You MUST KEEP the original style (informal, slang, punctuation). Do NOT make it formal.
3. ANONYMIZATION: REPLACE EACH entity (People, Locations, Brands) with generic placeholders or alternatives.
Output ONLY the rewritten sentence.""",
        "user": """Original: "{text}"
Sentiment: {label}
Task: Rewrite the sentence above. Ensure the output is grammatically correct and preserves the {label} sentiment.
Output ONLY the rewritten sentence."""
    },
    "simple": {
        "system": """You are an expert in text anonymization. Your task is to anonymize the given text by replacing named entities (people, locations, organizations) with similar but different entities, while preserving the semantic meaning and structure of the text.""",
        "user": """Anonymize the following text:

"{text}"

Return ONLY the anonymized text without any explanations."""
    },
    "strict": {
        "system": """You are a privacy expert. Your ONLY task is to replace personal identifiable information (PII) in text.
RULES:
- Replace names of people with different names
- Replace locations with different locations
- Replace organization names with different ones
- Keep the same sentence structure
- Do NOT add explanations""",
        "user": """Text to anonymize: "{text}"

Anonymized text:"""
    }
}


class OllamaAnonymizer:
    """Class for anonymization using LLM via Ollama."""
    
    def __init__(
        self,
        model_name: str = "gemma2:2b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_tokens: int = 256,
        prompt_style: str = "paraphrase",
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the anonymizer with Ollama.
        
        Args:
            model_name: Ollama model name (gemma2:2b, llama3.2, mistral, phi3, etc.)
            base_url: Ollama server URL
            temperature: Generation temperature (0.0-1.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            max_tokens: Maximum number of output tokens
            prompt_style: Prompt style ("paraphrase", "simple", "strict")
            system_prompt: Custom system prompt (override)
            user_prompt_template: Custom user template (override)
            verbose: Show status messages
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Setup prompts
        if prompt_style in PROMPT_TEMPLATES:
            template = PROMPT_TEMPLATES[prompt_style]
            self.system_prompt = system_prompt or template["system"]
            self.user_prompt_template = user_prompt_template or template["user"]
        else:
            self.system_prompt = system_prompt or PROMPT_TEMPLATES["simple"]["system"]
            self.user_prompt_template = user_prompt_template or PROMPT_TEMPLATES["simple"]["user"]
        
        # Check connection
        if self.verbose:
            self._check_connection()
    
    def _check_connection(self) -> bool:
        """
        Verify connection to Ollama server.
        
        Returns:
            True if connection is OK
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.verbose:
                    print(f"✅ Connected to Ollama ({self.base_url})")
                    print(f"   Available models: {', '.join(model_names[:5])}" + 
                          ("..." if len(model_names) > 5 else ""))
                
                # Check if model is available (considering tags like :latest)
                model_base = self.model_name.split(':')[0]
                available = any(model_base in m for m in model_names)
                
                if not available:
                    print(f"⚠️  Model '{self.model_name}' not found")
                    print(f"   Download with: ollama pull {self.model_name}")
                else:
                    print(f"   Selected model: {self.model_name}")
                
                return True
            else:
                print(f"❌ Ollama connection error: {response.status_code}")
                return False
        
        except requests.exceptions.ConnectionError:
            print("❌ Unable to connect to Ollama")
            print(f"   Make sure Ollama is running on {self.base_url}")
            print("   Start with: ollama serve")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _clean_response(self, text: str) -> str:
        """
        Clean the model's response.
        
        Args:
            text: Raw response
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove common courtesy phrases
        garbage_phrases = [
            "Let me know if you have another example",
            "Let me know if you'd like",
            "Let me know if you want",
            "Let me know if you need",
            "Here is the anonymized",
            "Here's the anonymized",
            "Here is the rewritten",
            "Here's the rewritten",
        ]
        
        for phrase in garbage_phrases:
            if phrase.lower() in text.lower():
                # Find and remove the phrase and everything that follows
                idx = text.lower().find(phrase.lower())
                text = text[:idx]
        
        # Remove common explanation patterns
        patterns = [
            r"\*\*?Explanation:?\*\*?.*$",
            r"\n*Let me know if.*$",
            r"\n*Note:.*$",
            r"\n*I changed.*$",
            r"\n*Changes made:.*$",
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove surrounding quotes
        text = text.strip()
        while (text.startswith('"') and text.endswith('"')) or \
              (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        # Normalize spaces
        text = " ".join(text.split())
        
        return text
    
    def _generate(self, prompt: str, retry: int = 3) -> Optional[str]:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: Prompt to send
            retry: Number of retry attempts on error
            
        Returns:
            Generated response or None
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": self.top_p,
                "repeat_penalty": 1.1
            }
        }
        
        for attempt in range(retry):
            try:
                response = requests.post(url, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    raw_response = result.get("response", "").strip()
                    return self._clean_response(raw_response)
                else:
                    if self.verbose:
                        print(f"⚠️  API error (attempt {attempt + 1}/{retry}): {response.status_code}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            except requests.exceptions.Timeout:
                if self.verbose:
                    print(f"⚠️  Timeout (attempt {attempt + 1}/{retry})")
                time.sleep(2 ** attempt)
            
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error (attempt {attempt + 1}/{retry}): {e}")
                time.sleep(2 ** attempt)
        
        return None
    
    def anonymize(
        self,
        sentence: str,
        label: Optional[str] = None
    ) -> Optional[str]:
        """
        Anonymize a single sentence.
        
        Args:
            sentence: Sentence to anonymize
            label: Sentiment label (optional, for contextual prompts)
            
        Returns:
            Anonymized sentence or None if error
        """
        # Format the prompt
        if label and "{label}" in self.user_prompt_template:
            prompt = self.user_prompt_template.format(text=sentence, label=label)
        else:
            prompt = self.user_prompt_template.format(text=sentence)
        
        return self._generate(prompt)
    
    def anonymize_batch(
        self,
        sentences: List[str],
        labels: Optional[List[str]] = None,
        show_progress: bool = True,
        skip_on_error: bool = True
    ) -> List[str]:
        """
        Anonymize a batch of sentences.
        
        Args:
            sentences: List of sentences to anonymize
            labels: List of labels (optional)
            show_progress: Show progress bar
            skip_on_error: Keep original sentence if there's an error
            
        Returns:
            List of anonymized sentences
        """
        anonymized = []
        
        if labels is None:
            labels = [None] * len(sentences)
        
        iterator = zip(sentences, labels)
        if show_progress:
            iterator = tqdm(list(iterator), desc="LLM Anonymization")
        
        for sentence, label in iterator:
            result = self.anonymize(sentence, label)
            
            if result is None or result.strip() == "":
                if skip_on_error:
                    anonymized.append(sentence)  # Keep original
                else:
                    anonymized.append("")
            else:
                anonymized.append(result)
        
        return anonymized
    
    def get_available_models(self) -> List[str]:
        """
        Get the list of available models on Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
        except Exception:
            pass
        return []
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving model info: {e}")
        
        return None
    
    def set_model(self, model_name: str) -> None:
        """
        Change the model to use.
        
        Args:
            model_name: Name of the new model
        """
        self.model_name = model_name
        if self.verbose:
            print(f"📦 Model changed to: {model_name}")
            self._check_connection()
    
    def set_temperature(self, temperature: float) -> None:
        """
        Change the generation temperature.
        
        Args:
            temperature: New temperature (0.0-1.0)
        """
        self.temperature = max(0.0, min(1.0, temperature))
        if self.verbose:
            print(f"🌡️  Temperature set to: {self.temperature}")


def create_anonymizer_from_config(config: Dict[str, Any]) -> OllamaAnonymizer:
    """
    Create an OllamaAnonymizer instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured OllamaAnonymizer instance
    """
    llm_config = config.get("llm", {})
    
    return OllamaAnonymizer(
        model_name=llm_config.get("model_name", "gemma2:2b"),
        base_url=llm_config.get("base_url", "http://localhost:11434"),
        temperature=llm_config.get("temperature", 0.4),
        max_tokens=llm_config.get("max_tokens", 256),
        system_prompt=llm_config.get("system_prompt"),
        user_prompt_template=llm_config.get("user_prompt_template"),
        verbose=config.get("general", {}).get("show_progress", True)
    )
