#!/usr/bin/env python3
"""
Ollama API Client
Interact with Ollama API programmatically
"""

import requests
import json
import sys
import argparse
from typing import Generator, Dict, Any

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate(self, model: str, prompt: str, stream: bool = True, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Generate text from a model"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            yield response.json()
    
    def chat(self, model: str, messages: list, stream: bool = True, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Chat with a model"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()
        
        if stream:
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            yield response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def show_model(self, model: str) -> Dict[str, Any]:
        """Show model information"""
        url = f"{self.base_url}/api/show"
        payload = {"name": model}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def pull_model(self, model: str) -> Generator[Dict[str, Any], None, None]:
        """Pull a model"""
        url = f"{self.base_url}/api/pull"
        payload = {"name": model}
        
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
    
    def create_model(self, name: str, modelfile: str) -> Generator[Dict[str, Any], None, None]:
        """Create a custom model from a Modelfile"""
        url = f"{self.base_url}/api/create"
        payload = {
            "name": name,
            "modelfile": modelfile
        }
        
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
    
    def delete_model(self, model: str) -> Dict[str, Any]:
        """Delete a model"""
        url = f"{self.base_url}/api/delete"
        payload = {"name": model}
        response = requests.delete(url, json=payload)
        response.raise_for_status()
        return {"status": "deleted", "model": model}
    
    def embeddings(self, model: str, prompt: str) -> Dict[str, Any]:
        """Generate embeddings"""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": prompt
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def interactive_chat(client: OllamaClient, model: str):
    """Interactive chat mode"""
    print(f"💬 Chat with {model} (type 'exit' to quit)")
    print("-" * 40)
    
    messages = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print(f"\n{model}: ", end="", flush=True)
        
        full_response = ""
        for chunk in client.chat(model, messages, stream=True):
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                full_response += content
        
        messages.append({"role": "assistant", "content": full_response})
        print()  # New line after response


def main():
    parser = argparse.ArgumentParser(description="Ollama API Client")
    parser.add_argument("--model", default="gpt-oss:latest", help="Model to use")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama API URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("prompt", help="Prompt text")
    gen_parser.add_argument("--temperature", type=float, default=0.7)
    gen_parser.add_argument("--max-tokens", type=int, default=500)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List models")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show model info")
    
    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull a model")
    pull_parser.add_argument("model_name", help="Model to pull")
    
    # Embeddings command
    embed_parser = subparsers.add_parser("embeddings", help="Generate embeddings")
    embed_parser.add_argument("text", help="Text to embed")
    
    args = parser.parse_args()
    
    client = OllamaClient(args.url)
    
    try:
        if args.command == "generate":
            print(f"Generating with {args.model}...")
            for chunk in client.generate(
                args.model, 
                args.prompt,
                temperature=args.temperature,
                num_predict=args.max_tokens
            ):
                if "response" in chunk:
                    print(chunk["response"], end="", flush=True)
            print()
            
        elif args.command == "chat":
            interactive_chat(client, args.model)
            
        elif args.command == "list":
            models = client.list_models()
            print("Available models:")
            for model in models.get("models", []):
                size = model.get("size", 0) / (1024**3)  # Convert to GB
                print(f"  • {model['name']} ({size:.1f} GB)")
                
        elif args.command == "show":
            info = client.show_model(args.model)
            print(json.dumps(info, indent=2))
            
        elif args.command == "pull":
            print(f"Pulling {args.model_name}...")
            for status in client.pull_model(args.model_name):
                if "status" in status:
                    print(f"  {status['status']}")
            print("✓ Pull complete")
            
        elif args.command == "embeddings":
            result = client.embeddings(args.model, args.text)
            embedding = result.get("embedding", [])
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 10 values: {embedding[:10]}")
            
        else:
            parser.print_help()
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure Ollama is running (ollama serve)", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()