"""
LLM provider management for multi-provider support.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProviderStatus:
    """Status information for an LLM provider."""
    available: bool
    error_message: Optional[str] = None
    model: Optional[str] = None
    initialized: bool = False

class LLMProviderManager:
    """Manager for multiple LLM providers with fallback support."""
    
    def __init__(self):
        """Initialize LLM provider manager."""
        self.providers = {}
        self.current_provider = None
        self.provider_status = {}
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
        
        # Initialize all available providers
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all available LLM providers."""
        # Try to initialize Anthropic
        try:
            self.initialize_anthropic()
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic: {e}")
        
        # Try to initialize OpenAI
        try:
            self.initialize_openai()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Try to initialize Google
        try:
            self.initialize_google()
        except Exception as e:
            logger.warning(f"Failed to initialize Google: {e}")
        
        # Set default provider
        if self.default_provider in self.providers:
            self.current_provider = self.default_provider
            logger.info(f"Set default provider to: {self.default_provider}")
        elif self.providers:
            # Use first available provider
            self.current_provider = list(self.providers.keys())[0]
            logger.info(f"Set fallback provider to: {self.current_provider}")
        else:
            logger.error("No LLM providers available!")
    
    def initialize_anthropic(self, api_key: Optional[str] = None) -> Any:
        """
        Initialize Anthropic Claude provider.
        
        Args:
            api_key: Anthropic API key (optional, uses environment variable if None)
            
        Returns:
            Initialized ChatAnthropic instance
        """
        try:
            from langchain_anthropic import ChatAnthropic
            
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            
            model = os.getenv("LLM_MODEL_ANTHROPIC", "claude-3-5-sonnet-20241022")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
            max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
            
            provider = ChatAnthropic(
                anthropic_api_key=api_key,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Test the provider
            test_response = provider.invoke("Hello")
            
            self.providers["anthropic"] = provider
            self.provider_status["anthropic"] = ProviderStatus(
                available=True,
                model=model,
                initialized=True
            )
            
            logger.info(f"Anthropic provider initialized with model: {model}")
            return provider
            
        except ImportError:
            error_msg = "langchain_anthropic not installed"
            self.provider_status["anthropic"] = ProviderStatus(
                available=False,
                error_message=error_msg
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to initialize Anthropic: {e}"
            self.provider_status["anthropic"] = ProviderStatus(
                available=False,
                error_message=error_msg
            )
            logger.error(error_msg)
            raise
    
    def initialize_openai(self, api_key: Optional[str] = None) -> Any:
        """
        Initialize OpenAI GPT provider.
        
        Args:
            api_key: OpenAI API key (optional, uses environment variable if None)
            
        Returns:
            Initialized ChatOpenAI instance
        """
        try:
            from langchain_openai import ChatOpenAI
            
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            model = os.getenv("LLM_MODEL_OPENAI", "gpt-4o")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
            max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
            
            provider = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Test the provider
            test_response = provider.invoke("Hello")
            
            self.providers["openai"] = provider
            self.provider_status["openai"] = ProviderStatus(
                available=True,
                model=model,
                initialized=True
            )
            
            logger.info(f"OpenAI provider initialized with model: {model}")
            return provider
            
        except ImportError:
            error_msg = "langchain_openai not installed"
            self.provider_status["openai"] = ProviderStatus(
                available=False,
                error_message=error_msg
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI: {e}"
            self.provider_status["openai"] = ProviderStatus(
                available=False,
                error_message=error_msg
            )
            logger.error(error_msg)
            raise
    
    def initialize_google(self, api_key: Optional[str] = None) -> Any:
        """
        Initialize Google Gemini provider.
        
        Args:
            api_key: Google API key (optional, uses environment variable if None)
            
        Returns:
            Initialized ChatGoogleGenerativeAI instance
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            if api_key is None:
                api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError("Google API key not provided")
            
            model = os.getenv("LLM_MODEL_GOOGLE", "gemini-1.5-pro")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
            max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
            
            provider = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Test the provider
            test_response = provider.invoke("Hello")
            
            self.providers["google"] = provider
            self.provider_status["google"] = ProviderStatus(
                available=True,
                model=model,
                initialized=True
            )
            
            logger.info(f"Google provider initialized with model: {model}")
            return provider
            
        except ImportError:
            error_msg = "langchain_google_genai not installed"
            self.provider_status["google"] = ProviderStatus(
                available=False,
                error_message=error_msg
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to initialize Google: {e}"
            self.provider_status["google"] = ProviderStatus(
                available=False,
                error_message=error_msg
            )
            logger.error(error_msg)
            raise
    
    def get_provider(self, provider_name: Optional[str] = None) -> Any:
        """
        Get LLM provider instance.
        
        Args:
            provider_name: Name of provider to get (uses current if None)
            
        Returns:
            LLM provider instance
        """
        if provider_name is None:
            provider_name = self.current_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        return self.providers[provider_name]
    
    def switch_provider(self, provider_name: str) -> bool:
        """
        Switch to a different LLM provider.
        
        Args:
            provider_name: Name of provider to switch to
            
        Returns:
            True if switch successful, False otherwise
        """
        if provider_name not in self.providers:
            logger.error(f"Provider '{provider_name}' not available")
            return False
        
        self.current_provider = provider_name
        logger.info(f"Switched to provider: {provider_name}")
        return True
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.
        
        Returns:
            List of available provider names
        """
        return list(self.providers.keys())
    
    def test_all_providers(self) -> Dict[str, bool]:
        """
        Test all initialized providers.
        
        Returns:
            Dictionary mapping provider names to test results
        """
        results = {}
        
        for provider_name, provider in self.providers.items():
            try:
                response = provider.invoke("Test message")
                results[provider_name] = True
                logger.info(f"Provider '{provider_name}' test: PASSED")
            except Exception as e:
                results[provider_name] = False
                logger.error(f"Provider '{provider_name}' test: FAILED - {e}")
        
        return results
    
    def get_provider_capabilities(self, provider: str) -> Dict[str, Any]:
        """
        Get capabilities and information for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with provider capabilities
        """
        if provider not in self.provider_status:
            return {"available": False, "error": "Provider not found"}
        
        status = self.provider_status[provider]
        capabilities = {
            "available": status.available,
            "initialized": status.initialized,
            "model": status.model,
            "error_message": status.error_message
        }
        
        # Add provider-specific capabilities
        if provider == "anthropic":
            capabilities.update({
                "supports_tools": True,
                "supports_streaming": True,
                "max_context": 200000,
                "supports_vision": True
            })
        elif provider == "openai":
            capabilities.update({
                "supports_tools": True,
                "supports_streaming": True,
                "max_context": 128000,
                "supports_vision": True
            })
        elif provider == "google":
            capabilities.update({
                "supports_tools": True,
                "supports_streaming": True,
                "max_context": 1000000,
                "supports_vision": True
            })
        
        return capabilities
    
    def compare_providers_response(self, message: str) -> Dict[str, str]:
        """
        Get responses from all available providers for comparison.
        
        Args:
            message: Message to send to all providers
            
        Returns:
            Dictionary mapping provider names to responses
        """
        responses = {}
        
        for provider_name, provider in self.providers.items():
            try:
                response = provider.invoke(message)
                responses[provider_name] = response.content if hasattr(response, 'content') else str(response)
                logger.info(f"Got response from {provider_name}")
            except Exception as e:
                responses[provider_name] = f"Error: {e}"
                logger.error(f"Failed to get response from {provider_name}: {e}")
        
        return responses
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary of all providers.
        
        Returns:
            Status summary dictionary
        """
        summary = {
            "current_provider": self.current_provider,
            "total_providers": len(self.providers),
            "available_providers": list(self.providers.keys()),
            "provider_status": self.provider_status,
            "default_provider": self.default_provider
        }
        
        return summary
    
    def reinitialize_provider(self, provider_name: str) -> bool:
        """
        Reinitialize a specific provider.
        
        Args:
            provider_name: Name of provider to reinitialize
            
        Returns:
            True if reinitialization successful, False otherwise
        """
        try:
            if provider_name == "anthropic":
                self.initialize_anthropic()
            elif provider_name == "openai":
                self.initialize_openai()
            elif provider_name == "google":
                self.initialize_google()
            else:
                logger.error(f"Unknown provider: {provider_name}")
                return False
            
            logger.info(f"Provider '{provider_name}' reinitialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reinitialize provider '{provider_name}': {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup all providers."""
        self.providers.clear()
        self.provider_status.clear()
        self.current_provider = None
        logger.info("All providers cleaned up") 