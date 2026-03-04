# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interfaces for LiteRT LM engines and conversations."""

import abc
import collections.abc
import enum
from typing import Any


class Backend(enum.Enum):
  """Hardware backends for LiteRT-LM."""

  UNSPECIFIED = 0
  CPU = 3
  GPU = 4
  NPU = 6


class AbstractEngine(abc.ABC):
  """Abstract base class for LiteRT-LM engines."""

  def __init__(
      self,
      model_path: str,
      backend: Backend,
      max_num_tokens: int = 512,
      cache_dir: str = "",
  ):
    """Initializes the instance.

    Args:
        model_path: Path to the model file.
        backend: The hardware backend used for inference.
        max_num_tokens: Maximum number of tokens for the KV cache.
        cache_dir: Directory for caching compiled model artifacts.
    """
    self.model_path = model_path
    self.backend = backend
    self.max_num_tokens = max_num_tokens
    self.cache_dir = cache_dir

  def __enter__(self) -> "AbstractEngine":
    """Initializes the engine resources."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the engine resources."""

  @abc.abstractmethod
  def create_conversation(self) -> "AbstractConversation":
    """Creates a new conversation for this engine."""


class AbstractConversation(abc.ABC):
  """Abstract base class for managing GenAI conversations."""

  def __init__(self):
    """Initializes the instance."""

  def __enter__(self) -> "AbstractConversation":
    """Initializes the conversation."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Releases the conversation."""

  @abc.abstractmethod
  def send_message(self, message: str | dict[str, Any]) -> dict[str, Any]:
    """Sends a message and returns the response.

    Args:
        message: The input message to send to the model.

    Returns:
        A dictionary containing the model's response. The structure is:
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    """

  @abc.abstractmethod
  def send_message_async(
      self, message: str | dict[str, Any]
  ) -> collections.abc.Iterator[dict[str, Any]]:
    """Sends a message and streams the response."""
