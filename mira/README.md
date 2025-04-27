# MIRA (Multimodal Idiom Recognition and Alignment)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation for the SemEval-2025 Task 1 (AdMIRe) system described in the paper *"PALI-NLP at SemEval-2025 Task 1: AdMIRe- Advancing Multimodal Idiomaticity Representation"*

## Overview

MIRA is a training-free framework for multimodal idiom understanding that combines:
- **In-context learning** for bias correction
- **Multi-step semantic-visual fusion** for fine-grained alignment
- **Self-consistency reasoning** for robust outputs

This implementation focuses on Subtask A: Ranking candidate images based on their relevance to idiomatic/literal interpretations of nominal compounds.

## Features

- 🚀 Training-free approach leveraging LLMs (GPT-4o)
- 📊 Multimodal alignment of text and image features
- 🔄 Self-consistency verification for stable rankings
- 🌍 Cross-lingual support (English/Portuguese)
- ⚡ Parallel processing capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mira.git
cd mira