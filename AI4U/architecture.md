# AI4U Architecture Overview

## 1. Client Interfaces
- Voice input
- Text chat
- Offline mode (TFLite models)

## 2. Core Components
- NLP Engine (local / hybrid)
- Intent classification
- Domain routing (health, education, finance…)

## 3. Offline Inference Engine
Uses:
- TensorFlow Lite
- Quantized models
- Minimal RAM footprint

## 4. API Layer
- Flask REST API
- Low-bandwidth response formatting

## 5. Security
- Local-first inference
- No cloud storage
- Privacy-by-design
