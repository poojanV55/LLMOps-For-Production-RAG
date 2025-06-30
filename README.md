# 🚀 LLMOps for Production RAG Workshop

## Overview

This workshop provides a comprehensive introduction to building and optimizing Retrieval-Augmented Generation (RAG) pipelines for production environments using Union Serverless. The workshop covers the entire lifecycle of RAG systems from baseline implementation to advanced optimization techniques.

## 📋 Workshop Structure

### 1. **Environment Setup & Dependencies**
- **Installation**: Uses `gradio` and Union Serverless platform
- **Authentication**: Union Serverless account creation and device-flow authentication
- **API Configuration**: OpenAI API key setup and secret management
- **Platform Access**: Union Serverless dashboard integration

### 2. **Baseline RAG Pipeline Creation**

#### Vector Store Creation
- **Workflow**: `create_vector_store()` function with configurable parameters
- **Features**:
  - Document chunking with customizable chunk sizes (default: 2048)
  - Multiple splitter options (character, recursive)
  - Embedding model selection (default: text-embedding-ada-002)
  - Document filtering with exclude patterns
  - Configurable document limits

#### Basic RAG Implementation
- **Workflow**: `rag_basic()` function for question-answering
- **Components**:
  - Document retrieval with similarity search
  - Optional reranking capabilities
  - Configurable number of retrieved documents (default: 20)
  - Final document selection (default: 5)
  - Customizable prompt templates
  - Multiple generation models support (GPT-4o-mini, etc.)

#### Vector Store Maintenance
- **Scheduled Updates**: Launch plans with fixed-rate scheduling (every 2 minutes)
- **Automated Refresh**: Continuous vector store updates
- **Resource Management**: Proper cleanup and deactivation procedures

#### Interactive Interface
- **Gradio App**: Real-time chat interface for RAG queries
- **Live Execution**: Union Serverless integration with execution tracking
- **User Experience**: Streaming responses with proper error handling

### 3. **Evaluation Dataset Bootstrapping**

#### Question-Answer Generation
- **Workflow**: `create_qa_dataset()` function
- **Features**:
  - Multiple questions per document (configurable)
  - Multiple answers per question (default: 5)
  - Parallel processing with `union.map_task()`
  - Structured dataset creation

#### LLM-Based Filtering
- **Workflow**: `create_llm_filtered_dataset()` function
- **Quality Control**:
  - LLM critic for answer quality assessment
  - Score-based filtering
  - Dataset preparation for evaluation

### 4. **RAG Hyperparameter Optimization**

#### Grid Search Implementation
- **Workflow**: `optimize_rag()` function
- **Optimization Areas**:
  - **Embedding Models**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
  - **Chunk Sizes**: Various document chunking strategies
  - **Splitters**: Character-based vs recursive splitting
  - **Search Parameters**: Retrieval configuration optimization
  - **Reranking**: Optional reranking pipeline integration
  - **Prompts**: Template optimization

#### Experiment Configurations
The workshop includes several pre-configured experiment files:
- `config/embedding_model_experiment.yaml`: Embedding model comparison
- `config/prompt_experiment.yaml`: Prompt template optimization
- `config/chunksize_experiment.yaml`: Chunk size impact analysis
- `config/splitter_experiment.yaml`: Document splitting strategies
- `config/reranking_experiment.yaml`: Reranking effectiveness
- `config/search_params_experiment.yaml`: Retrieval parameter tuning

### 5. **Reactive Pipeline Integration**

#### Automated Workflow
- **Knowledge Base Updates**: Scheduled document processing
- **Trigger-Based Evaluation**: Automatic dataset creation on knowledge base updates
- **Optimization Triggers**: HPO execution on new evaluation datasets
- **Artifact Management**: Union artifacts for workflow coordination

#### Launch Plan Orchestration
- **Scheduled Execution**: Fixed-rate scheduling for knowledge base updates
- **Event-Driven Triggers**: Artifact-based workflow activation
- **Resource Management**: Proper activation/deactivation procedures

## 🛠️ Technical Components

### Core Libraries & Frameworks
- **Union Serverless**: Cloud-native workflow orchestration
- **Flyte**: Workflow engine integration
- **Gradio**: Interactive web interface
- **Pandas**: Data manipulation and analysis
- **OpenAI**: LLM integration for generation and evaluation

### Key Features
- **Scalable Architecture**: Cloud-native design with Union Serverless
- **Modular Design**: Reusable components and workflows
- **Evaluation Framework**: Comprehensive RAG assessment
- **Automated Optimization**: Hyperparameter tuning with grid search
- **Production Ready**: Scheduling, monitoring, and error handling

## 🎯 Learning Objectives

By completing this workshop, participants will:

1. **Understand RAG Fundamentals**: Build a complete RAG pipeline from scratch
2. **Master LLMOps Practices**: Implement production-ready workflows with proper scheduling and monitoring
3. **Optimize Performance**: Learn hyperparameter optimization techniques for RAG systems
4. **Evaluate Quality**: Implement comprehensive evaluation frameworks using LLM-as-a-judge
5. **Deploy at Scale**: Use Union Serverless for scalable, cloud-native RAG deployments

## 📊 Workshop Outcomes

### Deliverables
- **Baseline RAG Pipeline**: Functional question-answering system
- **Evaluation Dataset**: Quality-assessed Q&A pairs for testing
- **Optimized Configuration**: Best-performing RAG parameters
- **Interactive Interface**: Gradio-based chat application
- **Automated Workflows**: Scheduled and trigger-based pipeline components

### Skills Acquired
- Union Serverless workflow development
- RAG system architecture and optimization
- LLM-based evaluation and filtering
- Hyperparameter optimization for NLP systems
- Production deployment and monitoring

## 🚀 Getting Started

1. **Prerequisites**: Union Serverless account, OpenAI API key
2. **Installation**: Follow the dependency installation steps in the notebook
3. **Authentication**: Complete Union Serverless login and API key setup
4. **Execution**: Run through the workshop sections sequentially
5. **Customization**: Modify parameters and experiment with different configurations

## 📈 Advanced Topics

The workshop also covers advanced concepts including:
- **Reactive Pipelines**: Event-driven workflow orchestration
- **Artifact Management**: Union artifacts for data lineage
- **Resource Optimization**: CPU and memory allocation strategies
- **Monitoring & Debugging**: Execution tracking and error handling
- **Scalability**: Cloud-native deployment considerations

