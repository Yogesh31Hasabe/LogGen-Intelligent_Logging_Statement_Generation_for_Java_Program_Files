# LogGen: Intelligent Logging Statement Generation for Java Program Files

![License](https://img.shields.io/badge/license-MIT-blue)
![Language](https://img.shields.io/badge/language-Java%20|%20Python-brightgreen)

## 📖 Overview

LogGen is an intelligent solution for generating logging statements in Java programs. Built using advanced **Large Language Models (LLMs)**, LogGen identifies optimal logging positions and generates high-quality, contextually relevant log messages without altering the underlying code structure.

By improving logging practices, LogGen enhances **software maintainability, observability**, and **developer productivity**, addressing challenges like insufficient or overwhelming logging.

---

## 🚀 Features

- **Automated Logging**: Predicts "what", "where", and "how" to log efficiently.
- **Context-Aware Statements**: Generates logs relevant to runtime data, improving debugging and system monitoring.
- **Advanced LLM Integration**: Leverages state-of-the-art models like **CodeT5** and **LLaMA**.
- **High Performance**: Outperforms existing solutions with superior BLEU and ROUGE scores.

---

## 🏗️ Project Architecture

LogGen employs a two-stage pipeline:

1. **Stage 1: Logging Position Prediction**
   - Uses an LLM to identify optimal positions for logging.
   - Annotates code with `<FILL_ME>` placeholders.

2. **Stage 2: Logging Statement Generation**
   - Replaces placeholders with semantically relevant log statements.
   - Ensures clarity, brevity, and contextual accuracy.

---

## 📊 Evaluation Metrics

LogGen achieves state-of-the-art performance using:
- **Stage 1**: Accuracy for predicting logging positions.
- **Stage 2**: BLEU & ROUGE scores to evaluate log quality and relevance.

| Metric         | Score   |
|----------------|---------|
| **BLEU**       | 0.6464  |
| **ROUGE-1**    | 0.8859  |
| **ROUGE-L**    | 0.8262  |

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- Java Development Kit (JDK) 8+
- Dependencies listed in `requirement.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LogGen-Intelligent_Logging_Statement_Generation_for_Java_Program_Files.git
   cd LogGen-Intelligent_Logging_Statement_Generation_for_Java_Program_Files
   ```

2. Install dependencies:
   ```bash
     pip install -r requirement.txt
  
## 🔍 Usage

### Dataset Preparation
- Use the included dataset generation script to process Java repositories.

### Train the Model
- Fine-tune Stage 1 and Stage 2 models using the provided training pipeline.

### Run Project 
 ```bash
     python3 stage1.py
```

```bash
     python3 stage2.py
```

### Output
- Logs are automatically generated and inserted into the specified positions.

---

## 🧪 Experimental Results

- LogGen achieves **36.84% accuracy** in predicting file-level logging positions.
- Generated logs surpass existing methods with BLEU and ROUGE metrics significantly exceeding competitors like **FastLog**.

| Model        | BLEU   | ROUGE-1 Recall | ROUGE-L Recall |
|--------------|--------|----------------|----------------|
| **LogGen**   | 0.6464 | 0.9116         | 0.8556         |
| **FastLog**  | 0.3343 | 0.5991         | 0.5953         |

---

## 💡 Future Work

- **Enhanced Context Parsing**: Integration with Abstract Syntax Trees (ASTs) and Code2Vec for better structural understanding.
- **Multi-Language Support**: Extend LogGen to support C, C++, and other programming languages.
- **Prompt Optimization**: Implement advanced prompt engineering techniques like Chain of Thought (CoT) for better results.

---

## 🧑‍💻 Contributors

- **Yogesh Hasabe** - [Yogesh31Hasabe](https://github.com/Yogesh31Hasabe)
- **Jay Patel** - [Jay Patel]()
- **Ashaka Mehta** - [ashaka11](https://github.com/ashaka11)
- **Sagar Dama** - [sagar110599](https://github.com/sagar110599)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
