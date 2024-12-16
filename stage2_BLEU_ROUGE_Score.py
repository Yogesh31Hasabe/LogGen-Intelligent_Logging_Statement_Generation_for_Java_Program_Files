import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

TRAIN_4_WITH_LOGS='''
package com.example.logging;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ComplexProcessor {
    private static final Logger LOGGER = Logger.getLogger(ComplexProcessor.class.getName());

    public static void main(String[] args) {
        LOGGER.info("ComplexProcessor application started.");

        List<String> data = fetchDataFromSource();
        if (data.isEmpty()) {
            LOGGER.warning("No data fetched from source.");
        } else {
            LOGGER.info("Data fetched successfully.");
        }

        try {
            processData(data);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error processing data: ", e);
        }

        LOGGER.info("ComplexProcessor application ended.");
    }

    private static List<String> fetchDataFromSource() {
        LOGGER.fine("Entering fetchDataFromSource method.");
        List<String> data = new ArrayList<>();
        // Simulate data fetching
        try {
            // Simulate potential delay or issue
            Thread.sleep(1000);
            data.add("Sample Data 1");
            data.add("Sample Data 2");
            LOGGER.fine("Data fetched: " + data);
        } catch (InterruptedException e) {
            LOGGER.log(Level.WARNING, "Data fetching interrupted: ", e);
        }
        LOGGER.fine("Exiting fetchDataFromSource method.");
        return data;
    }

    private static void processData(List<String> data) throws Exception {
        LOGGER.fine("Entering processData method with data: " + data);
        if (data == null || data.isEmpty()) {
            LOGGER.warning("No data to process.");
            throw new Exception("Data is null or empty.");
        }
        for (String item : data) {
            LOGGER.fine("Processing item: " + item);
            // Simulate processing
            if ("Sample Data 2".equals(item)) {
                LOGGER.warning("Encountered known issue with item: " + item);
            }
        }
        LOGGER.fine("Exiting processData method.");
    }
}
'''

TRAIN_4_WITHOUT_LOGS='''
package com.example.logging;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ComplexProcessor {
    private static final Logger LOGGER = Logger.getLogger(ComplexProcessor.class.getName());

    public static void main(String[] args) {
        <FILL_ME>

        List<String> data = fetchDataFromSource();
        if (data.isEmpty()) {
            <FILL_ME>
        } else {
            <FILL_ME>
        }

        try {
            processData(data);
        } catch (Exception e) {
            <FILL_ME>
        }

        <FILL_ME>
    }

    private static List<String> fetchDataFromSource() {
        <FILL_ME>
        List<String> data = new ArrayList<>();
        // Simulate data fetching
        try {
            // Simulate potential delay or issue
            Thread.sleep(1000);
            data.add("Sample Data 1");
            data.add("Sample Data 2");
            <FILL_ME>
        } catch (InterruptedException e) {
            <FILL_ME>
        }
        <FILL_ME>
        return data;
    }

    private static void processData(List<String> data) throws Exception {
        <FILL_ME>
        if (data == null || data.isEmpty()) {
            <FILL_ME>
            throw new Exception("Data is null or empty.");
        }
        for (String item : data) {
            <FILL_ME>
            // Simulate processing
            if ("Sample Data 2".equals(item)) {
                <FILL_ME>
            }
        }
        <FILL_ME>
    }
}
'''

TRAIN_3_WITH_LOGS='''
package com.example.logging;

import java.util.logging.Level;
import java.util.logging.Logger;

public class SampleClass {
    private static final Logger LOGGER = Logger.getLogger(SampleClass.class.getName());

    public static void main(String[] args) {
        LOGGER.info("Application started.");

        try {
            int result = divide(10, 0);
            LOGGER.info("Division result: " + result);
        } catch (ArithmeticException e) {
            LOGGER.log(Level.SEVERE, "Exception occurred: ", e);
        }

        LOGGER.info("Application ended.");
    }

    private static int divide(int a, int b) {
        LOGGER.fine("Entering divide method with parameters: " + a + ", " + b);
        int result = a / b;
        LOGGER.fine("Exiting divide method with result: " + result);
        return result;
}
}
'''

TRAIN_3_WITHOUT_LOGS='''
package com.example.logging;

import java.util.logging.Level;
import java.util.logging.Logger;

public class SampleClass {
    private static final Logger LOGGER = Logger.getLogger(SampleClass.class.getName());

    public static void main(String[] args) {
        <FILL_ME>

        try {
            int result = divide(10, 0);
            <FILL_ME>
        } catch (ArithmeticException e) {
            <FILL_ME>
        }

        <FILL_ME>
    }

    private static int divide(int a, int b) {
        <FILL_ME>
        int result = a / b;
        <FILL_ME>
        return result;
}
}
'''

# Directories for input and output
input_dir = './test_files/input'
output_dir = './test_files/generated'
comparison_dir = './test_files/output'

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

# Initialize tokenizer and model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct').to(device)

# Set up text generation pipeline
text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Function to generate logs for placeholders in Java code
def generate_logs(java_code):
    messages = [
        {
            "role": "system", 
            "content": 
            (
                "You are a Java log generation expert. In the generated output just give the plain Java output without any other formating"
                "Strictly return just the genereated Java code with accurate logs only and nothing else"
                "Replace every `<FILL_ME>` placeholder in the provided Java code with accurate and context-specific logging statements. Follow these guidelines:"
                "1. Analyze Context:"
                "   - Understand the operation or event taking place near each `<FILL_ME>` placeholder."
                "   - Ensure the logging statement matches the context of the operation."
                "2. Logging Levels:"
                "   - Use INFO for successful or expected operations (e.g., adding items, updating quantities, displaying inventory)."
                "   - Use WARNING for potential issues (e.g., when updating an already existing item)."
                "   - Use SEVERE for critical errors (e.g., item not found or insufficient quantity)."
                "3. Log Content:"
                "   - Ensure each log statement includes descriptive text explaining the event and relevant variables such as `name` or `quantity` for clarity."
                "4. Preserve Code Integrity:"
                "   - Do not modify or delete any part of the code other than replacing `<FILL_ME>` placeholders."
                "   - Keep the formatting and structure of the code identical to the original."
                "5. Output:"
                "   - Return the complete code with all `<FILL_ME>` placeholders replaced by appropriate logging statements."
                "   - Ensure that the rest of the code remains completely unchanged."
                "Return the updated code with all placeholders replaced and no other changes."
            )
        },

        # Example 1
        {"role": "user", "content": """
    Just give me the logging statement for the <FILL_ME> placeholder in the below code. Give me a complete code with logging statements included Do not change the original code and give no explanation.

    Input Code:
    public int add(int a, int b) {
        <FILL_ME>
        int result = a + b;
        <FILL_ME>
        return result;
    }
    """},

        {"role": "system", "content": """
    public int add(int a, int b) {
        System.out.println("Entering method: add with parameters a=" + a + ", b=" + b);
        int result = a + b;
        System.out.println("Result of addition: " + result);
        return result;
    }
    """},

        # Example 2
        {"role": "user", "content": """
    Just give me the logging statement for the <FILL_ME> placeholder in the below code. Give me a complete code with logging statements included Do not change the original code and give no explanation.

    Input Code:
    public double divide(double a, double b) {
        <FILL_ME>
        if (b == 0) {
            <FILL_ME>
            throw new IllegalArgumentException("Divider cannot be zero");
        }
        double result = a / b;
        <FILL_ME>
        return result;
    }
    """},

        {"role": "system", "content": """
    public double divide(double a, double b) {
        System.out.println("Entering method: divide with parameters a=" + a + ", b=" + b);
        if (b == 0) {
            System.err.println("IllegalArgumentException: Divider cannot be zero");
            throw new IllegalArgumentException("Divider cannot be zero");
        }
        double result = a / b;
        System.out.println("Result of division: " + result);
        return result;
    }
    """},
    {"role": "user", "content": """
    Just give me the logging statement for the <FILL_ME> placeholder in the below code. Give me a complete code with logging statements included Do not change the original code and give no explanation.

    Input Code:

    """+TRAIN_4_WITHOUT_LOGS},

        {"role": "system", "content": """
    HERE is the output code with <FILL_ME> replaced with high quality logs

    """+TRAIN_4_WITH_LOGS},
    {"role": "user", "content": """
    Just give me the logging statement for the <FILL_ME> placeholder in the below code. Give me a complete code with logging statements included. Please Please Do not change the original code and give no explanation.

    Input Code:

    """+TRAIN_3_WITHOUT_LOGS},

        {"role": "system", "content": """
    HERE is the output code with <FILL_ME> replaced with high quality logs

    """+TRAIN_3_WITH_LOGS},
    {"role": "user", "content": """
    Learn from the above examples and Just give me the logging statement for the <FILL_ME> placeholder in the below code. Give me the java code with <FILL_ME> replaced by logging statement. Please give me a high quality log which encompasses a large context. Do not give me anything else except the code. Do not change the original code and give no explanation.

    Input Code:

    """+java_code},
    ]

    outputs = text_generation_pipeline(
        messages,
        max_new_tokens=2000,
        temperature=0.1,
    )
    
    
    # Extract the 'content' field from the assistant's message
    # Navigate the structure to extract the Java code
    if isinstance(outputs, list):
        for entry in outputs:
            if isinstance(entry, dict) and 'generated_text' in entry:
                generated_text = entry['generated_text']
                if isinstance(generated_text, list):
                    for sub_entry in generated_text:
                        if isinstance(sub_entry, dict) and sub_entry.get('role') == 'assistant':
                            content = sub_entry.get('content', '')
                            # Extract the code block from the content
                            start_index = content.find("```java")
                            end_index = content.rfind("```")
                            if start_index != -1 and end_index != -1:
                                return content[start_index + len("```java"):end_index].strip()

    # If no valid code is found, raise an error
    raise ValueError(f"Unable to extract Java code from output: {outputs}")

# Function to calculate BLEU and ROUGE scores
def calculate_scores(generated_code, reference_code):
    # Tokenize for BLEU and ROUGE
    generated_tokens = generated_code.split()
    reference_tokens = reference_code.split()
    
    # BLEU score
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    
    # ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_code, generated_code)
    
    return bleu_score, rouge_scores

# Process Java files
for file_name in os.listdir(input_dir):
    if file_name.endswith('.java'):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        reference_path = os.path.join(comparison_dir, file_name)
        # print(f"Reference Path: {reference_path}")

        # Check if the generated file already exists
        if os.path.exists(output_path):
            print(f"Generated file already exists for {file_name}, using it for comparison.")
            # Read the already generated code
            with open(output_path, 'r') as file:
                generated_code = file.read()
        else:
            # Read the input Java file
            with open(input_path, 'r') as file:
                input_code = file.read()

            # Generate logs
            generated_code = generate_logs(input_code)
            print(f"\n Generated Code: {generated_code} \n\n")

            # Save the generated code to the output directory
            if not isinstance(generated_code, str):
                raise ValueError(f"Generated code is not a string: {generated_code}")

            with open(output_path, 'w') as file:
                file.write(generated_code)

        # If reference file exists, calculate scores
        if os.path.exists(reference_path):
            with open(reference_path, 'r') as file:
                reference_code = file.read()
            
            # Calculate BLEU and ROUGE scores
            bleu, rouge = calculate_scores(generated_code, reference_code)
            
            # Print scores for this file
            print(f"Scores for {file_name}:")
            print(f"  BLEU: {bleu}")
            print(f"  ROUGE-1: {rouge['rouge1']}")
            print(f"  ROUGE-L: {rouge['rougeL']}")
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print("Processing complete. Generated Java files with logs are saved to:", output_dir)