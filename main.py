import stage1, stage2
import os
import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
from scipy.spatial.distance import cdist
import numpy as np

java_file_path = "./test.java"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Test java code for which logs needs to be generated which is given as the last user prompt in the below list of messages 
java_code = """
import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class MultiThreadedFileProcessor {

    private static final Logger logger = Logger.getLogger(MultiThreadedFileProcessor.class.getName());

    public static void main(String[] args) {
        <FILL_ME>;

        List<String> filePaths = Arrays.asList("file1.txt", "file2.txt", "file3.txt"); // Sample file paths
        int threadCount = 4;

        <FILL_ME>;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);

        for (String filePath : filePaths) {
            executor.submit(() -> processFile(filePath));
        }

        executor.shutdown();
        try {
            if (executor.awaitTermination(1, TimeUnit.MINUTES)) {
                <FILL_ME>;
            } else {
                <FILL_ME>;
            }
        } catch (InterruptedException e) {
            <FILL_ME>;
        }

        <FILL_ME>;
    }

    private static void processFile(String filePath) {
        <FILL_ME>;

        File file = new File(filePath);
        if (!file.exists()) {
            <FILL_ME>;
            return;
        }

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            int lineCount = 0;
            while ((line = reader.readLine()) != null) {
                processLine(line);
                lineCount++;
            }
            <FILL_ME>;
        } catch (IOException e) {
            <FILL_ME>;
        }
    }

    private static void processLine(String line) {
        <FILL_ME>;

        try {
            // Simulate processing
            Thread.sleep(100);
            if (line.contains("error")) {
                <FILL_ME>;
            }
        } catch (InterruptedException e) {
            <FILL_ME>;
            Thread.currentThread().interrupt(); // Restore interrupt status
        }
    }
}
"""

stage1_model, stage1_tokenizer= stage1.setup()
stage2_pipeline= stage2.setup()

with open(java_file_path, 'r') as file:
    # Read the entire contents of the file into a string
    java_file_content = file.read()
clean_pattern = r"(//.*?$)|(/\*.*?\*/)|(/\*\*.*?\*/)|(^\s*import\s.*?;$)|(^\s*$)"
masked_lines=[]
result=re.findall(clean_pattern, java_file_content, flags=re.DOTALL | re.MULTILINE)
for res in result:
  masked_lines.append("".join(res))
partially_masked_content = re.sub(clean_pattern, '<MASKED>', java_file_content, flags=re.DOTALL | re.MULTILINE)
original_content_labels=[]
split_pattern = r'(?<=[;{}])\s*(?=\S)|(?<=<MASKED>)\s*(?=\n|$)'
mask_count=0
for line in re.split(split_pattern, partially_masked_content):
  record={}
  if '<MASKED>' in line:
    record['is_code']=False
    record['content']=masked_lines[mask_count]
    mask_count=mask_count+1
  else:
    record['is_code']=True
    record['content']=line.strip() # this removes extra space. Formatting is f
  original_content_labels.append(record)
code_content = [d['content'] for d in original_content_labels if d['is_code'] ]
code= "\n".join(code_content)
log_positions=stage1.predict_log_positions(code, stage1_model, stage1_tokenizer, device)

for code_index in log_positions:
        # Locate the corresponding entry in original_content_labels
        for j, entry in enumerate(original_content_labels):
            if entry['is_code'] and entry['content'] == code_content[code_index]:
                # Insert <FILLME> after the current entry
                original_content_labels.insert(j + 1, {'is_code': True, 'content': '<FILL_ME>'})
                break

code_with_placeholder= "\n".join(d['content'] for d in original_content_labels)
code_with_logs=stage2.generate_logs(stage2_pipeline,code_with_placeholder)

with open("./test-logs.java", 'a') as file:
    file.write(code_with_logs)