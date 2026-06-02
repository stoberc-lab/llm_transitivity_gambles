# Transitivity of Preference in LLMs 
Code for evaluating transitivity adherence in large language models using binary gamble choices. Models are presented with pairs of gambles, their first-token choice probabilities are exracted and the resulting choice probabilities are tested against four probabilistic models of transitivity: Weak Stochastic Transitivity (WST), Mixture Model of Transitive Preferences (MMTP), Medium-Certain Transitivity (MCT), and High-Certain Transitivity (HCT). 

Three conditions are manipulated: question format, generation temperature, and contextual memory. 

## Repository Structure 

### Directories

#### analysis

R scripts for processing formatted analysis data into results using the transitivity checks for each transitive model.

#### inputs

TEst input data is laoded from this directory by default if no path is specified. These files and sub-directories provide the prompt templates and values for the data to be passed to the LLMs in the test runs.

#### test_configs

Configuration files are loaded from here by default if no path is specified. These files can be used to set up the parameters of a test such as memory, seeds, temperatures, etc.

### main.py

CEntral file to begin a test. Takes parameters from the command line to determine what to process, and then begins the test.

### benchmark_logger.py

Handles console and filesystem logging of the processing occurring during each test run.

### benchmark_runner.py

Called from main.py to load data and execute the test process.

### model_handler.py

Handles loading and interfacing with the LLMs using Transformers, pytorch, and custom functions to extract needed responses.

### prompt.py

Object class for storing data and responses about a specific item from the prompt template and variable combinations.

### generate_gambles_memoryfull_permutation_sets.py

Simple script to generate randomized orderings of prompts for memory trial testing.

### memory_handler.py

Handles data for contextual memory condition.

### test_handler.py

Loads test and prompt data.

### output_handler.py

Outputs data from LLM prompt responses to the configured output directory.

### process_*.py

Converts .out files to formatted csvs for analysis.

## Environment 

The Meta Llama 2 and 3 require a Hugging Face access token. To run the 70B parameter sized models all experiment was run on HPC hardware provided by the Research Support Solutions and in part by the National Science Foundation under grant number CNS-1429294 at the University of Missouri, Columbia MO. DOI: https://doi.org/10.32469/10355/69802.

## Pipeline 

The experiment runs in three stages: generate choice, aggregate to choice probabilities, then test transitivity. 

### 1. Generate Choices 

main.py runs one model for one benchmark and config, writing for each prompt .out file under results/<benchmark>/<config_name>/. 

Model names and reivision IDs for all 20 models are listed in Appendix of the paper. 

For the memory condition, first generate the ordered presentation sets, then run main.py with benchmark gambles_memoryfull and a memoryfull config. 

### 2. Aggregate to choice probabilities 

The process_results_probs.py reads the .out files and writes one csv per question format, with one column per gamble pair (A_B, B_A, A_C, ...). Each cell is normalized probability preferring the first listed gamble. The output is written to <benchmakr>/<config_name>/<format>.csv. 

### 3. Test Transitivity 
The scripts in analysis/ apply the four transitivity models. Each file defines a function (test_wst, test_mct, ...) that takes a data frame of choice probabilities and returns it with per-triad adherence result and a total adherence score. 

Before running the analysis, the gamble pairs are aggregated to control for positional bias: aggregated probability for a pair is teh mean of the first-positioned probability cell and one minus its reverse (P(A_B) + (1 - P(B_A))/2). This single value is what the function evaluates. 

A score of 1 means that from all 10 triads in a certain gamble set met the condition; 0 means at least 1 failed. 
