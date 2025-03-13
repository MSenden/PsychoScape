import os
import time
import openai
import tomllib
import pandas as pd

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from src.utils.parsing import parse_directories
from src.utils.semantic import load_configurations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

BASEPATH = os.environ['BASEPATH']


# Define utility functions
def process_dictionary(dictionary, required_keys):
    """
    Process the dictionary extracted from the LLM output.

    Parameters:
    - dictionary: dict, the LLM output dictionary.
    - required_keys: list, the required keys.
    - allowed_values: list, the allowed values for the required keys.

    Returns:
    - new_dict: dict, the extracted dictionary.
    """

    allowed_values = ['yes', 'no']
    conversion = {"yes": True, "no": False}

    try:
        # Verify that the required fields are present
        if not all(key in dictionary for key in required_keys):
            raise ValueError("Missing required fields in the LLM output")

        new_dict = {
            key: conversion[dictionary[key]]
            for key in required_keys if dictionary[key] in allowed_values
        }
        if len(new_dict) != len(required_keys):
            raise ValueError("Invalid values in the LLM output")

        return new_dict

    except Exception:
        return None


def safe_dictionary_extraction(dimension, categories, dimensions_text, chain,
                               retries, delay):
    """
    Safely extract the dictionary from the LLM output.

    Parameters:
    - dimension: str, the dimension to extract.
    - categories: list, the categories.
    - dimensions_text: str, the dimensions text.
    - chain: LLMChain, the LLM chain to invoke.
    - retries: int, the number of retries allowed.
    - delay: int, the delay between retries.

    Returns:
    - dictionary: dict, the extracted dictionary.
    """
    categories_text = "\n".join(categories)
    chain_input = {
        "dimension": dimension,
        "categories": categories_text,
        "analysis": dimensions_text
    }
    attempt = 0
    while attempt < retries:
        dictionary = chain.invoke(chain_input)
        cluster_dimensions = process_dictionary(dictionary,
                                                required_keys=categories)
        time.sleep(delay)
        if cluster_dimensions:
            return cluster_dimensions
        else:
            attempt += 1

    return None


def get_dimension_df(csv_path, dim_id, dimension_df, dimension, categories,
                     chain, retries, delay):
    filename = f"dimension_{dim_id}_{dimension.lower().replace(' ','_')}.csv"

    if os.path.exists(os.path.join(csv_path, filename)):
        print('Loading from CSV')
        return pd.read_csv(os.path.join(csv_path, filename))

    print('Processing')
    dimension_dict = []
    for cluster in tqdm(dimension_df.index):
        analysis = dimension_df.loc[cluster, "Dimensions"]
        cluster_dict = safe_dictionary_extraction(dimension, categories,
                                                  analysis, chain, retries,
                                                  delay)
        cluster_dict['Cluster ID'] = cluster
        dimension_dict.append(cluster_dict)

    dimension_df = pd.DataFrame(dimension_dict)
    dimension_df.to_csv(os.path.join(csv_path, filename), index=False)
    return pd.DataFrame(dimension_dict)


DIMENSIONS_PROMPT = PromptTemplate(
    input_variables=["dimension", "categories", "analysis"],
    template="""
            You are an expert in psychology. 
            You are provided with an analysis of research within a psychological cluster along the following dimensions:

            1. **Appliedness**: The extent to which the research is fundamental (theoretical, conceptual, or basic cognitive/behavioral processes, phenomena, or effects) or applied (directly aimed to inform clinical, organizational, educational, forensic, technological applications, or policy-relevant).
            2. **Psychological Domain**: The primary domain of psychology covered in the research (e.g., cognitive psychology, social psychology, developmental psychology, clinical psychology, personality psychology, industrial-organizational psychology, educational psychology, neuropsychology, forensic psychology).
            3. **Cognitive vs. Affective Focus**: The relative emphasis on cognitive processes (e.g., memory, decision-making, problem-solving) versus affective processes (e.g., emotions, motivation, mood disorders).
            4. **Individual vs. Social Focus**: The extent to which the research focuses on individual-level psychological processes (e.g., perception, attention, executive function) versus social and interpersonal phenomena (e.g., group dynamics, persuasion, prejudice, social norms).
            5. **Theory Engagement**: The extent to which the research is theory-driven (hypothesis testing) versus data-driven (exploratory, descriptive).
            6. **Theory Scope**: The scope of the theory under investigation.  
               - **Broad Framework**: A theory that attempts to provide a high-level, integrative explanation of cognition, behavior, or psychological phenomena (e.g., Dual-Process Theory, Cognitive Dissonance, Social Identity Theory, Self-Determination Theory).  
               - **Domain-Specific Theory**: A well-established framework within a psychological subfield (e.g., Attachment Theory, Theory of Mind, Prospect Theory, Working Memory Model, Feature Integration Theory, Attentional Control Theory).  
               - **Micro-Theory**: A narrowly scoped theory or model that explains a specific psychological process (e.g., a mechanism specific to a given phenomenon).  
            7. **Methodological Approach**: The methodological approach used in the research.  
               - **Experimental**: Controlled studies with independent and dependent variables to establish causal relationships.  
               - **Observational (Correlational / Descriptive)**: Studies that measure variables in naturally occurring settings without active manipulation.  
               - **Survey-Based / Psychometric**: Studies using structured surveys, self-report instruments, or psychometric scales.  
               - **Qualitative Research**: Studies using interviews, thematic analysis, discourse analysis, or ethnographic methods.  
               - **Computational / Modeling**: Research using computational models, simulations, or AI-based approaches.  
               - **Meta-Analytic / Systematic Review**: Research synthesizing existing studies through quantitative or systematic review methods.  
            8. **Qualitative vs. Quantitative**: The degree to which the research employs **qualitative** (e.g., interviews, thematic coding) versus **quantitative** (e.g., statistical modeling, psychometric analysis) methodologies.
            9. **Interdisciplinarity**: The extent to which the research integrates concepts and methods from other fields (e.g., neuroscience, sociology, economics, linguistics, computer science, philosophy, artificial intelligence).  
               - **Low**: Confined to a single discipline.  
               - **Medium**: Uses some concepts from another field but is primarily psychological.  
               - **High**: Actively integrates multiple disciplines.  
               - **Very High (Transdisciplinary)**: Merges disciplines or includes non-academic stakeholders (e.g., policy-makers, industry collaborations).  

            Your task is to focus solely on the dimension of {dimension} and provide a binary indication ("yes" or "no") of whether the research within the cluster falls within the specified categories.
            Here are the categories for this dimension:
            {categories}

            **Output Format:**

            Please present your findings in **JSON format** with the following structure:
            {{
                "Category 1": "yes" / "no",
                "Category 2": "yes" / "no",
                "Category 3": "yes" / "no",
                ...
                Category n: "yes" / "no"
            }}

            ```

            **Instructions:**
            - **Accuracy is crucial**: Ensure all information is directly supported by the provided analysis. Do not include information not present in the analysis or make external assumptions.
            - **Consistency**: Ensure that the evaluation of the dimension does not contradict the provided analysis (including other dimensions).
            - **Focus and Precision**: Only evaluate the dimension of {dimension} and provide a binary response for each category. Do not include any additional information or explanations.
            - **Proper Category Naming**: Ensure that the categories are named correctly and accurately reflect the content of the analysis. ONLY use the provided categories and replace Category 1, Category 2, etc. with the actual category names.
            - **Binary Response**: Ensure that the response for each category is binary (yes or no) and does not include any other text or explanations. Categories are not mutually exclusive, and multiple categories can be marked as "yes" if they apply to the analysis.

            **Here is the analysis of the research within the cluster along the dimension of {dimension}:**
            {analysis}""",
)

if __name__ == '__main__':
    configurations = load_configurations()
    directories = parse_directories()

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    dimensions_chain = DIMENSIONS_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV file
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'], 'Psychology')
    checkpoint_path = os.path.join(BASEPATH,
                                   directories['internal']['checkpoints'])
    cluster_csv_file = 'clusters_defined_distinguished_trends_assessed.csv'
    cluster_df = pd.read_csv(os.path.join(csv_directory, cluster_csv_file))
    narrative_dimension_df = cluster_df[["Cluster ID", "Dimensions"]]

    # Define the output path
    dimension_csv_file = 'dimensions.csv'
    dimension_categories = {
        'Appliedness': [
            'Fundamental', 'Translational', 'Clinical', 'Organizational',
            'Educational', 'Forensic', 'Legal', 'Technological Exploitation',
            'Poolicy-Relevant'
        ],
        'Psychological Domain': [
            'Cognitive Psychology',
            'Social Psychology',
            'Developmental Psychology',
            'Clinical Psychology',
            'Personality Psychology',
            'Industrial-Organizational Psychology',
            'Educational Psychology',
            'Neuropsychology',
            'Forensic Psychology',
            'Legal Psychology',
        ],
        'Cognitive vs. Affective Focus':
        ['Cognitive Processes', 'Affective Processes'],
        'Individual vs. Social Focus': ['Individual-Level', 'Social-Level'],
        'Theory Engagement': ['Data-Driven', 'Hypothesis-Driven'],
        'Theory Scope':
        ['Broad Framework', 'Domain-Specific Theory', 'Micro-Theory'],
        'Methodological Approach': [
            'Experimental', 'Observational', 'Survey-Based', 'Psychometric',
            'Qualitative', 'Computational', 'Meta-Analytic',
            'Systematic Review'
        ],
        'Qualitative vs. Quantitative':
        ['Qualitative', 'Quantitative', 'Mixed Methods'],
        'Interdisciplinarity': ['Low', 'Medium', 'High', 'Very High']
    }

    for dimension_id, (dimension,
                       categories) in enumerate(dimension_categories.items(),
                                                start=1):

        dimension_df = get_dimension_df(checkpoint_path, dimension_id,
                                        narrative_dimension_df, dimension,
                                        categories, dimensions_chain,
                                        configurations['llm']['retries'],
                                        configurations['llm']['delay'])

        data = {
            f"{dimension}_{category}": dimension_df[category].values
            for category in categories
        }

        if dimension_id == 1:
            dim_cat_df = pd.DataFrame(data)
        else:
            dim_cat_df = pd.concat([dim_cat_df, pd.DataFrame(data)], axis=1)

    # Add Cluster ID to the DataFrame
    dim_cat_df['Cluster ID'] = narrative_dimension_df['Cluster ID']

    # Save the cluster definitions to a CSV file
    dim_cat_df.to_csv(os.path.join(csv_directory, dimension_csv_file),
                      index=False)
