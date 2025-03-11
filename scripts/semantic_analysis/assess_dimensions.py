import os
import time
import openai
import tomllib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from src.utils.parsing import parse_directories
from src.utils.hypersphere import get_centroids
from src.utils.semantic import load_configurations, safe_dictionary_extraction, get_abstract_strings
from src.utils.load_and_save import load_embedding_shards, align_to_df

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

BASEPATH = os.environ['BASEPATH']

# Deine prompt temlates
DIMENSIONS_PROMPT = PromptTemplate(
    input_variables=["title", "abstracts"],
    template="""
            You are an expert in psychology and scientific text analysis. 
            You are provided with a list of psychological research abstracts that belong to the cluster '{title}'.
            
            Your task is to identify **key psychological dimensions** that best describe the cluster, guided by the following dimensions:

            1. **Appliedness**: The extent to which the research is fundamental (theoretical, conceptual, or basic cognitive/behavioral processes) or applied (clinical, organizational, educational, forensic, technological, or policy-relevant).
            2. **Psychological Domain**: The primary domain of psychology covered in the research (e.g., cognitive psychology, social psychology, developmental psychology, clinical psychology, personality psychology, industrial-organizational psychology, educational psychology, neuropsychology, forensic psychology).
            3. **Cognitive vs. Affective Focus**: The relative emphasis on cognitive processes (e.g., memory, decision-making, problem-solving) versus affective processes (e.g., emotions, motivation, mood disorders).
            4. **Individual vs. Social Focus**: The extent to which the research focuses on individual-level psychological processes (e.g., perception, attention, executive function) versus social and interpersonal phenomena (e.g., group dynamics, persuasion, prejudice, social norms).
            5. **Theory Engagement**: The extent to which the research is theory-driven (hypothesis testing) versus data-driven (exploratory, descriptive). Identify specific psychological theories if applicable.
            6. **Theory Scope**: The scope of the theory under investigation, ranging from specific mechanisms to broad overarching psychological frameworks. Identify any major psychological theories, such as **dual-process theory, cognitive dissonance, reinforcement learning, social identity theory, attachment theory, self-determination theory, embodied cognition, etc.** if applicable (else, no general theory). There might be more than one framework.
            7. **Methodological Approach**: The methodological approach used in the research (e.g., experimental, observational, survey-based, longitudinal, qualitative, meta-analysis, computational modeling). Identify specific methods if applicable (e.g., fMRI, reaction time measures, eye-tracking, structured interviews, behavioral coding, implicit association tests).
            8. **Qualitative vs. Quantitative**: The degree to which the research employs **qualitative** (e.g., interviews, discourse analysis, thematic coding) versus **quantitative** (e.g., statistical modeling, psychometrics, inferential testing) methodologies.
            9. **Interdisciplinarity**: The extent to which the research integrates concepts and methods from other fields (e.g., neuroscience, sociology, economics, linguistics, computer science, philosophy, artificial intelligence).

            Examples are not exhaustive and abstracts may contain multiple dimensions.

            **Output Format:**

            Please present your findings in **JSON format** with the following structure:
            {{
                "Dimension 1 - Appliedness": "Brief assessment of the research's appliedness.",
                "Dimension 2 - Psychological Domain": "Brief classification of the domain of psychology.",
                "Dimension 3 - Cognitive vs. Affective Focus": "Brief description of the relative emphasis on cognitive vs. affective processes.",
                "Dimension 4 - Individual vs. Social Focus": "Brief description of whether the research focuses on individual psychological mechanisms or social/group-level processes.",
                "Dimension 5 - Theory Engagement": "Brief assessment of the research's theory engagement.",
                "Dimension 6 - Theory Scope": "Brief assessment of the research's theory scope. Identify specific psychological frameworks if applicable.",
                "Dimension 7 - Methodological Approach": "Brief overview of the methodological approach used in the research. Identify specific methods if applicable.",
                "Dimension 8 - Qualitative vs. Quantitative": "Brief classification of the study as qualitative, quantitative, or mixed methods.",
                "Dimension 9 - Interdisciplinarity": "Brief assessment of the research's interdisciplinarity."
            }}

            ```

            **Instructions:**
            - **Accuracy is crucial**: Ensure all information is directly supported by the provided abstracts. Do not include information not present in the abstracts or make external assumptions.

            - **Clarity and Precision**: Assessments and descriptions should be clear and accurately reflect the content of the abstracts.

            - **Conciseness**: Do not include any additional text or explanations beyond the specified JSON output. Do not generate more output than necessary.

            **Here are the abstracts:**
            {abstracts}""",
)

if __name__ == '__main__':
    configurations = load_configurations()
    required_fields = configurations['dimensions']['required_fields']
    directories = parse_directories()

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    dimensions_chain = DIMENSIONS_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV file
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'], 'Psychology')
    article_csv_file = 'articles_merged_cleaned_filtered_clustered.csv'
    article_df = pd.read_csv(os.path.join(csv_directory, article_csv_file))
    article_df = article_df[article_df['Type'] == 'Research']

    # Define the output path
    cluster_csv_file = 'clusters_defined_distinguished_trends.csv'
    cluster_df = pd.read_csv(os.path.join(csv_directory, cluster_csv_file))
    output_file = os.path.join(
        csv_directory, cluster_csv_file.replace('.csv', '_assessed.csv'))
    checkpoints_file = os.path.join(BASEPATH,
                                    directories['internal']['checkpoints'],
                                    'assessment_checkpoint.csv')

    if os.path.exists(checkpoints_file):
        cluster_dimensions_df = pd.read_csv(checkpoints_file)
        cluster_dimensions = cluster_dimensions_df.to_dict('records')
        processed_clusters = set(
            cluster_dimensions_df['Cluster ID'].values.tolist())

        print(f'{len(processed_clusters)} clusters have been processed.')
    else:
        cluster_dimensions = []
        processed_clusters = set()

    # Load the embeddings and PMIDs
    shard_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['psycho'])
    files = glob(os.path.join(shard_directory, '*.h5'))
    embeddings, pmids = load_embedding_shards(files)

    # Align the embeddings and PMIDs to the DataFrame
    aligned_embeddings, aligned_pmids = align_to_df(embeddings, pmids,
                                                    article_df)

    # Free memory
    del embeddings
    del pmids

    # Get the centroids of the clusters
    centroids = get_centroids(aligned_embeddings,
                              article_df['Cluster ID'].values)
    centroid_similarities = centroids.dot(centroids.T) - np.eye(
        centroids.shape[0])

    # Get cluster labels
    labels = article_df['Cluster ID'].values

    # Get the abstracts
    abstracts = article_df['Abstract'].values

    for label, centroid in tqdm(enumerate(centroids)):

        if label in processed_clusters:
            continue

        cluster_embeddings = aligned_embeddings[labels == label]
        cluster_abstracts = abstracts[labels == label]

        number_of_abstracts = min(
            configurations['dimensions']['max_abstract_number'],
            len(cluster_abstracts))

        similarity = centroid.dot(cluster_embeddings.T)
        top_indices = np.argsort(similarity)[::-1][:number_of_abstracts]

        cluster_abstracts = '\n\n'.join(cluster_abstracts[top_indices])

        cluster_title = cluster_df.loc[label, 'Title']

        chain_input = {"title": cluster_title, "abstracts": cluster_abstracts}

        extracted_dict = safe_dictionary_extraction(
            required_fields, chain_input, dimensions_chain,
            configurations['llm']['retries'], configurations['llm']['delay'])

        # Join the keys as a single string with each key as a title and a line break between them
        cluster_dimensions_dict = {
            'Dimensions':
            '\n'.join(
                [f'{key}: {value}' for key, value in extracted_dict.items()])
        }

        cluster_dimensions_dict['Cluster ID'] = label
        cluster_dimensions.append(cluster_dimensions_dict)
        processed_clusters.add(label)

        cluster_dimensions_df = pd.DataFrame(cluster_dimensions)
        cluster_dimensions_df.to_csv(checkpoints_file, index=False)

        print(cluster_dimensions)
        break

    # Convert the cluster definitions to a DataFrame
    cluster_dimensions_df = pd.DataFrame(cluster_dimensions)

    # Merge the cluster definitions with the existing cluster definitions
    cluster_dimensions_df = cluster_df.merge(cluster_dimensions_df,
                                             on='Cluster ID')

    # Save the cluster definitions to a CSV file
    cluster_dimensions_df.to_csv(output_file, index=False)
