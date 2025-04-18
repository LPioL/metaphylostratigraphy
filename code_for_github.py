#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:43:25 2025

@author: llopez06
"""



import numpy as np
import pandas as pd
import math
from collections import Counter
from scipy.spatial import distance

import networkx as nx
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import stringdb

from collections import defaultdict
import csv
from gprofiler import GProfiler
import dataframe_image as dfi
import re
import os, datetime 
import h5py
#import pyhgnc

import seaborn as sns
from scipy.stats import ttest_ind, entropy, ks_2samp, chisquare

from scipy.stats import hypergeom


# Define the path for the results directory
results_dir = 'Results_age_age_gladyshev' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

# Ensure the directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)





##############################################
# Functions
##############################################

def plot_normalized_distributions_with_stats(test_data, age_human, output_path, name, color, lower_offset=-0.02):
    """
    Perform an over-representation test for all evolutionary categories
    and plot the comparison of distributions with stars for significant categories.
    Enhanced visualization with better fonts and styling.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import hypergeom
    
    plt.style.use('seaborn')
    
    evolutionary_order = [
        "All living organisms", "Eukaryota", "Opisthokonta", "Holozoa", "Metazoa",
        "Eumetazoa", "Bilateria", "Deuterostomia", "Chordata", "Olfactores",
        "Craniata", "Euteleostomi", "Tetrapoda", "Amniota", "Mammalia",
        "Eutheria", "Boreoeutheria", "Euarchontoglires", "Primates"
    ]

    # Align the data
    test_data_aligned = test_data.set_index(1).reindex(evolutionary_order, fill_value=0).reset_index()
    age_human_aligned = age_human.set_index(1).reindex(evolutionary_order, fill_value=0).reset_index()

    # Calculate statistics
    total_test = test_data_aligned['percentage'].sum()
    total_age_human = age_human_aligned['percentage'].sum()

    # Perform statistical tests
    results = []
    for idx, row in test_data_aligned.iterrows():
        category = row['condition']
        observed_count = row['percentage']
        expected_count = age_human_aligned.loc[idx, 'percentage']
        
        M = total_age_human
        n = expected_count
        N = total_test
        x = observed_count
        
        p_value = hypergeom.sf(x - 1, M, n, N)
        enrichment_score = (observed_count / total_test) / (expected_count / total_age_human)
        results.append({'category': category, 'p_value': p_value, 'enrichment_score': enrichment_score})

    results_df = pd.DataFrame(results)

    # Normalize values
    test_data_aligned['normalized_percentage'] = test_data_aligned['percentage']
    age_human_aligned['normalized_percentage'] = age_human_aligned['percentage'] / age_human_aligned['percentage'].sum() * test_data_aligned['normalized_percentage'].sum()

    # Create plot with enhanced styling
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot bars and lines
    bars = ax.bar(test_data_aligned[1],
                  test_data_aligned['normalized_percentage'],
                  alpha=0.6, label=name, color=color)  # Material Design Blue
    # #ax.bar(age_human_aligned[1],
    #        age_human_aligned['normalized_percentage'],
    #        alpha=0.6, label='Age Human Data', color='#FF9800')  # Material Design Orange
    
    # # Add continuous lines
    # ax.plot(test_data_aligned[1], 
    #         test_data_aligned['normalized_percentage'], 
    #         color='#1976D2',  # Darker blue
    #         marker='o', 
    #         linestyle='-', 
    #         linewidth=2,
    #         markersize=8)
    # ax.plot(age_human_aligned[1], 
    #         age_human_aligned['normalized_percentage'], 
    #         color='#F57C00',  # Darker orange
    #         marker='o', 
    #         linestyle='-', 
    #         linewidth=2,
    #         markersize=8)

    # Add significance stars
    for idx, row in results_df.iterrows():
        if row['p_value'] <= 0.05:
            bar_height = bars[idx].get_height()
            ax.text(idx, bar_height + 0.01, '*', 
                   ha='center', va='bottom', 
                   fontsize=36, color='black', fontweight='bold')

    # Customize fonts and labels
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Title with bold font
    ax.set_title('Distribution of Evolutionary Categories', 
                 fontsize=24, 
                 fontweight='bold', 
                 pad=20)

    # X-axis styling
    ax.set_xticks(range(len(evolutionary_order)))
    plt.setp(ax.set_xticklabels(evolutionary_order, 
                               rotation=45, 
                               ha='right', 
                               fontsize=16), 
             fontweight='bold')

    # Y-axis styling
    ax.set_ylabel('Gene count', 
                  fontsize=22, 
                  fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    # Make y-axis labels bold
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Set y-axis limit with some padding
    max_value = max(test_data_aligned['normalized_percentage'].max(),
                   age_human_aligned['normalized_percentage'].max())
    ax.set_ylim(0, max_value * 1.15)

    # Legend styling
    ax.legend(fontsize=18, 
             frameon=True, 
             fancybox=True, 
             shadow=True, 
             loc='upper right',
             prop={'weight': 'bold', 'size': 18})

    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save with high DPI
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    return results_df



def plot_combined_gene_histogram(upregulated, downregulated, age_human_genes, evolutionary_order, output_path, y_max=None):
    """
    Plots a histogram with stacked bars for upregulated and downregulated genes,
    and marks significant categories with a bold, larger star closer to the top.
    
    Parameters:
        upregulated (list): List of upregulated genes.
        downregulated (list): List of downregulated genes.
        age_human_genes (pd.DataFrame): DataFrame with gene symbols and their evolutionary categories.
        evolutionary_order (list): Ordered list of evolutionary categories.
        output_path (str): File path to save the plot.
        y_max (float, optional): Upper limit for the y-axis.
    """
    plt.figure(figsize=(16, 12))  # Increased figure size for better visualization
    
    # Map gene symbols to evolutionary categories
    upregulated_categories = age_human_genes[age_human_genes[0].isin(upregulated)][1]
    downregulated_categories = age_human_genes[age_human_genes[0].isin(downregulated)][1]
    
    # Count occurrences of each category
    upregulated_counts = upregulated_categories.value_counts().reindex(evolutionary_order, fill_value=0)
    downregulated_counts = downregulated_categories.value_counts().reindex(evolutionary_order, fill_value=0)
    total_counts = upregulated_counts + downregulated_counts
    
    # Get background counts from age_human_genes
    background_counts = age_human_genes[1].value_counts().reindex(evolutionary_order, fill_value=0)
    total_background = background_counts.sum()
    
    # Perform over-representation test using background counts
    p_values = []
    for idx, category in enumerate(evolutionary_order):
        observed_count = total_counts[idx]
        expected_count = background_counts[idx]
        p_value = hypergeom.sf(observed_count - 1, total_background, expected_count, total_counts.sum()) if total_background > 0 else 1.0
        p_values.append(p_value)
    
    # Plot stacked bars with red and green, slightly transparent
    bar_width = 0.8
    categories = np.arange(len(evolutionary_order))
    
    plt.bar(categories, upregulated_counts, width=bar_width, color='red', alpha=0.7, label=r'$\bf{Upregulated}$')  # Red with transparency, bold label
    plt.bar(categories, downregulated_counts, width=bar_width, bottom=upregulated_counts, color='green', alpha=0.7, label=r'$\bf{Downregulated}$')  # Green with transparency, bold label
    
    # Mark significant categories with bold, larger stars closer to the top
    for idx, p_value in enumerate(p_values):
        if p_value <= 0.05:
            plt.text(idx, total_counts[idx] + 0.02 * total_counts.max(), '*', fontsize=40, ha='center', color='black', fontweight='bold')
    
    # Formatting
    plt.xticks(categories, evolutionary_order, rotation=90, fontsize=22, fontweight='bold')  # Much bigger x-axis ticks
    plt.yticks(fontsize=22, fontweight='bold')  # Much bigger y-axis ticks
    plt.ylabel("Gene Count", fontsize=26, fontweight='bold')  # Bigger y-axis label
    plt.xlabel("Evolutionary Age", fontsize=26, fontweight='bold')  # Bigger x-axis label
    
    # Adjust legend properties
    legend = plt.legend(fontsize=22, frameon=True, fancybox=True)  # Bigger legend text
    for text in legend.get_texts():
        text.set_fontweight('bold')  # Make legend labels bold
    
    plt.title("Stacked Histogram of Upregulated and Downregulated Genes", fontsize=28, fontweight='bold')  # Bigger title
    
    # Adjust y-axis upper limit if specified
    if y_max is not None:
        plt.ylim(0, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


condition_dfs = []  # At start of section
##############################################
# Analysis age of all human genes, data from https://www.sciencedirect.com/science/article/pii/S0093775418302264#bib0016
##############################################

age_human_genes = pd.read_csv('./database_age_human_genes/1-s2.0-S0093775418302264-mmc2.csv', header=None, delimiter = ';')
age_human_genes = age_human_genes[[0,1]]
age_human_genes = age_human_genes.dropna()
age_human_genes[1] = age_human_genes[1].str.replace(',', '.').astype(float).round()
dict_ages = {
    1: "All living organisms",
    2: "Eukaryota",
    3: "Opisthokonta",
    4: "Holozoa",
    5: "Metazoa",
    6: "Eumetazoa",
    7: "Bilateria",
    8: "Deuterostomia",
    9: "Chordata",
    10: "Olfactores",
    11: "Craniata",
    12: "Euteleostomi",
    13: "Tetrapoda",
    14: "Amniota",
    15: "Mammalia",
    16: "Eutheria",
    17: "Boreoeutheria",
    18: "Euarchontoglires",
    19: "Primates"
}

age_human_genes[1] = age_human_genes[1].map(dict_ages)
name_conditions = ['all_protein_coding_genes']

filtered_df = age_human_genes
# Calculate the percentage of genes per evolutionary age
percentage_df = filtered_df[1].value_counts(normalize=False).reset_index()
percentage_df.columns = [1, 'percentage']
# Order by the evolutionary ages
evolutionary_order = [
    "All living organisms", "Eukaryota", "Opisthokonta", "Holozoa", "Metazoa",
    "Eumetazoa", "Bilateria", "Deuterostomia", "Chordata", "Olfactores",
    "Craniata", "Euteleostomi", "Tetrapoda", "Amniota", "Mammalia",
    "Eutheria", "Boreoeutheria", "Euarchontoglires", "Primates"
]

# Ensure all categories in evolutionary_order are present by reindexing and filling missing with 0
percentage_df = percentage_df.set_index(1).reindex(evolutionary_order, fill_value=0).reset_index()


percentage_df[1] = pd.Categorical(
    percentage_df[1],
    categories=evolutionary_order,
    ordered=True
)
percentage_df = percentage_df.sort_values(1)

# Add a column for condition name for identification in the combined plot
percentage_df['condition'] = name_conditions[0]

# Append this data to the combined DataFrame
combined_percentage_df = pd.DataFrame()
combined_percentage_df = pd.concat([combined_percentage_df, percentage_df])
age_human_genes_df = combined_percentage_df

# Plotting the combined histogram
plt.figure(figsize=(14, 10))
ax = sns.barplot(x=1, y='percentage', hue='condition', data=combined_percentage_df, palette='viridis')
plt.title('Combined Percentage of Genes per Evolutionary Age Across Conditions')
plt.xlabel('Evolutionary Ages')
plt.ylabel('Gene count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend(title='Condition')
plt.tight_layout()
plt.savefig(f'./{results_dir}/combined_genes_evolutionary_ages_histogram.png', dpi=300)
plt.show()

##############################################
# Age gladyshev database  https://age-meta.com/
##############################################

# Load the dataset from the uploaded file
file_path = './database_age_human_genes/aging_diffexpr_data.csv'
df = pd.read_csv(file_path, sep='\t')

# Ensure the dataset has the necessary columns
required_columns = ['entrez', 'genesymbol', 'Brain_pval', 'Human_pval', 'Muscle_pval', 'Liver_pval', 'All_pval']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"The column '{col}' is missing from the dataset")

# Filter for human-specific genes with p-value < 0.05 across tissues
def filter_genes_by_pval(df, pval_column):
    return df[df[pval_column] < 0.05]

conditions = {
    'brain_genes': filter_genes_by_pval(df, 'Brain_adj_pval'),
    'human_genes': filter_genes_by_pval(df, 'Human_adj_pval'),
    'muscle_genes': filter_genes_by_pval(df, 'Muscle_adj_pval'),
    'liver_genes': filter_genes_by_pval(df, 'Liver_adj_pval'),
    'all_genes': filter_genes_by_pval(df, 'All_adj_pval'),
}

# Save each condition to a separate CSV file
for condition_name, condition_data in conditions.items():
    condition_output_path = os.path.join(results_dir, f'{condition_name}.csv')
    condition_data.to_csv(condition_output_path, index=False)
    print(f"Condition '{condition_name}' saved to {condition_output_path}")

# Combine all genes into one DataFrame and drop duplicates
all_filtered_genes = pd.concat(conditions.values()).drop_duplicates()

# Save the combined results to a CSV file
combined_output_path = os.path.join(results_dir, 'combined_filtered_human_genes.csv')
all_filtered_genes.to_csv(combined_output_path, index=False)
print(f"Combined filtered genes saved to {combined_output_path}")

# Display the first few rows of the result
print(all_filtered_genes.head())

# Reload human_genes for further processing
human_genes = conditions['human_genes']

# load mapping
mapping_mouse_human = pd.read_csv('././database_age_human_genes/hcop-1737733278553_649genes_sig.txt', delimiter = '\t' )
mapping_mouse_human = mapping_mouse_human[['Primary symbol', 'Ortholog symbol']]
dict_mapping_mouse_human = {k: list(v) for k,v in mapping_mouse_human.groupby('Primary symbol')['Ortholog symbol']}
dict_mapping_mouse_human =  {k: v[0] for k,v in dict_mapping_mouse_human.items()}
# Translate mouse gene symbols to human gene symbols in human_genes
human_genes['genesymbol'] = human_genes['genesymbol'].apply(lambda x: dict_mapping_mouse_human.get(x, x))

# Display the updated human_genes DataFrame
print(human_genes.head())
# Filter genes based on Human_logFC values
human_genes_up = human_genes[human_genes['Human_logFC'] > 0]
human_genes_down = human_genes[human_genes['Human_logFC'] < 0]

# Save the filtered results to separate CSV files
human_genes_up_path = os.path.join(results_dir, 'human_genes_up.csv')
human_genes_down_path = os.path.join(results_dir, 'human_genes_down.csv')

human_genes_up.to_csv(human_genes_up_path, index=False)
human_genes_down.to_csv(human_genes_down_path, index=False)

# Display summary of the filtering
print(f"Number of upregulated genes: {len(human_genes_up)}")
print(f"Number of downregulated genes: {len(human_genes_down)}")
print(f"Upregulated genes saved to: {human_genes_up_path}")
print(f"Downregulated genes saved to: {human_genes_down_path}")

human_genes = list(human_genes['genesymbol'])
human_genes_up  = list(human_genes_up['genesymbol'])
human_genes_down  = list(human_genes_down['genesymbol'])

conditions_analysis = [human_genes, human_genes_up, human_genes_down]
name_conditions = ['human_genes', 'human_genes_up', 'human_genes_down']
colors = ['#2196F3', 'red','green']


# Initialize an empty DataFrame for combined results
combined_percentage_df = pd.DataFrame()

for i, condition in enumerate(conditions_analysis):
    # Filter the DataFrame based on the gene list
    filtered_df = age_human_genes[age_human_genes[0].isin(condition)]
    
    # Calculate the percentage of genes per evolutionary age
    percentage_df = filtered_df[1].value_counts(normalize=False).reset_index()
    percentage_df.columns = [1, 'percentage']
    
    # Order by the evolutionary ages
    evolutionary_order = [
        "All living organisms", "Eukaryota", "Opisthokonta", "Holozoa", "Metazoa",
        "Eumetazoa", "Bilateria", "Deuterostomia", "Chordata", "Olfactores",
        "Craniata", "Euteleostomi", "Tetrapoda", "Amniota", "Mammalia",
        "Eutheria", "Boreoeutheria", "Euarchontoglires", "Primates"
    ]
    
    # Ensure all categories in evolutionary_order are present by reindexing and filling missing with 0
    percentage_df = percentage_df.set_index(1).reindex(evolutionary_order, fill_value=0).reset_index()

    
    percentage_df[1] = pd.Categorical(
        percentage_df[1],
        categories=evolutionary_order,
        ordered=True
    )
    percentage_df = percentage_df.sort_values(1)
    
    # Add a column for condition name for identification in the combined plot
    percentage_df['condition'] = name_conditions[i]

    # Append this data to the combined DataFrame
    combined_percentage_df = pd.concat([combined_percentage_df, percentage_df])
    condition_dfs.append((name_conditions[i], percentage_df))  # Store for analysis
    
    # Individual histograms
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=1, y='percentage', data=percentage_df, palette='viridis')
    plt.title(f'Percentage of Genes per Evolutionary Age - {name_conditions[i]}')
    plt.xlabel('Evolutionary Ages')
    plt.ylabel('Gene count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(f'./{results_dir}/genes_evolutionary_ages_histogram_{name_conditions[i]}.png', dpi=300)
    plt.show()


    plot_normalized_distributions_with_stats(percentage_df, age_human_genes_df, f'./{results_dir}/genes_evolutionary_ages_histogram_stats_{name_conditions[i]}.png', name_conditions[i], colors[i], lower_offset=-0.5)
    
# Plotting the combined histogram
plt.figure(figsize=(14, 10))
ax = sns.barplot(x=1, y='percentage', hue='condition', data=combined_percentage_df, palette='viridis')
plt.title('Combined Percentage of Genes per Evolutionary Age Across Conditions')
plt.xlabel('Evolutionary Ages')
plt.ylabel('Gene count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.legend(title='Condition')
plt.tight_layout()
plt.savefig(f'./{results_dir}/combined_genes_evolutionary_ages_histogram.png', dpi=300)
plt.show()

plot_combined_gene_histogram(human_genes_up, human_genes_down,age_human_genes, evolutionary_order,  f'./{results_dir}/genes_evolutionary_ages_histogram_stats_combined_gladyshevcellMeta.png', y_max=900)






