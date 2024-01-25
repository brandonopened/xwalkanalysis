import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_source, file_target):
    df_source = pd.read_csv(file_source)
    df_target = pd.read_csv(file_target)
    return df_source, df_target

def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    else:
        return ""

def find_matches(df_source, df_target):
    df_target['combined'] = df_target['humanCodingScheme'].astype(str) + ' ' + df_target['fullStatement'].astype(str)

    df_source['fullStatement'] = df_source['fullStatement'].apply(preprocess_text)
    df_target['combined'] = df_target['combined'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    vec_source = vectorizer.fit_transform(df_source['fullStatement'])
    vec_target = vectorizer.transform(df_target['combined'])

    cosine_similarities = cosine_similarity(vec_source, vec_target)

    results = []
    for idx, row in df_source.iterrows():
        similarity_scores = list(enumerate(cosine_similarities[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Filter scores greater than 0.5 and limit to top 5
        top_matches = [score for score in similarity_scores if score[1] > 0.5][:5]

        for match_idx, score in top_matches:
            results.append({
                'Source_HumanCodingScheme': row['humanCodingScheme'],
                'Source_fullStatement': row['fullStatement'],
                'Target_HumanCodingScheme': df_target.iloc[match_idx]['humanCodingScheme'],
                'Target_FullStatement': df_target.iloc[match_idx]['fullStatement'],
                'Score': score
            })

    return results

# File paths
file_source = 'source.csv'  # Rename your source file to source.csv
file_target = 'target.csv'  # Rename your target file to target.csv

# Load data and find matches
df_source, df_target = load_data(file_source, file_target)
matches = find_matches(df_source, df_target)

# Convert results to a DataFrame
results_df = pd.DataFrame(matches)

# Create an ExcelWriter object
with pd.ExcelWriter('crosswalk.xlsx', engine='xlsxwriter') as writer:
    # Export the results to a 'matches' worksheet
    results_df.to_excel(writer, sheet_name='matches', index=False)

    # Filter the results for matches with a score less than 0.5 (50%)
    low_probability_matches = results_df[results_df['Score'] < 0.5]

    # Get all unique 'humanCodingScheme' values from the target
    all_target_humancodingschemes = set(df_target['humanCodingScheme'].unique())

    # Get 'humanCodingScheme' values from the low probability matches
    matched_humancodingschemes = set(low_probability_matches['Target_HumanCodingScheme'])

    # Determine the missing standards
    missing_standards = all_target_humancodingschemes - matched_humancodingschemes

    # Create a DataFrame for missing standards with corresponding 'fullStatement'
    missing_standards_df = df_target[df_target['humanCodingScheme'].isin(missing_standards)][['humanCodingScheme', 'fullStatement']].drop_duplicates()

    # Export the 'missing standards' worksheet
    missing_standards_df.to_excel(writer, sheet_name='missing standards', index=False)

print("Exported the results to crosswalk.xlsx")