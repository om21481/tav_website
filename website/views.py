from django.shortcuts import render,  HttpResponseRedirect
from django.http import HttpResponse
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly
import subprocess
import zipfile
# from data_downloader import DataDownloader
import os

def home(request):
    return render(request, 'index.html')

# def run_external_script():
#     script_path = '/content/drive/MyDrive/PLATFORM_AUG2024/compute_and_store.py'
#     try:
#         # Run the external script and capture the output
#         result = subprocess.run(['python3', script_path], check=True, text=True, capture_output=True)
#         print("Script output:", result.stdout)  # Output from the script
#         return None  # No error
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred: {e.stderr}")
#         return f"An error occurred while running the script: {e.stderr}"
# # Load the data
file_path = r"C:\Users\denma\Desktop\Repos\Alok Proj\tav_website\website\expanded_embeddings.csv"

# try:
#   df=pd.read_csv(file_path)
# except:
#   run_external_script()
#   df=pd.read_csv(file_path)

# #external_script_path="/content/drive/MyDrive/PLATFORM_AUG2024/compute_and_store.py"

# dim_columns = [f'Dim_{i}' for i in range(1, 385)]




# def create_zip(directory_path, zip_filename):
#     """Create a zip file from the contents of a directory."""
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_STORED) as zipf:
#         for root, _, files in os.walk(directory_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 arcname = os.path.relpath(file_path, directory_path)
#                 zipf.write(file_path, arcname)


def index():
    return render('index.html')

def services():
    return render('services.html')

def data_downloader(request):
    if request.method == 'POST':
        gse_id = request.form.get('gse_id')
        
        # Use the DataDownloader class
        downloader = DataDownloader(gse_id)
        directory_path = downloader.dataDownloader()

        # Ensure directory path is not empty or invalid
        if not os.path.isdir(directory_path):
            return "Invalid directory", 400

        # Create a zip file of the directory contents
        zip_filename = f"{gse_id}.zip"
        create_zip(directory_path, zip_filename)

        # Return the zip file for download
        return send_file(zip_filename, as_attachment=True)

    return render_template('data_downloader.html')

def embeddings(request):
    search_results = None
    pca_plot_div = None
    scree_plot_div = None
    scree_plot_div1 = None
    cosine_plot_div = None 
    cosine_similarities = None
    error_message = None
    disease_name = None
    cutoff_value = None

    if request.method == 'POST':
        df=pd.read_csv(file_path)
        dim_columns = [f'Dim_{i}' for i in range(1, 385)]

        disease_name = request.POST.get('disease_name')
        print("disease name: ", disease_name)
        search_results = df[df['disease'].str.contains(disease_name, case=False, na=False)]

        if not search_results.empty:
            # Keep only unique GSE_IDs
            unique_results = search_results.drop_duplicates(subset='GSE_ID')
            embeddings = unique_results[dim_columns].values
            n_samples, n_features = embeddings.shape

            # Check if PCA with at least 2 components is feasible
            n_components = min(3, n_samples, n_features)
            n_components1 = min(n_samples, n_features)
            if n_components < 2:
                error_message = f"Not enough data for PCA with {n_components} components. Available samples: {n_samples}, features: {n_features}."
            else:
                # Perform PCA
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(embeddings)
                explained_variance = pca.explained_variance_ratio_
                
                pca1 = PCA(n_components=n_components1)
                principal_components1 = pca1.fit_transform(embeddings)
                explained_variance1 = pca1.explained_variance_ratio_

                # Create 3D PCA plot if n_components >= 3
                if n_components >= 3:
                    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
                    pca_df['disease'] = unique_results['disease'].tolist()
                    pca_df['GSE_ID'] = unique_results['GSE_ID'].tolist()

                    # Combine disease and GSE_ID for hover information
                    pca_df['disease_with_id'] = pca_df.apply(lambda row: f"{row['disease']} (GSE_ID: {row['GSE_ID']})", axis=1)

                    fig_pca = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                                            color='disease_with_id', hover_name='disease_with_id', title='3D PCA Plot', width=2000, height=600)
                    pca_plot_div = json.dumps(fig_pca, cls=plotly.utils.PlotlyJSONEncoder)
                # Create 2D   if n_components == 2
                elif n_components == 2:
                    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])
                    pca_df['disease'] = unique_results['disease'].tolist()
                    pca_df['GSE_ID'] = unique_results['GSE_ID'].tolist()

                    # Combine disease and GSE_ID for hover information
                    pca_df['disease_with_id'] = pca_df.apply(lambda row: f"{row['disease']} (GSE_ID: {row['GSE_ID']})", axis=1)

                    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='disease_with_id', hover_name='disease_with_id', title='2D PCA Plot', width=2000, height=600)
                    pca_plot_div = json.dumps(fig_pca, cls=plotly.utils.PlotlyJSONEncoder)

                # Create scree plot
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(x=list(range(1, len(explained_variance) + 1)), y=explained_variance, name='Individual'))
                fig_scree.add_trace(go.Scatter(x=list(range(1, len(explained_variance) + 1)), y=np.cumsum(explained_variance), mode='lines+markers', name='Cumulative'))
                fig_scree.update_layout(title='Scree Plot', xaxis_title='Principal components', yaxis_title='Explained variance ratio')
                scree_plot_div = json.dumps(fig_scree, cls=plotly.utils.PlotlyJSONEncoder)
                
                fig_scree1 = go.Figure()
                fig_scree1.add_trace(go.Bar(x=list(range(1, len(explained_variance1) + 1)), y=explained_variance1, name='Individual'))
                fig_scree1.add_trace(go.Scatter(x=list(range(1, len(explained_variance1) + 1)), y=np.cumsum(explained_variance1), mode='lines+markers', name='Cumulative'))
                fig_scree1.update_layout(title='Scree Plot', xaxis_title='Principal components', yaxis_title='Explained variance ratio')
                scree_plot_div1 = json.dumps(fig_scree1, cls=plotly.utils.PlotlyJSONEncoder)

                # Perform cosine similarity
                similarity_matrix = cosine_similarity(embeddings)

                # Create a heatmap with GSE IDs as labels
                fig_cosine = go.Figure(data=go.Heatmap(
                    z=similarity_matrix,
                    x=unique_results['GSE_ID'],
                    y=unique_results['GSE_ID'],
                    colorscale='Viridis'
                ))
                fig_cosine.update_layout(title='Cosine Similarity Matrix', height=1200)
                cosine_plot_div = json.dumps(fig_cosine, cls=plotly.utils.PlotlyJSONEncoder)

                # Get cosine similarity data with names
                disease_names = unique_results['disease'].tolist()
                gse_ids = unique_results['GSE_ID'].tolist()
                cosine_similarities = []
                cutoff_value = float(request.POST.get('cutoff', 0.7))
                for i in range(len(disease_names)):
                    for j in range(i + 1, len(disease_names)):
                        similarity_value = round(similarity_matrix[i, j], 3)
                        if cutoff_value is None or similarity_value >= cutoff_value:
                            cosine_similarities.append({
                                'disease_1': disease_names[i],
                                'gse_id_1': gse_ids[i],
                                'disease_2': disease_names[j],
                                'gse_id_2': gse_ids[j],
                                'similarity': similarity_value
                            })
                
                # Sort cosine similarities from highest to lowest
                cosine_similarities = sorted(cosine_similarities, key=lambda x: x['similarity'], reverse=True)

    # return render(request, 'index.html', {disease_name=disease_name, search_results=search_results, pca_plot_div=pca_plot_div, scree_plot_div=scree_plot_div, scree_plot_div1=scree_plot_div1, cosine_plot_div=cosine_plot_div, cosine_similarities=cosine_similarities, error_message=error_message, cutoff_value=cutoff_value})
        context = {
        'disease_name': disease_name,
        'search_results': search_results,
        'pca_plot_div': pca_plot_div,
        'scree_plot_div': scree_plot_div,
        'scree_plot_div1': scree_plot_div1,
        'cosine_plot_div': cosine_plot_div,
        'cosine_similarities': cosine_similarities,
        'error_message': error_message,
        'cutoff_value': cutoff_value
        }

        return render(request, 'index.html', context)

def data_generator():
    return render('data_generator.html')

def data_analyzer():
    return render('data_analyzer.html')

def all_in_one():
    return render('all_in_one.html')