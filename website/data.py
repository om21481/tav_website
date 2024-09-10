import os
import re
import shutil
import gzip
import wget
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as PdfPages
from collections import Counter
from scipy.stats import ttest_ind
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
import statsmodels.stats.multitest as smm
import qnorm
from goatools import obo_parser
from nltk.corpus import stopwords
import GEOparse
import multiprocessing
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
# Ensure the necessary packages are installed
import nltk
import spacy
from gprofiler import GProfiler
nltk.download('stopwords')
nlp = spacy.load("en_core_sci_sm")
from urllib.request import urlopen
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages

import zipfile


def list_directories(folder_path):
    all_entries = os.listdir(folder_path)
    directories = [entry for entry in all_entries if os.path.isdir(os.path.join(folder_path, entry))]
    return directories

class DataDownloader:
    def __init__(self, gse_id: str,dir_path:str=None):
        self.gse_id = gse_id
        try:
            self.dir_path = dir_path if dir_path else os.getcwd()
        except Exception as e:
            print(f"An error occurred while setting the directory path: {e}")
            self.dir_path = None  # Assign None or a default value if an error occurs

        try:
            self.gse = GEOparse.get_GEO(self.gse_id)
        except Exception as e:
            print(f"An error occurred while initializing GEOparse object: {e}")
            self.gse = None  # Assign None or handle the error as needed

    def dataDownloader(self) -> str:
        """Download and process data from GEO based on the provided GSE ID and return the directory where files are saved."""
        try:

            saved_dir=os.path.join(self.dir_path, self.gse_id)
            try:
                if os.path.exists(saved_dir):
                  shutil.rmtree(saved_dir)
                  print(f"Removed existing directory: {saved_dir}")
                os.makedirs(saved_dir, exist_ok=True)
            except OSError as e:
                print(f"An error occurred while creating directory {saved_dir}: {e}")
                return None
            try:
                os.chdir(saved_dir)
            except OSError as e:
                print(f"An error occurred while changing directory to {saved_dir}: {e}")
                return None


            # Download GEO data

            # self.gse = GEOparse.get_GEO(self.gse_id)
            geo_type = self.gse.metadata['platform_id'][0]


            # Process data based on GEO platform ID
            data = ''
            if geo_type == 'GPL18573':
                data = self.RNA_Count_Data()
            elif geo_type == 'GPL10558':
                data = self.Data_Processing_and_downloading_Illumina()
            elif geo_type == 'GPL6480':
                data = self.Agilent_data_preparation()
            elif geo_type == 'GPL570':
                data = self.Data_Processing_and_downloading_Affy()
            else:
                raise ValueError(f"Unsupported GEO platform ID: {geo_type}")
            try:
                os.chdir(self.dir_path)
            except OSError as e:
                print(f"An error occurred while changing directory to {self.dir_path}: {e}")
                return None



        except Exception as e:
            print(f"An error occurred: {e}")
            saved_dir = ''
        print(f"Data for the {self.gse_id} saved to the {saved_dir}")

        return saved_dir

    def Meta_Data(self, gse_id,gse):
            """Retrieve and process meta data."""
            #gse = GEOparse.get_GEO(geo=gse_id)

            pheno = gse.phenotype_data
            val = gse.metadata['summary'][0]
            meta = self.Meta_Data_Processed(pheno, val)
            meta = self.clean_column_names(meta)
            return meta

    def clean_column_names(self, df):
            """Clean column names by replacing non-alphanumeric characters with underscores."""
            df.columns = [re.sub(r'[^A-Za-z0-9]+', '_', col) for col in df.columns]
            return df

    def Meta_Data_Processed(self, data, term):

      data = self.Data_Discrepantic_col(data)
      data.replace('NA', 'Not Available', inplace=True)
      data.replace(pd.NA, 'Not Available', inplace=True)
      data_p = self.Numeric_column(data)
      data_1 = self.Data_Target_Variable(data_p, term)  # Placeholder method
      data_2 = self.Binary_data(data)  # Placeholder method
      pheno_processed = pd.concat([data_1, data_2], axis=1)
      pheno_processed = self.Date_removal(pheno_processed)  # Placeholder method
      x = pheno_processed.T
      pheno_processed = x.loc[~x.index.duplicated(keep='first')].T
      cols = list(pheno_processed.columns)
      req_cols = [i for i in cols if pheno_processed[i].nunique() != 1]
      data_req = pheno_processed[req_cols]
      data_req = self.filter_columns_by_na_threshold(data_req, threshold=0.2)  # Placeholder method
      return data_req

    def filter_columns_by_na_threshold(self, df, threshold=0.4):
        """Filter columns based on 'Not Available' values."""
        na_proportion = df.apply(lambda col: (col == 'Not Available').sum() / len(df))
        filtered_df = df.loc[:, na_proportion <= threshold]
        return filtered_df

    def Data_Discrepantic_col(self, data):
        """Handle columns with similar information."""
        req_data = self.extract_columns_with_same_last_word(data)
        merged_data = self.merge_columns_with_same_info(req_data)
        data = data.drop(columns=req_data.columns)
        data = pd.concat([data, merged_data], axis=1)
        return data

    def extract_columns_with_same_last_word(self, df):
        """Extract columns with similar last word."""
        last_words = df.columns.str.split('[.]').str[-1]
        d = dict(zip(list(df.columns), last_words))
        count = dict(Counter(last_words))
        more = [i for i in count.keys() if count[i] != 1]
        cols = [k for k, v in d.items() if v in more]
        return df[cols]

    def merge_columns_with_same_info(self, data):
        """Merge columns with similar variable info."""
        column_last_words = data.columns.str.split('[._]').str[-1]
        columns_dict = {}
        for col, last_word in zip(data.columns, column_last_words):
            if last_word not in columns_dict:
                columns_dict[last_word] = []
            columns_dict[last_word].append(col)
        merged_df = pd.DataFrame()
        for last_word, cols in columns_dict.items():
            if len(cols) > 1:
                merged_df[last_word] = data[cols].bfill(axis=1).iloc[:, 0]
            else:
                merged_df[last_word] = data[cols[0]]
        return merged_df

    def Numeric_column(self, data):
        """Identify and return numeric columns."""
        numeric_columns = [col for col in data.columns if self.is_column_numeric(data[col])]
        cols = list(set(data.columns).difference(set(numeric_columns)))
        df_numeric = data[cols]
        return df_numeric

    def is_column_numeric(self, col):
        """Check if a column is numeric."""
        try:
            col.astype(float)
            return True
        except ValueError:
            return False

    def Binary_data(self, data):
        """Return binary columns."""
        binary_columns = [col for col in data.columns if self.is_binary(data[col])]
        return data[binary_columns]

    def is_binary(self, col):
        """Check if a column is binary."""
        unique_values = col.unique()
        return len(unique_values) in [2, 3]

    def Date_removal(self, data):
        """Remove date columns."""
        date_columns = [col for col in data.columns if self.is_date_column(data[col])]
        return data.drop(columns=date_columns)

    def is_date_column(self, series):
        """Check if a column is a date column."""
        try:
            pd.to_datetime(series)
            return True
        except ValueError:
            return False






    def Data_Processing_and_downloading_Illumina(self):
            """Process and download Illumina data."""
            path = os.getcwd()

            # Fetch and process data
            data = self.GSE_Data_Preparation(self.gse_id,self.gse)  # Placeholder method
            #print(data)
            data = data.T

            meta = self.Meta_Data(self.gse_id,self.gse)  # Placeholder method
            missing_percentage = meta.isnull().mean()
            meta = meta.loc[:, missing_percentage <= 0.40]

            probe_data = self.probe_to_gene_id(self.gse_id)  # Placeholder method
            data = self.Gene_mapping_Illumia(data, probe_data)  # Placeholder method
            cols = list(meta.columns)
            #print("Columns:", cols)

            # Processing and saving data
            for i in cols:
                file_path = os.path.join(path, i)
                data['Condition'] = list(meta[i])
                data_req = self.Relevant_Data(data, 'Condition', top_n=10)  # Placeholder method
                if data_req.shape[0] > 10:
                    if data_req.isna().sum().sum() != 0:
                        data_req = self.impute_missing_values(data_req, condition_col='Condition')  # Placeholder method
                        data_req = self.drop_unwanted_columns(data_req)  # Placeholder method
                    os.mkdir(file_path)
                    curr_dir = os.path.join(path, file_path)
                    os.chdir(curr_dir)
                    data_req.to_csv(self.gse_id + '_' + i + '.csv')
            os.chdir(path)
            print('Data Generation done')
            return



    def GSE_Data_Preparation(self, gse_id,gse):
            """Prepare data from GEO."""
            #gse = GEOparse.get_GEO(geo=gse_id)

            supp = gse.metadata['supplementary_file']
            #print("Supplementary Files:", supp)
            sampl = list(gse.gsms)
           # print("Samples:", sampl)
            values = []
            probe_id = list(gse.gsms[sampl[0]].table['ID_REF'])
            #print("Probe IDs:", probe_id)
            for i in sampl:
                values.append(gse.gsms[i].table['VALUE'])
           # print("Length of Probe IDs:", len(probe_id))
            #print("Values:", values)
            df = pd.DataFrame(values)
            df = df.T
            df.columns = sampl
            df.index = probe_id
            return df

    def probe_to_gene_id(self, gse_id: str) -> pd.DataFrame:
        """Download and parse the probe-to-gene ID mapping file for the given GEO series ID."""
        #print(f"Processing GSE ID: {gse_id}")

        base_path = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/soft/"
        #print(f"Base path: {base_path}")

        # Fetch file names from the directory
        try:
            data = urlopen(base_path).read().decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data from {base_path}: {e}")

        # Extract file names from the HTML content
        file_links = [line.split('"')[1] for line in data.split() if 'href=' in line and line.endswith('</a>')]

        if not file_links:
            raise ValueError(f"No files found for GSE ID {gse_id}")

        # Determine which file to download
        if len(file_links) == 1:
            file_name = file_links[0]
        else:
            print("Multiple files found:")
            for idx, link in enumerate(file_links):
                print(f"{idx}: {link}")
            try:
                index = int(input("Select the file index: "))
                file_name = file_links[index]
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid index provided: {e}")

        file_path = base_path + file_name
        response = wget.download(file_path)

        # Decompress the file if necessary
        if response.endswith('.gz'):
            with gzip.open(response, 'rb') as f_in:
                with open(response[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(response)
            response = response[:-3]

        # Read and parse the file content
        with open(response, 'r') as file:
            content = file.readlines()

        # Find the start and end of the platform table
        start, end = 0, 0
        for idx, line in enumerate(content):
            if '!platform_table_begin' in line:
                start = idx
            if '!platform_table_end' in line:
                end = idx

        # Convert the platform table to a DataFrame
        df = pd.DataFrame([line.split("\t") for line in content[start+2:end]])
        df.columns = content[start+1].split('\t')

        return df



    def quantile_normalization_geo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform quantile normalization on the input dataframe using the qnorm package.

        Parameters:
        df (pd.DataFrame): DataFrame where rows are samples and columns are genes.

        Returns:
        pd.DataFrame: Quantile normalized DataFrame.
        """
        try:
            # Perform quantile normalization
            normalized_values = qnorm.quantile_normalize(df.values)

            # Create a DataFrame with the normalized values
            normalized_df = pd.DataFrame(normalized_values, index=df.index, columns=df.columns)
            return normalized_df
        except Exception as e:
            print(f"An error occurred in quantile_normalization_geo_data: {e}")
            return pd.DataFrame()

    def Gene_mapping_Illumia(self, data: pd.DataFrame, probe_data: pd.DataFrame) -> pd.DataFrame:

        # print("this is probe data\n", probe_data)
        # print("this is data\n", data)

        # Process the data
        id = list(data.columns)
        f = probe_data[probe_data['ID'].isin(id)]
        #print("this is f\n", f)

        data = data[list(f['ID'])]
        data.columns = list(f['ILMN_Gene'])
        x = data.T
        #print("this is x\n", x)

        x = x.reset_index()
        x = x.groupby('index').mean()
        max_val = max(x.max())
        if max_val > 100:
            min_val = min(x.min())
            x = x - min_val
            x = np.log2(x + 1)

        # Use the class method for quantile normalization
        x = self.quantile_normalization_geo_data(x)
        x = x.T

        empty_named_columns = [col for col in x.columns if col == '']
        x.drop(columns=empty_named_columns, inplace=True)
        #print("this is from Gene_mapping_Illumia\n", x)
        return x
    def Data_Target_Variable(self, data, term):

            columns = []
            string_columns = data.select_dtypes(include='object').columns.tolist()
            data = data[string_columns]

            for i in list(data.columns):
                if i.startswith('char'):
                    columns.append(i)

            data = data[columns]
            req_cols = []

            for i in columns:
                name = self.Term_Identifier(data, i, term)  # Placeholder method
                req_cols.append(name)

            req_cols = [i for i in req_cols if len(i) != 0]
            data_p = data[req_cols]
            return data_p

    def term_identifier(self, data, col, term):
        """Identify columns matching a provided term."""
        val = self.column(data, col)
        term = self.mti(term)
        pattern = r'[\/\+\.]'
        terms = []
        for i in term:
            words = re.split(pattern, i)
            for j in words:
                terms.append(j)
        common = set(terms).intersection(set(val))
        col_name = ''
        if len(common) != 0:
            col_name = col
        return col_name

    def mti(self, text):
        """Extract terms from text using named entity recognition."""
        ents = nlp(text).ents
        ents = list(ents)
        ents = [str(i) for i in ents]
        ents = [i.lower() for i in ents]
        Terms = []
        if len(ents) != 0:
            for i in ents:
                i = i.split(' ')
                for j in i:
                    Terms.append(j)
        return Terms

    def column(self, data, col):
        """Extract column values from DataFrame."""
        return data[col].tolist()

    def Relevant_Data(self, df: pd.DataFrame, column_name: str, top_n: int = 10) -> pd.DataFrame:

        """Extract relevant data based on TF-IDF scores."""
        try:
            def preprocess_text(text):
                if pd.isnull(text) or text.lower() in ['none', 'na', 'nan', 'not available']:
                    return ''
                text = text.lower()
                text = re.sub(r'[^a-zA-Z0-9]', '_', text)
                return text

            df['Condition'] = df[column_name].apply(preprocess_text)
            df = df[df['Condition'] != '']
            combined_text = " ".join(df['Condition'].tolist())
            vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray().flatten()
            tfidf_df = pd.DataFrame({'word': feature_names, 'score': tfidf_scores})
            tfidf_df = tfidf_df.sort_values(by='score', ascending=False)
            top_keywords = tfidf_df.head(top_n)

            return df

        except Exception as e:
            print(f"An error occurred in Relevant_Data: {e}")
            return df




    def Term_Identifier(self, data, col, term) -> str:
        """Identify if the column name matches the terms extracted from the provided term description."""
        val = self.Column(data, col)
        term_list = self.MTI(term)
        pattern = r'[\/\+\.]'
        terms = []

        for i in term_list:
            words = re.split(pattern, i)
            terms.extend(words)

        common = set(terms).intersection(set(val))
        return col if len(common) != 0 else ''

    def Column(self, data, col) -> list:
        """Extract terms from the column values."""
        val = list(data[col])
        result = [self.MTI(i) for i in val]
        return [j for sublist in result for j in sublist]

    def MTI(self, text) -> list:
        """Extract named entities from the text and return as a list of terms."""
        ents = nlp(text).ents
        ents = [str(i).lower() for i in ents]
        terms = [j for i in ents for j in i.split(' ')]
        return terms

    def impute_missing_values(df, condition_col='Condition'):
      # Drop columns with more than 40% missing values
      missing_percentage = df.isnull().mean()
      df = df.loc[:, missing_percentage <= 0.40]

      # Split the dataframe by the condition column
      groups = df.groupby(condition_col)

      # List to hold the processed dataframes
      processed_dfs = []
      # List to hold columns that have NaN mean in any group
      cols_to_drop = set()

      # Process each group
      for name, group in groups:
          group = group.iloc[:, :-1]  # Assuming last column is the condition column
          all_nan_cols = group.columns[group.isnull().all()].tolist()
          cols_to_drop.update(all_nan_cols)

          group.fillna(group.mean(), inplace=True)

          nan_mean_cols = group.columns[group.mean().isnull()].tolist()
          cols_to_drop.update(nan_mean_cols)

          processed_dfs.append(group)

      # Drop columns that have NaN mean in any group or all NaN values
      for i, processed_df in enumerate(processed_dfs):
          processed_dfs[i] = processed_df.drop(columns=list(cols_to_drop))

      # Combine the processed groups back into a single dataframe
      result_df = pd.concat(processed_dfs)
      result_df[condition_col] = list(df[condition_col])  # Restore the condition column

      return result_df

    def quantile_normalization_geo_data(self, df: pd.DataFrame) -> pd.DataFrame:
      try:
        normalized_values = qnorm.quantile_normalize(df.values)
        normalized_df = pd.DataFrame(normalized_values, index=df.index, columns=df.columns)
        return normalized_df

      except Exception as e:
        print(f"An error occurred in quantile_normalization_geo_data: {e}")
        return pd.DataFrame()
      
# Data Generator class

class DataGenerator:
    def __init__(self, origPath: str = None, targetClass: str = 'Condition',
                 pcaVariance: float = 0.95, minComponents: int = 2,
                 maxComponents: int = 10, minValue: int = 3,
                 timesVector: list = None):
        # Use the provided path or default to the current working directory
        self.origPath = origPath or os.getcwd()
        self.targetClass = targetClass
        self.pcaVariance = pcaVariance
        self.minComponents = minComponents
        self.maxComponents = maxComponents
        self.minValue = minValue
        self.timesVector = timesVector if timesVector else list(range(3, 6, 3))

    def dataGenerator(self):
        """Process each directory and apply the data generation logic."""
        try:
            # savedDir = os.path.join(self.origPath, "GENERATED_DATA")
            # try:
            #     if os.path.exists(savedDir):
            #         shutil.rmtree(savedDir)
            #         print(f"Removed existing directory: {savedDir}")
            #     os.makedirs(savedDir, exist_ok=True)
            # except OSError as e:
            #     print(f"An error occurred while creating directory {savedDir}: {e}")
            #     return None
            # try:
            #     os.chdir(savedDir)
            # except OSError as e:
            #     print(f"An error occurred while changing directory to {savedDir}: {e}")
            #     return None

            folders = self.listDirectories()
            #print("Folders:", folders)
            for folder in folders:
                folderPath = os.path.join(self.origPath, folder)
                os.chdir(folderPath)
                files = os.listdir()
                if files:
                    dataFile = files[0]
                    #print("Processing file:", dataFile)
                    self.dataGeneration(dataFile)
            os.chdir(self.origPath)
            return self.origPath
        except Exception as e:
            print(f"Error in dataGenerator: {e}")
            return None

    def listDirectories(self) -> list:
        """List all directories in the given folder."""
        try:
            allEntries = os.listdir(self.origPath)
            directories = [entry for entry in allEntries if os.path.isdir(os.path.join(self.origPath, entry))]
            return directories
        except Exception as e:
            print(f"Error listing directories: {e}")
            return []

    def classCorrector(self, df: pd.DataFrame, data: pd.DataFrame, className: str) -> pd.DataFrame:
        """Correct class labels based on the original data."""
        try:
            minVal = df[className].min()
            maxVal = df[className].max()
            for i in range(len(data)):
                if data[className][i] <= minVal:
                    data[className][i] = int(minVal)
                elif data[className][i] >= maxVal:
                    data[className][i] = int(maxVal)
                else:
                    data[className][i] = int(round(data[className][i]))
            data[className] = data[className].fillna(0).astype(int)
            return data
        except Exception as e:
            print(f"Error in classCorrector: {e}")
            return data

    def dataGeneration(self, dataPath: str):
        """Generate synthetic data based on PCA and Gaussian Mixture Models."""
        try:
            df = pd.read_csv(dataPath)
            print("Initial data:\n", df.head())

            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            if df.shape[0] < 10:
                print('Number of samples should be at least 10')
                return

            cols = df.columns

            labelEncoder = preprocessing.LabelEncoder()
            if self.targetClass in df.columns:
                df[self.targetClass] = labelEncoder.fit_transform(df[self.targetClass])
            else:
                print(f'{self.targetClass} column not found')
                return

            targetCol = df[self.targetClass]
            df.drop(self.targetClass, axis=1, inplace=True)
            print("Data types after dropping target class:", df.dtypes)

            numSamp = df.shape[0]
            pca = PCA(self.pcaVariance)
            df = pca.fit_transform(df)
            df = pd.DataFrame(df)

            df[self.targetClass] = targetCol
            df.columns = df.columns.astype(str)
            nComponentsRange = np.arange(self.minComponents, self.maxComponents)
            print("Number of components range:", nComponentsRange)
            print("PCA data:", df)

            models = [GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(df) for n in nComponentsRange]
            print("Minimum BIC for {} components".format(self.minValue))

            model = GaussianMixture(n_components=self.minValue, covariance_type="full", random_state=0)
            model.fit(df)
            print("Model Convergence:", model.converged_)

            for i in self.timesVector:
                dataNew = model.sample(i * numSamp)
                dataNew = pd.DataFrame(dataNew[0])
                targetCol = dataNew.iloc[:, -1]
                dataNew = dataNew.iloc[:, :-1]
                digitsNew = pca.inverse_transform(dataNew)
                digitsNew = pd.DataFrame(digitsNew)
                digitsNew[self.targetClass] = targetCol

                self.classCorrector(df, digitsNew, self.targetClass)
                digitsNew[self.targetClass] = labelEncoder.inverse_transform(digitsNew[self.targetClass])
                digitsNew.columns = cols
                digitsNew.to_csv(f'GMM_augmented_data_{i}_times.csv', index=True)

            print('Data Generation done')
            return 0
        except Exception as e:
            print(f"Error in dataGeneration: {e}")
            return -1
        


go_obo_url = "/Users/omgarg/Desktop/projects/techmedbuddy/tav_website/website/go-basic.obo"
go = obo_parser.GODag(go_obo_url)

class Analysis:
    

    def __init__(self, directory,gse_id):
        self.gse_id = gse_id
        self.directory = directory

    def process_directory(self):
        print("path from process_directory",self.directory)
        # Set the working directory
        os.chdir(self.directory)
        orig, syn = self.Files_List(os.getcwd())

        # Multiprocess the files
        self.multiprocess_files(orig, syn)
        self.original_data_analysis(orig)
        self.one_Hot_Matrix_Pathways(self.directory)
        self.final_enrichment_file(self.directory)
        self.all_analysis(self.directory)
        self.zipped_Result(self.gse_id, self.directory)

    def Files_List(self,path):# used
      os.chdir(path)
      files=os.listdir()
      files=[i for i in files if i.endswith('.csv')]
      orig,syn='',[]
      for i in files:
          f=i.split('.')[0]
          if f.endswith('times'):
              syn.append(i)
          else:
              orig=i
      return orig,syn

    def multiprocess_files(self,orig_file_name, files_list): #used#
      # Number of parallel processes
      num_processes = 10
      # Create a multiprocessing Pool
      with multiprocessing.Pool(processes=num_processes) as pool:
          # Prepare the arguments for each process
          args = [(orig_file_name, syn_file_name) for syn_file_name in files_list]
          # print("args",args)
          # print("args type",type(args))
          # print("args len",len(args))
          # Map the list of files to the function that processes them
          pool.map(self.process_file, args)

    def process_file(self, args):
      orig_file_name, syn_file_name = args
      print(orig_file_name, syn_file_name)
      self.Data_Evaluation(orig_file_name, syn_file_name)


    def Data_Evaluation(self, orig_file_name, syn_file_name):
        orig_path = os.getcwd()
        f_name = 'Augmented_Data_Analysis_' + syn_file_name[:-4]
        orig_data, syn_data = self.File_Input(orig_file_name, syn_file_name)

        # Create the directory if it doesn't exist
        os.makedirs(f_name, exist_ok=True)
        os.chdir(os.path.join(os.getcwd(), f_name))

        self.Augmentation_Transcriptome_Data_Performance(orig_data, syn_data)
        os.chdir(orig_path)
        print('Analysis Done')
        return


    def File_Input(self, orig_file_name, syn_file_name):
        orig_data = pd.read_csv(orig_file_name)
        syn_data = pd.read_csv(syn_file_name)
        if orig_data.columns[0] == 'Unnamed: 0':
            orig_data = orig_data.iloc[:, 1:]
        if syn_data.columns[0] == 'Unnamed: 0':
            syn_data = syn_data.iloc[:, 1:]
        return orig_data, syn_data



    def Augmentation_Transcriptome_Data_Performance(self, orig_data, syn_data):
        data_path = os.getcwd()
        orig_data, syn_data = self.Zero_Removal(orig_data, syn_data)
        top_per_genes = 0.01
        per_gene_sel = 0.2
        con_vec = dict(Counter(orig_data['Condition']))
        class_1, class_2 = list(dict(sorted(con_vec.items(), key=lambda x: x[1], reverse=True)[:2]).keys())
        freq = dict(Counter(syn_data.iloc[:, -1]))
        freq_df = pd.DataFrame(dict(classes=freq.keys(), Freq=freq.values()))
        freq_df.to_csv('Frequency_classes.csv')

        aug_data = self.data_merge(orig_data, syn_data)
        aug_data_df = self.gene_selection_var_based(aug_data, per_gene_sel)
        gr, ls, eq = self.Sample_level_DEA(aug_data_df)
        cl1_cl2_dea = self.P_val_class_wise(aug_data_df, class_1, class_2)
        print(data_path)
        self.Multi_P_Val_Enrichment(data_path)
        top_genes, dea_res = self.top_genes_selected(cl1_cl2_dea, top_per_genes)
        print("top genes\n", top_genes)
        print("dea_res\n", dea_res)

        os.chdir(data_path)
        print(os.getcwd())
        dea_res.to_csv('DEA for Classes ' + str(class_1) + ' vs ' + str(class_2) + '.csv')
        self.enrichment_overreprsentation_top(dea_res, class_1, class_2)
        self.volcano_plot(cl1_cl2_dea, class_1, class_2)
        return



    def Zero_Removal(self,orig_data,syn_data):#used
      x=dict(orig_data.iloc[:,:-1].sum())
      gene_zero=[]
      for i in x:
          if x[i]==0:
              gene_zero.append(i)
      non_zero=set(list(orig_data.columns[:-1])).difference(set(gene_zero))
      req_cols=list(non_zero)+[orig_data.columns[-1]]
      orig_df =  orig_data[req_cols]
      syn_df  =  syn_data[req_cols]
      return orig_df,syn_df


    def data_merge(self,orig,syn):#used#
      '''
      orig and synthetic data should have the same column order
      Returns the combined data  also reaname the index '''
      orig_df = orig.copy()
      syn_df=syn.copy()
      syn_df = syn_df[orig_df.columns]
      orig_id=['Orig_Sample_'+str(i) for i in range(1,len(orig_df)+1)]
      syn_id=['Syn_Sample_'+str(i) for i in range(1,len(syn_df)+1)]
      orig_df.index = orig_id
      syn_df.index=syn_id
      merged_data=pd.concat([orig_df,syn_df],axis=0)
      return merged_data


    def gene_selection_var_based(self,data,per_gene_rem):#used#
      '''data = Augmented data
      per_gene_rem = Percenta of genes to remove
      This function helps to remove a selected percent of genes based upon variance'''
      data_df = data.copy()
      target_var_name=data_df.columns[-1]
      target_val = list(data_df.iloc[:,-1])
      data_val = data_df.iloc[:,:-1]
      genes_variance=dict(data_val.var())
      genes_var_sort=sorted(genes_variance.items(),key=lambda x:x[1])
      n=int(per_gene_rem*len(genes_var_sort))
      genes_selected=genes_var_sort[n+1:]
      genes=list(dict(genes_selected).keys())
      data_selected=data_df[genes]
      data_selected[target_var_name] = target_val
      return data_selected



    def Sample_level_DEA(self,aug_data):#used#
        '''This function performs t test for the genes between the original and the synthetic data
        aug_data - Augmented data i.e. Original + synthetic data
        Returns :
        The list of genes having p values greater than 0.05, less than 0.05 and equal to 0.05'''
        aug_data = aug_data.copy()
        data_id=pd.Series(list(aug_data.index)).apply(lambda x:x[:4])
        Data_Type=[]
        for i in data_id:
            if i.startswith('O'):
                Data_Type.append('ORIG')
            else:
                Data_Type.append('SYN')
        aug_data['Data_Type']=Data_Type
        gene_list = list(aug_data.columns[:-2])
        Genes_p_val_gr,Genes_p_val_less,Genes_p_val_eq={},{},{}
        aug_orig=aug_data[aug_data['Data_Type']=='ORIG']    #original dataframe
        aug_syn=aug_data[aug_data['Data_Type']=='SYN']      #Synthetic dataframe
        for i in gene_list: #i=Gene name
            orig_gene=list(aug_orig[i])
            syn_gene=list(aug_syn[i])
            p_val = ttest_ind(orig_gene,syn_gene)[1]
            if p_val>0.05:
                Genes_p_val_gr[i] = p_val
            elif p_val<0.05:
                Genes_p_val_less[i] = p_val
            else:
                Genes_p_val_eq[i] = p_val
        print('No. of genes generated with non significant difference are: ',len(Genes_p_val_gr))
        print('No. of genes generated with significant difference are: ',len(Genes_p_val_less))
        print('No. of genes generated with P val equals 0: ',len(Genes_p_val_eq))
        gr=pd.DataFrame(dict(Genes_Gr_pt_5=list(Genes_p_val_gr.keys())))
        ls=pd.DataFrame(dict(Genes_Ls_pt_5=list(Genes_p_val_less.keys())))
        eq=pd.DataFrame(dict(Genes_eq_0=list(Genes_p_val_eq.keys())))
        return gr,ls,eq



    def P_val_class_wise(self,aug_data, class_1, class_2):
        aug_data = self.quantile_normalizing(aug_data)
        target_var_name = list(aug_data.columns)[-1]
        classes = [class_1, class_2]
        aug_data_df = aug_data[aug_data[target_var_name].isin(classes)]
        gene_list = list(aug_data.columns[:-1])
        class_1_df = aug_data_df[aug_data_df[target_var_name] == class_1]
        class_2_df = aug_data_df[aug_data_df[target_var_name] == class_2]

        lfc_val = []
        P_Value = []

        for i in gene_list:
            class_1_gene_val = list(class_1_df[i])
            class_2_gene_val = list(class_2_df[i])
            # Log fold change (LFC)
            fc = np.mean(class_1_gene_val) - np.mean(class_2_gene_val)
            # P-value from t-test
            p_val = ttest_ind(class_1_gene_val, class_2_gene_val)[1]
            lfc_val.append(fc)
            P_Value.append(p_val)
        dea_res = pd.DataFrame(dict(Genes=gene_list, LFC=lfc_val, P_Val=P_Value))
        # Adjusting p-values using Holm correction
        dea_res['Adj_P_Val'] = smm.multipletests(P_Value, method='holm')[1]
        dea_res.to_csv('DEA_for_class_' + str(class_1) + '_vs_' + str(class_2) + '.csv', index=False)
        return dea_res



    def quantile_normalizing(self,aug_data):#used
      target_var_name=aug_data.columns[-1]
      target = list(aug_data.iloc[:,-1])
      aug_data_t=aug_data.iloc[:,:-1].T
      aug_data_qn=qnorm.quantile_normalize(aug_data_t)
      aug_data_qn_t = pd.DataFrame(aug_data_qn).T
      aug_data_qn_t[target_var_name] = target
      return aug_data_qn_t


    def Multi_P_Val_Enrichment(self,data_path,p_val_vec=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]):#u
      print("multi_p_val_enrichment",data_path)
      for i in p_val_vec:
          try:
              #print("multi_p_val_enrichment",path)
              self.Enrichment(data_path,i)
              path = data_path + '/Enrichment_at_' + str(i)
              self.Depth_Pathways(path)
          except:
              pass
      return p_val_vec


    def Enrichment(self,data_path,p_val): #used#
      os.chdir(data_path)
      files=os.listdir()
      f=''
      for i in files:
          if (i.startswith('DEA')) and (i.endswith('.csv')):
              f=i
      df = pd.read_csv(f)
      result=self.enrichment_overreprsentation(df,p_val)
      os.mkdir('Enrichment_at_'+str(p_val))
      enr_dir=os.path.join(data_path,'Enrichment_at_'+str(p_val))
      os.chdir(enr_dir)
      result.to_csv('Enrichment_adj_P_val'+str(p_val)+'.csv')
      os.chdir(data_path)
      return



    def enrichment_overreprsentation(self,data,p_val):
      gp = GProfiler(return_dataframe=True)
      data_filt=data[data['Adj_P_Val']<p_val]
      genes=list(data_filt['Genes'])
      bg = list(data['Genes'])
      res=gp.profile(organism='hsapiens', query=genes, background=bg,no_evidences=False,
                  significance_threshold_method='fdr')
      return res


    def Depth_Pathways(self,file_path): #used#
      orig_path = os.getcwd()
      os.chdir(file_path)
      f=''
      for i in os.listdir():
          if i.startswith('Enrich'):
              f=i
      enr=pd.read_csv(f)
      enr_req=enr[enr['source'].isin(["GO:BP",'GO:CC','GO:MF'])]
      ids = list(enr_req['native'])
      Depth=[]
      for i in ids:
          Depth.append(self.find_ancestor_depth(i))
      enr_req['Depth']=Depth
      enr_req=enr_req[['source','name','p_value','term_size','intersection_size','intersections','Depth']]
      enr_req['Enriched_Ratio'] = enr_req['intersection_size']/enr_req['term_size']
      file_name = file_path.split('/')[-1]+'_Depth.csv'
      enr_req.to_csv(file_name)
      os.chdir(orig_path)
      return enr_req



    def find_ancestor_depth(self,go_term_id): #used#

    #Find the ancestors and depth of a given GO term.

      try:
          if go_term_id not in go:
              #print(f"GO term {go_term_id} not found in 'go' dictionary.")
              return None

          go_term = go[go_term_id]

          ancestors = go_term.get_all_parents()


          # Handle case when there are no ancestors (i.e., the GO term is at the top level)
          if not ancestors:
              #print(f"No ancestors found for GO term {go_term_id}. The GO term might be a top-level term.")
              depth = 0  # or set to 1 if you want to count the term itself as the first level
          else:
              # Calculate the depth of the GO term based on its ancestors
              depth = max(go[ancestor].depth for ancestor in ancestors)
              #print(f"Depth found for GO term {go_term_id}: {depth}")

          return depth

      except KeyError as e:
          #print(f"KeyError: The GO term or one of its ancestors is not found. Error details: {e}")
          return None
      except Exception as e:
          #print(f"An error occurred while processing GO term {go_term_id}: {e}")
          return None




    def top_genes_selected(self,dea_res,per_gene_select):

        p_val=dea_res['P_Val']
        dea_res['Adj_P_Val'] = smm.multipletests(dea_res['P_Val'], method='holm')[1]#stats.p_adjust(FloatVector(dea_res['P_Val']), method = 'holm',n=len(dea_res))
        n=int(per_gene_select*len(dea_res))
        genes=list(dea_res.sort_values(by='Adj_P_Val')['Genes'])[:n]
        all_genes=list(dea_res['Genes'])
        DEA=[]
        for i in all_genes:
            if i in genes:
                DEA.append('T')
            else:
                DEA.append('F')
        dea_res['DEA']=DEA
        print("top_genes_Selected")
        return genes,dea_res




    def enrichment_overreprsentation_top(self,data,class_1,class_2):#used#
        '''Enrichment analysis for the deg obtained
        Input : Result of DEA analysis
        Returns : A dataframe having list of pathways.
        P value adjustment medthod used : gSCS, the default one , as it takes into account
        that the different enrichment pathways are not independent of each other'''
        gp = GProfiler(return_dataframe=True)
        genes=list(data[data['DEA']=='T']['Genes'])
        bg = list(data['Genes'])
        res=gp.profile(organism='hsapiens', query=genes, background=bg,no_evidences=False,
                    significance_threshold_method='fdr')
        res.to_csv('Enrichment_Analysis_ORA_'+str(class_1)+'_vs_'+str(class_2)+'.csv')
        return res



    def volcano_plot(self,dea_result,class_1,class_2):#used
      '''Visualize the LFC v/s P values for all genes'''
      data=dea_result
      data['New_P_Val'] = -np.log10(data['P_Val'])
      plt.figure(figsize=(7,5))

      sns.set_style("darkgrid", {"axes.facecolor": ".9"})
      sns.scatterplot(x='LFC',y='New_P_Val',data=data,s=12)
      plt.xlabel('logFC',fontsize=15,font='DejaVu Serif')
      plt.ylabel('-log10(P value)',fontsize=15,font='DejaVu Serif')
      name='DEA_between_'+str(class_1)+' vs '+ str(class_2)+'.jpeg'
      plt.savefig(name)
      print('V_D')
      return



    def original_data_analysis(self,orig_file_name):#used#
      orig_path = os.getcwd()
      file_name = orig_file_name#input('Enter the name of the original file:')
      orig_data=pd.read_csv(file_name)
      if orig_data.columns[0]=='Unnamed: 0':
          orig_data=orig_data.iloc[:,1:]
      else:
          orig_data=orig_data
      freq=dict(Counter(orig_data.iloc[:,-1]))
      freq_df=pd.DataFrame(dict(classes = freq.keys(),Freq = freq.values()))
      data_id = 'Original_data'#input('Enter the name of folder with which you will save original data:')
      os.mkdir(data_id)
      os.chdir(os.path.join(os.getcwd(),data_id))
      freq_df.to_csv('Frequency_classes.csv')
      per_gene_sel = 0.2#float(input('Enter the amount of low variant genes to be filtered out:'))
      print('Unique classes present in data are: ',orig_data.iloc[:,-1].unique())
      con_vec=dict(Counter(orig_data['Condition']))
      class_1,class_2=list(dict(sorted(con_vec.items(),key=lambda x:x[1],reverse=True)[:2]).keys())
      top_per_genes = 0.01
      self.Original_Data_Performance(orig_data,per_gene_sel,class_1,class_2,top_per_genes)
      self.Multi_P_Val_Enrichment(os.getcwd())
      os.chdir(orig_path)
      print('Analysis Done_Original_data_analysis')
      return


    def Original_Data_Performance(self,orig_data,per_gene_sel,class_1,class_2,top_per_genes):#used
      orig_data=self.Zero_Removal_orig(orig_data)
      data=self.gene_selection_var_based(orig_data,per_gene_sel)
      cl1_cl2_dea = self.P_val_class_wise(data,class_1,class_2)
      top_genes,dea_res=self.top_genes_selected(cl1_cl2_dea,top_per_genes)
      enr = self.enrichment_overreprsentation_top(dea_res,class_1,class_2)
      self.volcano_plot(cl1_cl2_dea,class_1,class_2)
      orig_tsne=self.TSNE_original_classes(orig_data,top_genes,class_1,class_2)
      dea_res.to_csv('DEA for Classes '+str(class_1)+' vs '+str(class_2)+'.csv')
      return



    def Zero_Removal_orig(self,orig_data):#used#
      x=dict(orig_data.iloc[:,:-1].sum())
      gene_zero=[]
      for i in x:
          if x[i]==0:
              gene_zero.append(i)
      non_zero=set(list(orig_data.columns[:-1])).difference(set(gene_zero))
      req_cols=list(non_zero)+[orig_data.columns[-1]]
      orig_df =  orig_data[req_cols]
      return orig_df



    
    def TSNE_original_classes(self,orig_data,genes,class_1,class_2):#used
        '''Shows the distribution upon reduced dimension for respective classes
        orig_data : original data
        genes : Genes selected based upon adjusted P values
        '''
        tar_var=list(orig_data.iloc[:,-1])
        tar_name = list(orig_data.columns)[-1]
        original_data=orig_data[genes]
        print(original_data.shape)
        num_samples = len(original_data)

        # tsne_orig = TSNE(n_components=2, learning_rate='auto',
        #                init='random', perplexity=30).fit_transform(original_data)
        perplexity_value = min(30, num_samples - 1)  # Ensure perplexity is less than number of samples
        tsne_orig = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity_value).fit_transform(original_data)
        tsne_orig = pd.DataFrame(tsne_orig)
        tsne_orig.columns = ['TSNE_1','TSNE_2']
        tsne_orig.index = [*range(1,len(tsne_orig)+1)]
        tsne_orig[tar_name] = tar_var
        classes = [class_1,class_2]
        tsne_orig=tsne_orig[tsne_orig[tar_name].isin(classes)]
        tsne_orig.to_csv('TSNE_data'+str(class_1)+'_vs_'+str(class_2)+'.csv')
        plt.figure(figsize=(20,15))
        plt.subplot(221)
        sns.boxplot(x=tar_name,y='TSNE_1',data=tsne_orig).set(title='Distribution of Expression for TSNE_1')
        plt.tight_layout(pad=4)
        plt.subplot(222)
        file_name= 'Distribution of features in classes upon reduced dimension' +'.pdf'
        P=PdfPages(file_name)
        sns.boxplot(x=tar_name,y='TSNE_2',data=tsne_orig).set(title='Distribution of Expression for TSNE_2')
        plt.tight_layout(pad=4)
        plt.subplot(223)
        sns.scatterplot(x='TSNE_1',y='TSNE_2',hue=tar_name,data=tsne_orig).set(title='Distribution of features in classes upon reduced dimension')
        P.savefig()
        P.close()
        return


    def one_Hot_Matrix_Pathways(self,path):#used
      df=self.Depth_Files(path)
      path_depth=dict(zip(df['name'],df['Depth']))
      x=df[['name','Times']]
      one_hot_matrix = pd.get_dummies(x, columns=['Times'])
      one_hot_matrix = one_hot_matrix.groupby('name').sum()
      cols=list(one_hot_matrix.columns)
      cols=['_'.join(i.split('_')[1:]) for i in cols]
      one_hot_matrix.columns=cols
      col=self.custom_sort(list(cols))
      one_hot_matrix=one_hot_matrix[col]
      cols=list(one_hot_matrix.columns)
      paths=list(one_hot_matrix.index)
      Depth=[]
      for i in paths:
          if i in path_depth:
              Depth.append(path_depth[i])
      one_hot_matrix['Depth']=Depth
      os.chdir(path)
      os.mkdir('Enrichment_Depth_Analysis')
      os.chdir(os.path.join(path,'Enrichment_Depth_Analysis'))
      one_hot_matrix.to_csv('Enriched_depth.csv')
      os.chdir(path)
      return one_hot_matrix




    def custom_sort(self,strings):#used
      def sort_key(s):
          match = re.match(r'(\d+)_times', s)
          if match:
              return int(match.group(1))
          elif s.isalnum():
              return ord(s[0])
          else:
              return float('inf')

      return sorted(strings, key=sort_key)




    def Depth_Files(self,path):#used
      enr_dir=self.Enrichment_Depth_Files(path)
      Data=pd.DataFrame()
      for i in enr_dir:
          os.chdir(i)
          files=os.listdir()
          for j in files:
              if j.endswith('Depth.csv'):
                  file_name=j
                  enr=pd.read_csv(file_name,index_col=0)
                  enr['Times']='_'.join(i.split('/')[-2].split('_')[-2:])
                  enr['P_Cutoff']=i.split('_')[-1]
                  Data=pd.concat([Data,enr])
      return Data



    def Enrichment_Depth_Files(self,path):
      main_directories = self.list_directories(path)
      enr_data=pd.DataFrame()
      path_dir = [os.path.join(path,i) for i in main_directories]
      paths=[]
      for i in path_dir:
          os.chdir(i)
          p_dir = list_directories(i)
          p_path=[os.path.join(i,j) for j in p_dir]
          for enr_path in p_path:
              paths.append(enr_path)
      return paths

    def list_directories(self,folder_path): #used
      # List all entries in the given folder
      all_entries = os.listdir(folder_path)
      # Filter out the entries that are directories
      directories = [entry for entry in all_entries if os.path.isdir(os.path.join(folder_path, entry))]
      return directories


    def final_enrichment_file(self,path):#used
      folders = self.list_directories(path)
      Enrich = pd.DataFrame()
      folder_path = [os.path.join(path, i) for i in folders]
      All_files = []

      for folder in folder_path:
          enrich_folders = self.list_directories(folder)
          enrich_path = [os.path.join(folder, j) for j in enrich_folders]

          for enrich_folder in enrich_path:
              All_files.append(enrich_folder)
              os.chdir(enrich_folder)
              files = os.listdir()

              for file_name in files:
                  if file_name.startswith('Enrichment_adj'):

                      enr = pd.read_csv(file_name)
                      enr = enr[['source', 'name', 'p_value', 'term_size', 'intersection_size', 'intersections']]
                      enr['Enriched_Ratio'] = enr['intersection_size'] / enr['term_size']
                      enr['Times']='_'.join(enrich_folder.split('/')[-2].split('_')[-2:])
                      enr['P_Val'] = (enrich_folder.split('/')[-1].split('_')[-1])
                      Enrich = pd.concat([Enrich, enr], ignore_index=True)

      x=Enrich[['name','Times']]
      one_hot_matrix = pd.get_dummies(x, columns=['Times'])
      one_hot_matrix = one_hot_matrix.groupby('name').sum()
      cols=list(one_hot_matrix.columns)
      cols=['_'.join(i.split('_')[1:]) for i in cols]
      one_hot_matrix.columns=cols
      col=self.custom_sort(list(cols))
      one_hot_matrix=one_hot_matrix[col]
      cols=list(one_hot_matrix.columns)
      paths=list(one_hot_matrix.index)
      d=dict(zip(Enrich['name'],Enrich['source']))
      Database=[]
      for i in one_hot_matrix.index:
          if i in d.keys():
              Database.append(d[i])
      one_hot_matrix['Database']=Database
      os.chdir(path)
      os.mkdir('All_Pathways_Enrichment')
      os.chdir(os.path.join(path,'All_Pathways_Enrichment'))
      Enrich.to_csv('All_Pathways_Enriched_data.csv')
      one_hot_matrix.to_csv('One_hot_matrix_pathways.csv')
      os.chdir(path)
      return one_hot_matrix


    def all_analysis(self,path):#used
      self.Boxplot_Times(path)
      self.Plot_Countplot(path)
      self.Enriched_sig_plot(path)
      self.Volcano_all_analysis(path)
      return



    def Boxplot_Times(self,data_path):#used
        os.chdir(data_path)
        pattern = re.compile(r'Enrichment*')
        augmented_elements = [item for item in os.listdir() if pattern.match(item)]
        file_path=[os.path.join(data_path,i) for i in augmented_elements]
        os.chdir(file_path[0])
        df = pd.read_csv('Enriched_depth.csv')
        df=df.drop('name',axis=1)

        # Reorder the 'Times' category to have 'Original_data' first if it exists
        df_melted = df.melt(id_vars=['Depth'], var_name='Times', value_name='Presence')

        # Filter out rows where 'Presence' is 0
        df_melted = df_melted[df_melted['Presence'] != 0]

        # Ensure 'Times' is a categorical variable and set the order
        times_order = ['Original_data'] if 'Original_data' in df.columns else []

        times_order += [col for col in df.columns if col != 'Original_data' and col != 'Depth']

        # Ensure 'Times' is a categorical variable and set the order
        df_melted['Times'] = pd.Categorical(df_melted['Times'], categories=times_order, ordered=True)

        # Create the box plot
        plt.figure(figsize=(12, 8))

        sns.violinplot(x='Times', y='Depth', data=df_melted, inner=None, palette="muted")  # Create the violin plot without inner annotations

        # Calculate medians for each category
        medians = df_melted.groupby('Times')['Depth'].median().reindex(df_melted['Times'].cat.categories)

        # Add jittered data points
        sns.stripplot(x='Times', y='Depth', data=df_melted, jitter=True, color='black', alpha=0.4)

        # Highlight medians with red color
        for i, median in enumerate(medians):
            print("In the for loop")
            plt.scatter(i, median, color='red', zorder=3)

        plt.xlabel('Times')

        plt.ylabel('Depth')

        plt.title('Depth information for Pathways with increasing N')

        plt.xticks(rotation=45)

        plt.show()


        return


    def Plot_Countplot(self,data_path):#used
      os.chdir(data_path)
      pattern = re.compile(r'All_Pathways*')
      augmented_elements = [item for item in os.listdir() if pattern.match(item)]
      file_path=[os.path.join(data_path,i) for i in augmented_elements]
      os.chdir(file_path[0])
      #os.chdir(data_path)
      df=pd.read_csv('All_Pathways_Enriched_data.csv')
      data=df[['source','Times','P_Val']]
      df_count = data.groupby(['Times', 'P_Val', 'source']).size().reset_index(name='count')
      df_count['P_Val']=df_count['P_Val'].apply(lambda x: float(x))
      df_count=df_count.sort_values(by='P_Val',ascending=False)
      with PdfPages('Countplot_Databases.pdf') as pdf:
          # Get the unique databases from the 'source' column
          unique_databases = list(df['source'].unique())

          # Loop through each unique database
          for database in unique_databases:
              # Filter the dataframe for the current database
              db=df_count[df_count['source']==database]
              times_order = self.arrange_columns(db['Times'].unique())
              db['Times'] = pd.Categorical(db['Times'], categories=times_order, ordered=True)
              g = sns.catplot(
                  data=db,
                  kind='bar',
                  x='source', y='count', hue='P_Val',
                  col='Times',
                  col_wrap=4,  # 4 plots per row
                  height=5, aspect=1
              )
              plt.suptitle(f'Frequency of P_Val for {database}')
              plt.ylabel('Frequency')
              pdf.savefig()
              plt.show()
      return



    def arrange_columns(self,columns):#used
      """
      Sort and arrange column names in the specified order:
      'Original_data', '3_times', '6_times', ..., '30_times'.
      """
      # Define the desired order
      desired_order = ['Original_data'] + [f'{i}_times' for i in range(3, 31, 3)]

      # Filter the available columns
      available_columns = [col for col in desired_order if col in columns]

      return available_columns


    def Enriched_sig_plot(self,data_path):#used
      os.chdir(data_path)
      pattern = re.compile(r'All_Pathways*')
      augmented_elements = [item for item in os.listdir() if pattern.match(item)]
      file_path=[os.path.join(data_path,i) for i in augmented_elements]
      os.chdir(file_path[0])
      df=pd.read_csv('All_Pathways_Enriched_data.csv')
      df=self.All_Data_filtered(df)
      df_res=df[df['Enriched_Ratio']>0.8]
      times_order = self.arrange_columns(df_res['Times'].unique())
      df_res['Times'] = pd.Categorical(df_res['Times'], categories=times_order, ordered=True)
      plt.figure(figsize=(12,8))
      sns.countplot(df_res['Times'])
      plt.xticks(rotation=90)
      plt.ylabel('Frequency')
      plt.xlabel('Times')
      plt.savefig('Frequency_enriched_ratio.pdf',dpi=650)
      return


    def All_Data_filtered(self,data):#used
      db=list(data['source'].unique())
      data_bp=pd.DataFrame()
      for i in db:
          x=self.patways_required(data,i).sort_values(by='Enriched_Ratio',ascending=False)
          data_bp=pd.concat([data_bp,x])
      return data_bp



    def patways_required(self,data,database):#used
      pathways=self.database_filter(data,database)
      x=data[data['source']==database].sort_values(by='p_value')
      ind=[]
      for i in pathways:
          indices=self.pathways_least_p_val(x,i)
          ind.append(indices)
      data=x.loc[ind,:]
      data=data.sort_values(by='p_value')
      return data

    def database_filter(self,data,database): #used
      x=data[data['source']==database].sort_values(by='p_value')
      pathways=list(x['name'].unique())
      return pathways

    def pathways_least_p_val(self,data,pathway):#used#
      a=data[data['name']==pathway]
      ind=a[a['p_value']==a['p_value'].min()].index[0]
      return ind



    def Volcano_all_analysis(self,path):#used
      path_file=path
      files_path=self.required_directories(path)
      p_max,lfc_min,lfc_max=self.vol_par(path)
      os.chdir(path_file)
      with PdfPages('Volcano_Plot_Analysis_scaled.pdf') as pdf:
          fig = plt.figure(figsize=(20, 20))
          for i, d in enumerate(files_path, start=1):
              plt.subplot(4, 3, i)
              p=self.Volcano_Graph(d,p_max,lfc_min,lfc_max)
              plt.tight_layout(pad=2)
          pdf.savefig(fig)

      plt.close(fig)
      return



    def vol_par(self,path):#used#
        files_path = self.required_directories(path)
        p_val,LFC_MIN,LFC_MAX=[],[],[]
        for file_path in files_path:
            os.chdir(file_path)
            file=[file for file in os.listdir() if (file.startswith('DEA') & file.endswith('csv'))][0]
            p,lmin,lmax=self.param_vol(file)
            p_val.append(p)
            LFC_MIN.append(lmin)
            LFC_MAX.append(lmax)
        p=pd.Series(p_val)
        p.replace([np.inf, -np.inf], np.nan, inplace=True)
        p.dropna(inplace=True)
        p_max_val = np.max(p)
        lfc_min_val=np.min(LFC_MIN)
        lfc_max_val=np.max(LFC_MAX)
        return p_max_val,lfc_min_val,lfc_max_val


    def param_vol(self,data_path):#used#
      data=pd.read_csv(data_path)
      data['-log10(p_value)'] = -np.log10(data['P_Val'])
      p_val = np.max(data['-log10(p_value)'])
      lfc = list(data['LFC'])
      lfc_min_val=np.min(lfc)
      lfc_max_val =np.max(lfc)
      return p_val,lfc_min_val,lfc_max_val




    def required_directories(self,path): #used
      os.chdir(path)
      pattern = re.compile(r'(Aug|Orig)', re.IGNORECASE)
      terms=os.listdir()
      filtered_terms = [term for term in terms if pattern.search(term)]
      filtered_terms=self.sort_terms(filtered_terms)
      path_files=[os.path.join(path,i) for i in filtered_terms]
      return path_files



    def sort_terms(self,terms): #used
      # Separate "Original_data" from the rest
      original_data = [term for term in terms if 'Original_data' in term]
      augmented_data = [term for term in terms if 'Augmented_Data' in term]

      # Regex pattern to extract numerical value
      pattern = re.compile(r'_(\d+)_times')

      # Sort augmented data based on the extracted numerical value
      sorted_augmented_data = sorted(augmented_data, key=lambda x: int(pattern.search(x).group(1)))

      # Combine original data and sorted augmented data
      sorted_terms = original_data + sorted_augmented_data

      return sorted_terms



    def Volcano_Graph(self,path,p_max,lfc_min,lfc_max): #used
      os.chdir(path)
      file=''
      for i in os.listdir():
          if (i.startswith('DEA') & i.endswith('.csv')):
              file=i
      data=pd.read_csv(file)

      data['-log10(p_value)'] = -np.log10(data['P_Val'])

      # Plotting
      #plt.figure(figsize=(5, 5))
      sns.set(style="whitegrid")
      title='_'.join(path.split('/')[-1].split('_')[-2:])
      # Plot points
      plt.scatter(list(data['LFC']),y=data['-log10(p_value)'])
      #sns.scatterplot(x='LFC', y='-log10(p_value)', data=data, color='blue').set_title(title)
      plt.title(title)
      plt.xlim([lfc_min,lfc_max])
      plt.ylim([0, p_max])
      return


    def zipped_Result(self,gse_id,path):#used
      os.chdir(path)
      folder_name='Results_of_'+gse_id
      os.mkdir(folder_name)
      target_path=os.path.join(path,folder_name)
      files=self.list_directories(path)
      files = [os.path.join(path,i) for i in files]
      for i in files:
          self.move_folder(i, target_path)
      # Path to the folder to be zipped
      folder_to_zip = target_path
      # Path to the output zip file
      zip_file_path = 'Results_of_your_analyis_' + gse_id + '.zip'

      # Call the function to zip the folder
      self.zip_folder(folder_to_zip, zip_file_path)
      return


    def move_folder(self,source, destination):
      try:
          # Move the folder from source to destination
          shutil.move(source, destination)
          print(f"Folder '{source}' moved to '{destination}' successfully.")
      except Exception as e:
          print(f"Error: {e}")



    def zip_folder(self,folder_path, zip_file_path):
      with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
          for root, dirs, files in os.walk(folder_path):
              for file in files:
                  file_path = os.path.join(root, file)
                  zipf.write(file_path, os.path.relpath(file_path, folder_path))


