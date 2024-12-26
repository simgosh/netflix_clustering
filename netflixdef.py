# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import mstats
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

# Load and preprocess datasets
def load_and_preprocess_data():
    # Load the datasets
    df = pd.read_csv("netflix_clustering/data/credits.csv")
    df1 = pd.read_csv("netflix_clustering/data/titles.csv")

    # Remove duplicates
    df = df.drop_duplicates()
    df1 = df1.drop_duplicates()

    # Merge the datasets
    netflix = df.merge(df1, how="inner", on=["id"])
    netflix = netflix.drop_duplicates(subset=['type', 'title', 'release_year'])

    # Clean 'type' column
    netflix['type'] = netflix['type'].str.strip().str.lower()

    # Remove unwanted columns
    netflix.drop(columns=['person_id', 'character', 'role', 'name', 'description'], axis=1, inplace=True)    # Fill missing values
    netflix.fillna({
        'imdb_votes': 0,
        'imdb_score': 0,
        'tmdb_popularity': 0,
        'tmdb_score': 0,
        'age_certification': 'Unknown',
        'title': 'Unknown',
        'main_genres': 'Unknown',
        'seasons': 0
    }, inplace=True)
    return netflix

def visualize_type(netflix):
    plt.figure(figsize=(10,8))
    plt.title("Movies and Tv Series", fontsize=16)
    plt.pie(netflix.type.value_counts(),
        labels=["Movie", "Show"],
        textprops={"fontsize":16},
        radius=1.1, startangle=90,
        shadow=True,
        colors=["saddlebrown", "olivedrab"],
        autopct='%.0f%%')
    plt.show()

def process_genres(netflix):
    print(netflix.genres.value_counts())
    netflix['genres'] = netflix["genres"].str.replace(r"[", '').\
    str.replace(r"'", '').str.replace(r"]", '')
    netflix['main_genres']= netflix["genres"].str.split(',').str[0]
    print(netflix[['genres', 'main_genres']])
    netflix.drop(columns=["genres", "id"], inplace=True)
    netflix.main_genres = netflix.main_genres.replace('', np.nan)
    print(netflix.main_genres.value_counts())

def count_genres(netflix):
    plt.figure(figsize=(12,8))
    plt.title("\nCount of Genres\n", fontsize=15)
    genre = pd.DataFrame(netflix.main_genres.value_counts())
    order = genre.index
    sns.countplot(y=netflix.main_genres, palette='PuRd_r', order=order)
    plt.xlabel('Movie Count')
    plt.ylabel('Genre')
    plt.show()  

def process_and_visualize_countries(netflix):
    # Clean 'production_countries' column and extract the first country
    print("Initial Production Countries Value Counts:")
    print(netflix['production_countries'].value_counts())
    
    # Clean up unwanted characters like brackets and quotes
    netflix['production_countries'] = netflix['production_countries'].str.replace(r"[',\[\]]", '', regex=True)
    
    # Extract the first country from the cleaned 'production_countries' column
    netflix['country'] = netflix['production_countries'].str.split(',| ').str[0].str.strip()
    
        # Drop the 'production_countries' column after extraction
    netflix.drop(columns=["production_countries"], inplace=True)
    
    # Print the updated 'country' column value counts
    print("\nProcessed Country Value Counts:")
    print(netflix['country'].value_counts())
    
    # Check for missing values in the 'country' column
    missing_values = netflix['country'].isna().sum()
    print(f"\nMissing values in 'country': {missing_values}")  # Ensure no missing values
    
    # Visualize the distribution of the top 15 countries by movie count
    country_counts = netflix['country'].value_counts().head(15)
    country_order = country_counts.index
    
    # Create the count plot
    plt.figure(figsize=(12, 8))
    plt.title("\nTop 15 Countries by Amount of Movies\n", fontsize=18)
    sns.countplot(y=netflix['country'], palette="coolwarm", order=country_order)
    plt.xlabel("Movie Count")
    plt.ylabel("Country")
    plt.show()

# Visualize age certification distribution
def visualize_age_certification(netflix):
    plt.figure(figsize=(12, 8))
    plt.title("Count of Certification for each Age Group")
    certificate = pd.DataFrame(netflix.age_certification.value_counts())
    order = certificate.index
    sns.countplot(y=netflix.age_certification, palette="coolwarm", order=order)
    plt.yticks(rotation=45)
    plt.xlabel("Count")
    plt.ylabel("Certificate")
    plt.show()

def process_age_certification(netflix):
    # Replace age certification codes with more descriptive labels
    age_certification_mapping = {
        'G': 'General Audience',
        'PG': 'Parental Guidance',
        'PG-13': 'Parents Strongly Cautioned',
        'R': 'Restricted',
        'NC-17': 'Adults Only',
        'TV-G': 'General Audience',
        'TV-Y': 'All Children',
        'TV-Y7': 'Directed to Older Children',
        'TV-PG': 'Parental Guidance Suggested',
        'TV-14': 'Parents Strongly Cautioned',
        'TV-MA': 'Mature Audience'
    }
    
    netflix["age_certification"] = netflix["age_certification"].replace(age_certification_mapping)
    # Handle missing values (replace empty strings with NaN)
    netflix['age_certification'] = netflix['age_certification'].replace('', np.nan)   
    # Print the value counts of age certifications
    print(netflix['age_certification'].value_counts())


def check_data(netflix):
    # Show basic descriptive statistics (e.g., count, mean, std, min, etc.)
    print(netflix.describe().T)  
    # Display general DataFrame info (e.g., datatypes, non-null counts)
    print(netflix.info())    
    # Print column names
    print(netflix.columns)   
    # Display counts of unique values in 'main_genres' column
    print(netflix["main_genres"].value_counts())   
    # Display counts of unique values in 'country' column
    print(netflix["country"].value_counts())
    netflix["main_genres"]=netflix["main_genres"].fillna("Unknown")
    # Drop 'imdb_id' column from the DataFrame
    netflix.drop(columns=['imdb_id'], inplace=True)
    print(netflix.isnull().sum())
    # Return the modified DataFrame
    return netflix

# Winsorize the data for 'tmdb_popularity'
def winsorize_tmdb_popularity(netflix):
    netflix['winsorized_tmdb_popularity'] = mstats.winsorize(netflix['tmdb_popularity'], limits=[0.05, 0.05])
    plt.figure(figsize=(10, 5))
    sns.histplot(netflix['tmdb_popularity'], kde=True)
    plt.title("Original tmdb_popularity (Before Winsorization)")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(netflix['winsorized_tmdb_popularity'], kde=True)
    plt.title("Winsorized tmdb_popularity (After Winsorization)")
    plt.show()

# Apply log transformation for IMDB votes
def log_transform_imdb_votes(netflix):
    netflix['log_imdb_votes'] = np.log1p(netflix['imdb_votes'])  # log1p handles zero values
    plt.figure(figsize=(10, 5))
    sns.histplot(netflix['log_imdb_votes'], kde=True)
    plt.title("Log Transformed IMDB_VOTES (After Transformation)")
    plt.show()

# Visualize IMDB score vs tmdb popularity
def visualize_imdb_vs_tmdb(netflix):
    sns.scatterplot(data=netflix,
                    x="log_imdb_votes",
                    y="winsorized_tmdb_popularity",
                    color="purple",
                    hue="main_genres")
    plt.title("Log-transformed IMDb Votes vs Winsorized TMDB Popularity")
    plt.xlabel("Log-transformed IMDb Votes")
    plt.ylabel("Winsorized TMDB Popularity")
    plt.show()

# Correlation matrix
def visualize_correlation_matrix(netflix):
    corr = netflix.select_dtypes(include=["Int64", "float64"])
    corr = corr.dropna()
    corr_matrix = corr.corr()
    sns.heatmap(data=corr_matrix, annot=True, fmt=".2g", cmap="coolwarm")
    plt.show()

# Sort by IMDB votes and get top 15 movies
def top_movies_by_imdb(netflix):
    netflix.reset_index(inplace=True)
    movies = netflix[netflix['type'] == 'movie']
    top_15_imdb_movie = movies[['title', 'log_imdb_votes', 'type', 'age_certification']].\
        sort_values(by='log_imdb_votes', ascending=False).head(15)
    return top_15_imdb_movie

# Visualize top 15 movies by IMDB votes
def visualize_top_15_movies_by_imdb(top_15_imdb_movie):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='log_imdb_votes', y='title', hue='age_certification',
                data=top_15_imdb_movie, palette='viridis')
    plt.title('Top 15 Movies by IMDB Votes')
    plt.xlabel('IMDB Votes')
    plt.ylabel('Title')
    plt.show()

def show_series(netflix):
    show = netflix[netflix['type'] == 'show']
    top_15_imdb_show = show[['title', 'log_imdb_votes', 'type']].\
    sort_values(by='log_imdb_votes', ascending=False).head(15)
# Check the top 15
    print(top_15_imdb_show)

# Visualize top 15 movies by IMDB votes
def visualize_top_15_movies_by_imdb(top_15_imdb_show):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='log_imdb_votes', y='title', hue='age_certification',
                data=top_15_imdb_show, palette='viridis')
    plt.title('Top 15 Movies by IMDB Votes')
    plt.xlabel('IMDB Votes')
    plt.ylabel('Title')
    plt.show()

# Find shows with the longest seasons
def long_shows(netflix):
    netflix["seasons"] = pd.to_numeric(netflix["seasons"], errors='coerce')
    long_shows = netflix[netflix["seasons"].between(9.0, 42.0)].sort_values(by="seasons", ascending=False)
    return long_shows

# Visualize long shows
def visualize_long_shows(long_shows):
    long_shows_25 = long_shows.head(25)
    plt.figure(figsize=(14, 8))
    sns.barplot(data=long_shows_25,
                x="seasons",
                y="title",
                hue="country")
    plt.title("Top 25 Film/Show by Seasons with Long Seasons")
    plt.xlabel("Seasons")
    plt.ylabel("Title")
    plt.show()    

def comedy_or_drama(netflix):
    romance_drama = netflix[(netflix["main_genres"] == "drama") | (netflix["main_genres"] == "romance")]
    rom_dram = romance_drama[['title', 'imdb_score', 'type', 'age_certification', 'country', 'seasons']].\
        sort_values(by='imdb_score', ascending=False).head(15)
    return rom_dram

def visualize_rom_drama(comedy_or_drama_data):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='title', y='imdb_score', hue='type', data=comedy_or_drama_data, palette='rocket_r')
    plt.title('Top Drama and Romance Show&Movie by IMDB Score')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.xlabel('Title')
    plt.ylabel('IMDB Score')
    plt.show()

def age_cert_children(netflix):
    children_shows = netflix[netflix['age_certification'] == 'All Children']
    children = children_shows[['title', 'imdb_score', 'type', 'age_certification', 'country', 'seasons']].\
    sort_values(by='imdb_score', ascending=False).head(15)
    print(children)

def visualize_for_all_children(age_cerf_children):
    #visualize for childrens shows
    plt.figure(figsize=(12, 6))
    sns.barplot(x='title', y='imdb_score', hue='country',
             data=age_cerf_children, palette='rocket_r')
    plt.title('Top Drama and Romance Show&Movie for Children by IMDB Score')
    plt.xticks(rotation=45)
    plt.xlabel('Title')
    plt.ylabel('IMDB Score')
    plt.show()

def turkiye_shows(netflix):
    turkiye_shows1 = netflix[netflix['country'] == 'TR']
    tr_show = turkiye_shows1[['title', 'age_certification', 'main_genres',
                   'release_year', 'imdb_score']].sort_values(by="imdb_score", ascending=False).head(20)
    print(tr_show)
    return tr_show

def turkiye_visualize(turkiye_shows):
    #visualize top 20 shows for turkey by imdb score
    plt.figure(figsize=(12, 6))
    sns.barplot(x='title', y='imdb_score', hue='main_genres',
             data=turkiye_shows, palette='rocket_r')
    plt.title('Top 20 Show&Movie for Turkey by IMDB Score')
    plt.xticks(rotation=45)
    plt.xlabel('Title')
    plt.ylabel('IMDB Score')
    plt.show()

def least_turkiye_shows(netflix):
    #see turkiye shows details
    turkiye_shows2 = netflix[netflix['country'] == 'TR']
    tr_show_least = turkiye_shows2[['title', 'age_certification', 'main_genres',
                   'release_year', 'imdb_score']].sort_values(by="imdb_score", ascending=False).tail(20)
    print(tr_show_least)
    return tr_show_least

def least_turkiye(least_turkiye_shows):
    #visualize least 20 shows for turkey by imdb score
    plt.figure(figsize=(12, 6))
    sns.barplot(x='title', y='imdb_score', hue='main_genres',
             data=least_turkiye_shows, palette='rocket_r')
    plt.title('Least 20 Show&Movie for Turkey by IMDB Score')
    plt.xticks(rotation=45)
    plt.xlabel('Title')
    plt.ylabel('IMDB Score')
    plt.show()

def mean_imdb_all_genres(netflix):
    # Calculate the mean IMDb score for each genre
    mean_imdb_by_genre = netflix.groupby("main_genres")["imdb_score"].mean()
    mean_imdb_by_genre_sort = mean_imdb_by_genre.sort_values(ascending=False)
    print(mean_imdb_by_genre_sort)
    return mean_imdb_by_genre_sort

def visualize_mean_imdb_all_genre(mean_imdb_by_genre_sort):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=mean_imdb_by_genre_sort.index, y=mean_imdb_by_genre_sort.values,
             palette="viridis")
    plt.title("Mean IMDb Score by Genre", fontsize=16)
    plt.xlabel("Genre", fontsize=12)
    plt.ylabel("Mean IMDb Score", fontsize=12)
    plt.xticks(rotation=45)  # Rotate genre names if they are long
    plt.tight_layout()
    plt.show()

def usa_detail(netflix):
    # Filter data for USA-produced shows
    usa = netflix[netflix['country'] == 'US']
    # Calculate the mean IMDb score for each genre in the USA
    mean_imdb_usa = usa.groupby("main_genres")["imdb_score"].mean().sort_values(ascending=False)
    print(mean_imdb_usa)
    return mean_imdb_usa

def usa_visualize(mean_imdb_usa):
    # Visualize the mean IMDb score by genre in the USA
    plt.figure(figsize=(12, 6))
    sns.barplot(x=mean_imdb_usa.index, y=mean_imdb_usa.values, palette="coolwarm")
    plt.title("Mean IMDb Score by Genre in USA", fontsize=16)
    plt.xlabel("Genre", fontsize=12)
    plt.ylabel("Mean IMDb Score", fontsize=12)
    plt.xticks(rotation=45)  # Rotate genre names for better readability
    plt.tight_layout()  # Ensure the plot fits well
    plt.show()

def avg_runtime(netflix):
    #average runtime in min
    avg_runtime_by_films = netflix.groupby(["main_genres", "title"])["runtime"].mean().sort_values(ascending=False).nlargest(15)
    print(avg_runtime_by_films)    
    
def hist_runtime(netflix):
    plt.figure(figsize=(12,6))
    sns.histplot(netflix["runtime"], bins=30,
             kde=True, color="skyblue")
    plt.title("Distribution of Movie/Show Runtimes", fontsize=18)
    plt.xlabel('Runtime (minutes)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()


def hist_tmdb_score(netflix):
    plt.figure(figsize=(12,6))
    sns.histplot(netflix["tmdb_score"], bins=30,
             kde=True, color="skyblue")
    plt.title("Distribution of Movie/Show TMDB Score", fontsize=18)
    plt.xlabel('TMDB Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

def hist_release_year(netflix):
    plt.figure(figsize=(12,6))
    sns.histplot(netflix["release_year"], bins=30,
             kde=True, color="skyblue")
    plt.title("Distribution of Movie/Show Release Year", fontsize=18)
    plt.xlabel('Release Year', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

def comedy_genre(netflix):
#movie genre details
    comedy = netflix[netflix['main_genres'] == 'comedy']
    comedy_detail = comedy[["title", "country", "age_certification",
                        "imdb_score", "type"]].sort_values(by="imdb_score", ascending=False)
    comedy_top=comedy_detail.head(20)
    print(comedy_top)
    return comedy_top

def visualize_comedy(comedy_top):
    plt.figure(figsize=(12,8))
    sns.barplot(x="title", y="imdb_score", data=comedy_top,
            palette="viridis")
    plt.title("Top 20 Comedy Films by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.xticks(rotation=90)
    plt.show()

def romance_genre(netflix):
    romance = netflix[netflix['main_genres'] == 'romance']
    romance_detail = romance[["title", "country", "age_certification",
                        "imdb_score", "type"]].sort_values(by="imdb_score", ascending=False)
    romance_top=romance_detail.head(20)
    print(romance_top)
    return romance_top

def visualize_romance(romance_top):
    #visualize
    plt.figure(figsize=(12,8))
    sns.barplot(x="title", y="imdb_score", data=romance_top,
            palette="viridis")
    plt.title("Top 20 Romance Films by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.xticks(rotation=90)
    plt.show()


#drama and action genres details
def drama_and_action(netflix):
    drama_action = netflix[(netflix['main_genres'] == 'drama') | (netflix['main_genres'] == 'action')]
    details = drama_action[["title", "country", "age_certification",
                        "imdb_score", "type", "main_genres"]].sort_values(by="imdb_score", ascending=False)
    drama_action_top=details.head(20)
    print(drama_action_top)
    return drama_action_top

def visaulize_drama_action(drama_action_top):
    plt.figure(figsize=(15,7))
    sns.barplot(data=drama_action_top,
            x="title",
            y="imdb_score",
            hue="country")
    plt.xticks(rotation=45)
    plt.xlabel("Title")
    plt.ylabel("IMDB Score")
    plt.title("Top 20 Drama or Action Film/Show by IMDB Score", fontsize=18)
    plt.show()

#documentation or crime genres
def documentation_crime(netflix):
    documentation_crime = netflix[(netflix['main_genres'] == 'crime') | (netflix['main_genres'] == 'documentation')]
    details_dc = documentation_crime[["title", "country", "age_certification",
                        "imdb_score", "type", "main_genres"]].sort_values(by="imdb_score", ascending=False)
    dc_top=details_dc.head(20)
    print(dc_top) 
    return dc_top

def visualize_doc_crime(dc_top):
    plt.figure(figsize=(12,6))
    sns.barplot(data=dc_top,
            x="title",
            y="imdb_score",
            hue="country")
    plt.xticks(rotation=90)
    plt.title("Documentation or Crime Film/Show Top 20 by IMDB Score")
    plt.xlabel("Title")
    plt.ylabel("IMDB Score")
    plt.show()

#details for korean produced 
def korean(netflix):
    #details for south korea
    korea = netflix[netflix['country'] == 'KR']
    korea_detail = korea[['title', "release_year", 'main_genres', "imdb_score",
                      "type", "age_certification", "tmdb_score"]].sort_values(by="imdb_score", ascending=False)
    korea_top_20 = korea_detail.head(20)
    print(korea_top_20)
    return korea_top_20

def korean_visualize(korea_top_20):
    plt.figure(figsize=(12,6))
    sns.barplot(data=korea_top_20,
            x="title",
            y="imdb_score",
            hue="main_genres")
    plt.xticks(rotation=90)
    plt.title("Top 20 Film/Shows in South Korea", fontsize=18)
    plt.xlabel("Title")
    plt.ylabel("IMDB Score")
    plt.show()

def romance_korea(netflix):
    romance_korea = netflix[(netflix['main_genres'] == 'romance') & (netflix['country'] == 'KR')]
    rom_kor = romance_korea[["title", "release_year", "type", 
                         "imdb_score", "tmdb_score", "main_genres"]].sort_values(by="imdb_score", ascending=False)
    print(rom_kor.head()) #just 2 record movie or show about romance category produced by Korea!!
    return rom_kor

def visualize_romance_korea(rom_kor):
    plt.pie(rom_kor.type.value_counts(),
        labels=["Rookie Historian Goo Hae-Ryung", " Tune in for Love"],
        textprops={"fontsize":16},
        radius=1.1, startangle=90,
        shadow=True,
        colors=["saddlebrown", "olivedrab"],
        autopct='%.0f%%')
    plt.title("Romance Films in South Korea")
    plt.show()

def long_season(netflix):
    long_season = netflix[netflix["seasons"]==42.0]
    print(long_season)

def between_seasons(netflix):
    netflix["seasons"] = pd.to_numeric(netflix["seasons"], errors='coerce')
    long_shows = netflix[netflix["seasons"].between(9.0, 42.0)].sort_values(by="seasons",ascending=False)
    print(long_shows)
    long_shows_25 = long_shows.head(25)
    return long_shows_25

def visualize_between_seasons(long_shows_25):
    plt.figure(figsize=(14,8))
    sns.barplot(data=long_shows_25,
            x="seasons",
            y="title",
            hue="country")
    plt.title("Top 25 Film/Show by Seaons which has long season")
    plt.xlabel("Seasons")
    plt.ylabel("Title")
    plt.show()

#8 seaons which has films or shows
def eight_season(netflix):
    eight_seasons = netflix[netflix["seasons"] == 8.0]
    seasons_eight = eight_seasons[["title", "age_certification", "country",
                               "release_year", "type", "imdb_score"]].sort_values(by="imdb_score",
                                                                                  ascending=False)
    print(seasons_eight)
    return seasons_eight

def visualize_eight_season(seasons_eight):
    plt.figure(figsize=(14,8))
    sns.barplot(data=seasons_eight,
            x="imdb_score",
            y="title",
            hue="country")
    plt.title("The Film/Show when have 8 seasons by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.ylabel("Title")
    plt.show() #gilmore girls that has the most rating by imdb score


def release_year_detail(netflix):
    release = netflix[netflix["release_year"].between(2010, 2024)].sort_values(by="release_year",ascending=False)
    release_shows = release[["title", "type", "imdb_score","country",
                              "main_genres", "release_year"]].sort_values(by="imdb_score",
                                                                                        ascending=False)
    release_shows_top = release_shows.head(25)
    print(release_shows_top)
    return release_shows_top

def visualize_release_year_show(release_shows_top):
    plt.figure(figsize=(14,8))
    sns.barplot(data=release_shows_top,
            x="title",
            y="imdb_score",
            hue="country")
    plt.title("Top 25 Film/Show Between 2010 and 2024 years by IMDB Score")
    plt.xlabel("Title")
    plt.xticks(rotation=90)
    plt.ylabel("IMDB Score")
    plt.show()


def under_1980(netflix):
    oldest_years = netflix[netflix["release_year"]<=1980]
    oldest = oldest_years[["title", "type","main_genres","country","imdb_score"]].sort_values(by="imdb_score",
                                                                                              ascending=False)
    oldest_top = oldest.head(25) 
    print(oldest_top)
    return oldest_top

def visualize_oldest(oldest_top):
    plt.figure(figsize=(14,8))
    sns.barplot(data=oldest_top,
            x="imdb_score",
            y="title",
            hue="country")
    plt.title("Top 25 Film/Show Under 1980 years by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.ylabel("Title")
    plt.show()
   
def history_genre(netflix):
    history_genres = netflix[netflix["main_genres"]=="history"]
    history = history_genres[["title", "country", "release_year", "type",
                              "imdb_score", "age_certification"]].sort_values(by="imdb_score",ascending=False)
    top_history = history.head(20)
    print(top_history)
    return top_history

def visualize_histor(top_history):
    plt.figure(figsize=(14,8))
    sns.barplot(data=top_history,
            x="imdb_score",
            y="title",
            hue="country")
    plt.title("Top 20 History Film/Show by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.ylabel("Title")
    plt.show()   


def animation_genre(netflix):
    animation = netflix[(netflix["main_genres"]=="animation") &
                         (netflix["age_certification"]=="All Children")]
    animation_detail = animation[["title", "country", "release_year", "imdb_score",
                                  "imdb_votes", "type"]].sort_values(by="imdb_score",
                                                                     ascending=False)
    print(animation_detail)
    return animation_detail

def visualize_animation(animation_detail):
    animation_top = animation_detail.head(20)
    plt.figure(figsize=(14,8))
    sns.barplot(data=animation_top,
            x="imdb_score",
            y="title",
            hue="country")
    plt.title("Top 20 Animation Film/Show for All Children by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.ylabel("Title")
    plt.show()      

def only_adult(netflix):
    adults = netflix[netflix["age_certification"]=="Adults Only"]
    adults_detail = adults[["title", "country", "release_year", "imdb_score",
                             "main_genres", "imdb_votes", "type"]].sort_values(by="imdb_score",
                                                                     ascending=False)
    print(adults_detail)
    return adults_detail

def visualize_adults(adults_detail):
    plt.figure(figsize=(14,8))
    sns.barplot(data=adults_detail,
            x="imdb_score",
            y="title",
            hue="country")
    plt.title("Film/Show Category for 'Only Adults' by IMDB Score")
    plt.xlabel("IMDB Score")
    plt.ylabel("Title")
    plt.show()     

def relationship_imdbs(netflix):
    plt.figure(figsize=(14,7))
    sns.scatterplot(data=netflix,
                    x="imdb_score",
                    y="imdb_votes",
                    hue="type")
    plt.title("Relationship of IMDB Votes vs. IMDB Score")
    plt.ylabel("IMDB Votes")
    plt.xlabel("IMDB Score")
    plt.show()

def corr_imdbs(netflix):
    # Calculate the correlation between IMDb votes and IMDb score
    correlation = netflix['imdb_votes'].corr(netflix['imdb_score'])
    print(f"Correlation between IMDb Votes and IMDb Score: {correlation}")

def distrubution_imdbs(netflix):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    sns.histplot(netflix['imdb_score'], kde=True, bins=20, color='skyblue')
    plt.title("Distribution of IMDb Scores", fontsize=16)
# Plot distribution of IMDb Votes
    plt.subplot(1, 2, 2)
    sns.histplot(netflix['imdb_votes'], kde=True, bins=20, color='salmon')
    plt.title("Distribution of IMDb Votes", fontsize=16)
    plt.tight_layout()
    plt.show()

def scaling(netflix):
    scaler = MinMaxScaler()
    netflix[['scaled_imdb_votes', 'scaled_imdb_score']] = scaler.fit_transform(netflix[['imdb_votes', 'imdb_score']])
# Now, plot the scaled version of both
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x='scaled_imdb_votes', y='scaled_imdb_score', data=netflix, alpha=0.6)
    plt.title("Scaled IMDb Score vs Scaled IMDb Votes", fontsize=16)
    plt.xlabel("Scaled IMDb Votes", fontsize=12)
    plt.ylabel("Scaled IMDb Score", fontsize=12)
    plt.show()


#feature engineering 
#For the three categorical variables that we have now, I create dummy variable, to represent them as 1s and 0

def dummies(netflix):
    title_column = netflix['title']
    netflix.set_index("title", inplace=True)
    dummy = pd.get_dummies(netflix[["type", "main_genres", "country"]], drop_first=True)
    netflix_dum = pd.concat([dummy, netflix], axis=1)
    netflix_dum.drop(["type", "main_genres","country"], axis=1, inplace=True)
    netflix_dum.index = title_column
    print(netflix_dum)
    return netflix_dum

def scalings(netflix_dum):
    scalers = MinMaxScaler()
    # Select only numeric columns (exclude non-numeric like index or text columns)
    numeric_columns = netflix_dum.select_dtypes(include=['float64', 'int64']).columns
    scaled = scalers.fit_transform(netflix_dum[numeric_columns])
    scaled_df = pd.DataFrame(scaled, columns=numeric_columns, index=netflix_dum.index)
    print("\nAfter scaling:")
    print(scaled_df.describe().T)   
    scaled_df.fillna(0, inplace=True)  # Fill NaN values with 0 (or use another method)
    # Combine the scaled columns with the one-hot encoded columns
    final_df = pd.concat([scaled_df, netflix_dum.drop(columns=scaled_df.columns)], axis=1)
# Print the final dataframe
    print("\nFinal DataFrame (scaled + one-hot encoded columns):")
    print(final_df.head())
    print(final_df.shape)
    return final_df

#decrease dimensions for clustering with PCA
# Apply PCA and return explained variance
def apply_pca(scaled_df, n_components=3):
    # Ensure only numeric columns are used for PCA
    numeric_columns = scaled_df.select_dtypes(include=['float64', 'int64']).columns
    scaled_df_numeric = scaled_df[numeric_columns]
    pca = PCA(n_components=n_components)  # If None, PCA will keep all components
    pca_result = pca.fit_transform(scaled_df_numeric)    
    # Create a DataFrame for the PCA result
    pca_df = pd.DataFrame(pca_result, index=scaled_df_numeric.index)   
    # Print explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", pca.explained_variance_ratio_.cumsum())
    # Print the PCA result
    print("\nPCA Result DataFrame:")
    print(pca_df.head()) 
    return pca, pca_df

def evaluate_clusters(pca_df, min_clusters=2, max_clusters=12):
    silhouette_scores = []  
    # Iterate over the number of clusters (from min_clusters to max_clusters)
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_df)
        
        # Calculate silhouette score
        score = silhouette_score(pca_df, cluster_labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score}")
    
    # Plot the silhouette scores for each cluster count
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title('Silhouette Scores for Different Cluster Counts')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.grid(True)
    plt.show()

def clustering(pca_df):
    # Applying K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, init='k-means++', max_iter=500, n_init=20)  # Set number of clusters to 4, bc acc. to silhouette skor
    kmeans.fit(pca_df)
# Add the cluster labels to the pca_df
    pca_df['Cluster'] = kmeans.labels_
# Check the cluster means for relevant features
    cluster_profiles = pca_df.groupby('Cluster').mean()
    print("Cluster Profiles (mean values for each cluster):")
    print(cluster_profiles)
       # Cluster centers (the centroids of each cluster)
    cluster_centers = kmeans.cluster_centers_
    print("Cluster Centers:")
    print(cluster_centers)
    print(pca_df.head())
    return pca_df, kmeans, cluster_centers

def visualize_clustering(pca_df, kmeans, cluster_centers):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_df[0], pca_df[1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result (Clustering) with Cluster Centers')
    plt.colorbar(scatter, label='Cluster w')
    plt.show()


def main():
    # Load and preprocess the data
    netflix = load_and_preprocess_data()

    # Visualize type distribution
    visualize_type(netflix)

    # Process and visualize genres
    process_genres(netflix)
    count_genres(netflix)

    # Visualize country distribution
    process_and_visualize_countries(netflix)

    # Process and visualize age certification
    process_age_certification(netflix)
    visualize_age_certification(netflix)
    
    #checking data recently
    netflix = check_data(netflix)

    # Apply Winsorization to TMDB popularity and visualize
    winsorize_tmdb_popularity(netflix)

    # Apply log transformation to IMDB votes
    log_transform_imdb_votes(netflix)

    # Visualize IMDB vs TMDB popularity
    visualize_imdb_vs_tmdb(netflix)

    # Visualize the correlation matrix
    visualize_correlation_matrix(netflix)

    # Get top movies by IMDB votes and visualize
    top_15_imdb_movie = top_movies_by_imdb(netflix)
    visualize_top_15_movies_by_imdb(top_15_imdb_movie)

    # Get long-running shows and visualize
    long_shows_data = long_shows(netflix)
    visualize_long_shows(long_shows_data)

    # Get and visualize the top 15 Romance/Drama shows
    comedy_or_drama_data = comedy_or_drama(netflix)
    visualize_rom_drama(comedy_or_drama_data)

    #turkiye shows
    turkiye_shows_1 = turkiye_shows(netflix)
    turkiye_visualize(turkiye_shows_1)

    least_turkiye_1 = least_turkiye_shows(netflix)
    least_turkiye(least_turkiye_1)

    mean = mean_imdb_all_genres(netflix)
    visualize_mean_imdb_all_genre(mean)

    mean_imdb_usa_data = usa_detail(netflix)
    usa_visualize(mean_imdb_usa_data)

    runtime = avg_runtime(netflix)

    distrubiton_runtime = hist_runtime(netflix)
    distrubiton_releaseyear = hist_release_year(netflix)
    distrubiton_tmdb = hist_tmdb_score(netflix)

    comedy = comedy_genre(netflix)
    visualize_comedy(comedy)

    romance = romance_genre(netflix)
    visualize_romance(romance)

    drama_action = drama_and_action(netflix)
    visaulize_drama_action(drama_action)

    documenatation_crime_det = documentation_crime(netflix)
    visualize_doc_crime(documenatation_crime_det)

    korean_south = korean(netflix)
    korean_visualize(korean_south)

    romance_korean = romance_korea(netflix)
    visualize_romance_korea(romance_korean)

    long_seasons = long_season(netflix)

    some_seasons = between_seasons(netflix)
    visualize_between_seasons(some_seasons)

    season8= eight_season(netflix)
    visualize_eight_season(season8)

    releaseshow = release_year_detail(netflix)
    visualize_release_year_show(releaseshow)

    underyears1980=under_1980(netflix)
    visualize_oldest(underyears1980)

    hist_genre = history_genre(netflix)
    visualize_histor(hist_genre)

    animation_genres=animation_genre(netflix)
    visualize_animation(animation_genres)

    adult = only_adult(netflix)
    visualize_adults(adult)

#checking relationships between imdbs
    relationship_imdbs(netflix)
    correlation_imdbs= corr_imdbs(netflix)
    distrubution_imdbs(netflix)
    scaling(netflix) #after scaling? not sure ask to teacher

#feature engineering
    netflix_dum = dummies(netflix)
    scaled_df = scalings(netflix_dum)
    pca, pca_df = apply_pca(scaled_df, n_components=3)

    evaluate_clusters(pca_df, min_clusters=2, max_clusters=10)

    pca_df, kmeans, cluster_centers = clustering(pca_df)  
    visualize_clustering(pca_df, kmeans, cluster_centers) 


if __name__ == "__main__":
    main()
